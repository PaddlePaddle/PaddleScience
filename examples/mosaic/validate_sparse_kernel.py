# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Isolate the released Triton sparse forward's BF16 online-softmax semantics.

Run --framework torch then --framework paddle with the same --output directory.
This is an inference diagnostic; autograd through casts does not reproduce the
reference's hand-written backward and must not be used for training validation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def quantized_qkv_attention(
    q, k, v, indices, block_size, query_chunk_size=8, recompute_chunks=False
):
    """Diagnostic: source Q/K/V rounding, joint FP32 softmax, no P rounding."""
    import paddle
    from einops import rearrange

    if recompute_chunks or paddle.is_grad_enabled():
        raise RuntimeError("QKV quantization diagnostic requires inference no_grad")
    with paddle.amp.auto_cast(enable=False):
        batch, length, heads, dim = k.shape
        blocks = length // block_size
        count = indices.shape[-1]
        offsets = (
            paddle.arange(batch * heads, dtype="int64").reshape([batch, heads, 1, 1])
            * blocks
        )
        choices = indices.transpose([0, 2, 1, 3]) + offsets
        banks = [
            rearrange(
                t.astype("bfloat16").astype("float32"),
                "b (n s) h d -> (b h n) s d",
                s=block_size,
            )
            for t in (k, v)
        ]
        query = (
            ((q.astype("float32") * dim**-0.5) * 1.44269504089)
            .astype("bfloat16")
            .astype("float32")
        )
        query = rearrange(
            query, "b (n s) (h g) d -> b h n g s d", s=block_size, h=heads
        )
        outputs = []
        for start in range(0, blocks, query_chunk_size):
            stop = min(start + query_chunk_size, blocks)
            rows = choices[:, :, start:stop].flatten()
            keys, values = [
                paddle.gather(bank, rows)
                .reshape([batch, heads, stop - start, count * block_size, dim])
                .unsqueeze(3)
                for bank in banks
            ]
            scores = paddle.matmul(
                query[:, :, start:stop], keys, transpose_y=True
            ) * np.log(2)
            outputs.append(
                paddle.matmul(paddle.nn.functional.softmax(scores, axis=-1), values)
            )
        return rearrange(
            paddle.concat(outputs, axis=2), "b h n g s d -> b (n s) (h g) d"
        ).astype(q.dtype)


def online_sparse_attention(
    q, k, v, indices, block_size, query_chunk_size=8, recompute_chunks=False
):
    if recompute_chunks:
        raise ValueError("Native emulation does not support training recomputation")
    import paddle

    if paddle.is_grad_enabled():
        raise RuntimeError("Triton forward emulation is inference-only; use no_grad")
    # Outer model autocast must not silently reduce the FP32 accumulators/GEMMs.
    with paddle.amp.auto_cast(enable=False):
        return _online_sparse_attention(q, k, v, indices, block_size, query_chunk_size)


def _online_sparse_attention(q, k, v, indices, block_size, query_chunk_size):
    import paddle
    from einops import rearrange

    batch, length, heads, dim = k.shape
    blocks = length // block_size
    count = indices.shape[-1]
    offsets = (
        paddle.arange(batch * heads, dtype="int64").reshape([batch, heads, 1]) * blocks
    )
    choices = indices.transpose([0, 2, 1, 3])
    banks = [rearrange(t, "b (n s) h d -> (b h n) s d", s=block_size) for t in (k, v)]
    # Preserve both multiplications and their FP32 rounding before the BF16 cast.
    queries = (
        ((q.astype("float32") * (dim**-0.5)) * 1.44269504089)
        .astype("bfloat16")
        .astype("float32")
    )
    queries = rearrange(
        queries, "b (n s) (h g) d -> b h n g s d", s=block_size, h=heads
    )
    outputs = []
    for start in range(0, blocks, query_chunk_size):
        stop = min(start + query_chunk_size, blocks)
        query = queries[:, :, start:stop]
        maximum = paddle.full(
            [batch, heads, stop - start, query.shape[3], block_size, 1], -float("inf")
        )
        denominator = paddle.zeros_like(maximum)
        accumulator = paddle.zeros([*maximum.shape[:-1], dim], dtype="float32")
        for choice in range(count):
            rows = (choices[:, :, start:stop, choice] + offsets).flatten()
            keys, values = [
                paddle.gather(bank, rows)
                .reshape([batch, heads, stop - start, block_size, dim])
                .astype("bfloat16")
                .astype("float32")
                .unsqueeze(3)
                for bank in banks
            ]
            scores = paddle.matmul(query, keys, transpose_y=True)
            new_maximum = paddle.maximum(maximum, scores.max(axis=-1, keepdim=True))
            rescale = 2.0 ** (maximum - new_maximum)
            probabilities = 2.0 ** (scores - new_maximum)
            denominator = denominator * rescale + probabilities.sum(
                axis=-1, keepdim=True
            )
            accumulator = accumulator * rescale + paddle.matmul(
                probabilities.astype("bfloat16").astype("float32"), values
            )
            maximum = new_maximum
        outputs.append(accumulator / denominator)
    return rearrange(
        paddle.concat(outputs, axis=2), "b h n g s d -> b (n s) (h g) d"
    ).astype(q.dtype)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    fixture = args.output / "kernel.npz"
    if args.framework == "torch":
        import torch

        if args.source is None:
            raise ValueError("--source is required for the Torch reference")
        sys.path.insert(0, str(args.source.resolve()))
        from ops import mosaic_sparse_attn

        torch.backends.cuda.matmul.allow_tf32 = False
        rng = np.random.default_rng(2026)
        arrays = {
            name: rng.standard_normal(shape).astype("float32")
            for name, shape in (
                ("q", (1, 512, 4, 64)),
                ("k", (1, 512, 2, 64)),
                ("v", (1, 512, 2, 64)),
            )
        }
        indices = np.empty((1, 4, 2, 3), dtype="int64")
        for block in range(4):
            for head in range(2):
                indices[0, block, head] = rng.choice(4, 3, replace=False)
        with torch.no_grad():
            result = (
                mosaic_sparse_attn(
                    *[
                        torch.tensor(arrays[name], device="cuda:0")
                        for name in ("q", "k", "v")
                    ],
                    torch.tensor(indices, device="cuda:0"),
                    128,
                )
                .cpu()
                .numpy()
            )
        np.savez(fixture, **arrays, indices=indices, reference=result)
        print(f"Saved original Triton forward: {fixture}")
        return

    import paddle
    from compare_traces import differences

    from ppsci.arch.mosaic_attention import sparse_attention

    paddle.set_device(args.device)
    arrays = np.load(fixture)
    q, k, v, indices = [
        paddle.to_tensor(arrays[name]) for name in ("q", "k", "v", "indices")
    ]
    size = int(arrays["block_size"]) if "block_size" in arrays else 128
    with paddle.no_grad(), paddle.amp.auto_cast(enable=False):
        emulated = online_sparse_attention(q, k, v, indices, size).numpy()
        math = sparse_attention(q, k, v, indices, size).numpy()
    report = dict(
        native_emulation=differences(emulated, arrays["reference"]),
        mathematical=differences(math, arrays["reference"]),
        scope="isolated forward only; FP32 GEMM on BF16-rounded operands; not backward validation",
    )
    np.savez(args.output / "paddle.npz", emulated=emulated, math=math)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("torch", "paddle"), required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="gpu:0")
    main(parser.parse_args())
