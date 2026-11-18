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

"""Fixed-input diagnostics at every original Torch sparse attention call.

Returns the original kernel output unchanged. Alternatives never feed subsequent
layers, isolating arithmetic effects from accumulated network/routing changes.
The online candidate models the BF16 Triton online softmax; none
of these diagnostics validate the released hand-written backward.
"""

import math

import torch
import torch.nn.functional as F


def difference(actual, reference):
    delta = actual.double() - reference.double()
    return dict(
        max_diff=float(delta.abs().max()),
        mean_diff=float(delta.abs().mean()),
        rmse=float(delta.square().mean().sqrt()),
    )


@torch.no_grad()
def fixed_input_attention(q, k, v, indices, block_size, mode, query_chunk=4):
    if mode not in ("math", "qkv-bf16", "online-bf16"):
        raise ValueError(f"Unknown diagnostic mode {mode}")
    batch, length, heads, dim = k.shape
    blocks, groups = length // block_size, q.shape[2] // heads
    out = torch.empty_like(q)
    scale = dim**-0.5
    for head in range(heads):
        kb = k[:, :, head].reshape(batch, blocks, block_size, dim)
        vb = v[:, :, head].reshape(batch, blocks, block_size, dim)
        for start in range(0, blocks, query_chunk):
            stop = min(blocks, start + query_chunk)
            choice = indices[:, start:stop, head]
            row = torch.arange(batch, device=q.device)[:, None, None]
            keys = kb[row, choice].reshape(batch * (stop - start), -1, dim)
            values = vb[row, choice].reshape_as(keys)
            query = q[
                :,
                start * block_size : stop * block_size,
                head * groups : (head + 1) * groups,
            ]
            query = query.reshape(
                batch * (stop - start), block_size, groups, dim
            ).transpose(1, 2)
            if mode == "math":
                result = F.scaled_dot_product_attention(
                    query, keys[:, None], values[:, None], enable_gqa=True
                )
            else:
                query = ((query.float() * scale) * 1.44269504089).bfloat16().float()
                keys, values = keys.bfloat16().float(), values.bfloat16().float()
                if mode == "qkv-bf16":
                    # Query already includes scale * log2(e); undo log2(e)
                    # only after the dot product, with no probability cast.
                    scores = (query @ keys[:, None].transpose(-1, -2)) * math.log(2)
                    result = scores.softmax(dim=-1) @ values[:, None]
                else:
                    maximum = torch.full(
                        (*query.shape[:-1], 1), -float("inf"), device=q.device
                    )
                    denominator = torch.zeros_like(maximum)
                    accumulator = torch.zeros_like(query, dtype=torch.float32)
                    for selected in range(indices.shape[-1]):
                        sl = slice(selected * block_size, (selected + 1) * block_size)
                        scores = query @ keys[:, None, sl].transpose(-1, -2)
                        new_maximum = torch.maximum(
                            maximum, scores.amax(dim=-1, keepdim=True)
                        )
                        rescale = torch.exp2(maximum - new_maximum)
                        probs = torch.exp2(scores - new_maximum)
                        denominator = denominator * rescale + probs.sum(
                            dim=-1, keepdim=True
                        )
                        accumulator = (
                            accumulator * rescale
                            + probs.bfloat16().float() @ values[:, None, sl]
                        )
                        maximum = new_maximum
                    result = accumulator / denominator
            out[
                :,
                start * block_size : stop * block_size,
                head * groups : (head + 1) * groups,
            ] = result.transpose(1, 2).reshape(
                batch, (stop - start) * block_size, groups, dim
            )
    return out


def install_diagnostics(sensitivity=False):
    import primitives

    original = primitives.mosaic_sparse_attn
    records = []

    @torch.no_grad()
    def diagnose(q, k, v, indices, block_size):
        reference = original(q, k, v, indices, block_size)
        record = dict(
            call=len(records),
            q_shape=list(q.shape),
            kv_shape=list(k.shape),
            selected_blocks=indices.shape[-1],
            block_size=block_size,
            modes={},
        )
        for mode in ("math", "qkv-bf16", "online-bf16"):
            actual = fixed_input_attention(q, k, v, indices, block_size, mode)
            record["modes"][mode] = difference(actual, reference)
            if not bool(torch.isfinite(actual).all()):
                raise ValueError(
                    f"Non-finite diagnostic call {len(records)}, mode {mode}"
                )
            del actual
        if sensitivity:
            # A one-ULP perturbation is a diagnostic, not a claim that the two
            # frameworks differ in precisely this direction at every element.
            shifted = torch.nextafter(q, torch.full_like(q, float("inf")))
            native_shifted = original(shifted, k, v, indices, block_size)
            sensitivity_record = dict(
                perturbation="each Q element moved one FP32 ULP toward +infinity; K/V/routes unchanged",
                q_max_change=float((shifted - q).abs().max()),
                native_output=difference(native_shifted, reference),
            )
            del native_shifted
            scale = q.shape[-1] ** -0.5
            before = ((q * scale) * 1.44269504089).bfloat16()
            after = ((shifted * scale) * 1.44269504089).bfloat16()
            sensitivity_record["q_bf16_bins_changed"] = int((before != after).sum())
            sensitivity_record["q_elements"] = q.numel()
            del before, after
            math_base = fixed_input_attention(q, k, v, indices, block_size, "math")
            math_shifted = fixed_input_attention(
                shifted, k, v, indices, block_size, "math"
            )
            sensitivity_record["math_output"] = difference(math_shifted, math_base)
            record["one_ulp_sensitivity"] = sensitivity_record
            del math_base, math_shifted, shifted
        records.append(record)
        print(f"Sparse diagnostic call {record['call']}: {record['modes']}", flush=True)
        if sensitivity:
            print(
                f"One-ULP sensitivity {record['call']}: {sensitivity_record}",
                flush=True,
            )
        return reference

    primitives.mosaic_sparse_attn = diagnose
    return records
