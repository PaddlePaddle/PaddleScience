import math

import paddle
from paddle import nn

import ppsci.contrib.neuralop.mpu.comm as comm
from ppsci.contrib.neuralop.mpu.mappings import gather_from_model_parallel_region
from ppsci.contrib.neuralop.mpu.mappings import scatter_to_model_parallel_region


class MultigridPatching2D(nn.Layer):
    def __init__(
        self,
        model,
        levels=0,
        padding_fraction=0,
        use_distributed=False,
        stitching=True,
    ):
        """Wraps a model inside a multi-grid patching"""
        super().__init__()

        self.skip_padding = (padding_fraction is None) or (padding_fraction <= 0)

        self.levels = levels

        if isinstance(padding_fraction, (float, int)):
            padding_fraction = [padding_fraction, padding_fraction]
        self.padding_fraction = padding_fraction

        n_patches = 2**levels
        if isinstance(n_patches, int):
            n_patches = [n_patches, n_patches]
        self.n_patches = n_patches

        self.model = model

        self.use_distributed = use_distributed
        self.stitching = stitching

        if levels > 0:
            print(
                "MGPatching("
                f"n_patches={self.n_patches}, "
                f"padding_fraction={self.padding_fraction}, "
                f"levels={self.levels}, "
                f"use_distributed={use_distributed}, "
                f"stitching={stitching})"
            )

        # If distributed patches are stiched, re-scale gradients to revert DDP averaging
        if self.use_distributed and self.stitching:
            for param in model.parameters():
                param.register_hook(
                    lambda grad: grad * float(comm.get_model_parallel_size())
                )

    def patch(self, x, y):
        # If not stitching, scatter truth, otherwise keep on every GPU
        if self.use_distributed and not self.stitching:
            y = make_patches(y, n=self.n_patches, p=0)
            y = scatter_to_model_parallel_region(y, 0)

        # Create padded patches in batch dimension (identity if levels=0)
        x = self._make_mg_patches(x)

        # Split data across processes
        if self.use_distributed:
            x = scatter_to_model_parallel_region(x, 0)

        return x, y

    def unpatch(self, x, y, evaluation=False):
        """Always stitch during evaluation"""
        if self.skip_padding:
            return x, y

        # Remove padding in the output
        if self.padding_height > 0 or self.padding_width > 0:
            x = self._unpad(x)

        # Gather patches if they are to be stitched back together
        if self.use_distributed and self.stitching:
            x = gather_from_model_parallel_region(x, dim=0)
        else:
            x = x

        # Stich patches or patch the truth if output left unstitched
        if self.stitching or evaluation:
            x = self._stitch(x)

        return x, y

    def _stitch(self, x):
        if self.skip_padding:
            return x

        # Only 1D and 2D supported
        assert x.ndim == 4, f"Only 2D patch supported but got input with {x.ndim} dims."

        if self.n_patches[0] <= 1 and self.n_patches[1] <= 1:
            return x

        # Size with padding removed
        size = x.size()

        # if self.mode == "batch-wise":
        B = size[0] // (self.n_patches[0] * self.n_patches[1])
        W = size[3] * self.n_patches[1]

        C = size[1]
        H = size[2] * self.n_patches[0]

        # Reshape
        x = x.transpose([0, 3, 2, 1])
        x = x.reshape([B, self.n_patches[0], self.n_patches[1], size[3], size[2], C])
        x = x.transpose([0, 5, 1, 4, 2, 3])
        x = x.reshape([B, C, H, W])

        return x

    def _make_mg_patches(self, x):
        levels = self.levels
        if levels <= 0:
            return x

        _, _, height, width = x.shape
        padding = [
            int(round(v))
            for v in [
                height * self.padding_fraction[0],
                width * self.padding_fraction[1],
            ]
        ]
        self.padding_height = padding[0]
        self.padding_width = padding[1]

        patched = make_patches(x, n=2**self.levels, p=padding)

        s1_patched = patched.size(-2) - 2 * padding[0]
        s2_patched = patched.size(-1) - 2 * padding[1]

        for level in range(1, levels + 1):
            sub_sample = 2**level
            s1_stride = s1_patched // sub_sample
            s2_stride = s2_patched // sub_sample

            x_sub = x[:, :, ::sub_sample, ::sub_sample]

            s2_pad = (
                math.ceil(
                    (s2_patched + (2**levels - 1) * s2_stride - x_sub.size(-1)) / 2.0
                )
                + padding[1]
            )
            s1_pad = (
                math.ceil(
                    (s1_patched + (2**levels - 1) * s1_stride - x_sub.size(-2)) / 2.0
                )
                + padding[0]
            )

            if s2_pad > x_sub.size(-1):
                diff = s2_pad - x_sub.size(-1)
                x_sub = paddle.nn.functional.pad(
                    x_sub, pad=[x_sub.size(-1), x_sub.size(-1), 0, 0], mode="circular"
                )
                x_sub = paddle.nn.functional.pad(
                    x_sub, pad=[diff, diff, 0, 0], mode="circular"
                )
            else:
                x_sub = paddle.nn.functional.pad(
                    x_sub, pad=[s2_pad, s2_pad, 0, 0], mode="circular"
                )

            if s1_pad > x_sub.size(-2):
                diff = s1_pad - x_sub.size(-2)
                x_sub = paddle.nn.functional.pad(
                    x_sub, pad=[0, 0, x_sub.size(-2), x_sub.size(-2)], mode="circular"
                )
                x_sub = paddle.nn.functional.pad(
                    x_sub, pad=[0, 0, diff, diff], mode="circular"
                )
            else:
                x_sub = paddle.nn.functional.pad(
                    x_sub, pad=[0, 0, s1_pad, s1_pad], mode="circular"
                )

            x_sub = x_sub.unfold(-1, s2_patched + 2 * padding[1], s2_stride)
            x_sub = x_sub.unfold(-3, s1_patched + 2 * padding[0], s1_stride)

            x_sub = x_sub.transpose([0, 2, 3, 4, 5, 1])
            x_sub = x_sub.reshape(
                [
                    patched.size(0),
                    s2_patched + 2 * padding[1],
                    s1_patched + 2 * padding[0],
                    -1,
                ]
            )
            x_sub = x_sub.transpose([0, 3, 2, 1])

            patched = paddle.concat((patched, x_sub), 1)

        return patched

    def _unpad(self, x):
        return x[
            ...,
            self.padding_height : -self.padding_height,
            self.padding_width : -self.padding_width,
        ]


# x : (batch, C, s) or (batch, C, h, w)
# y : (n*batch, C, s/n + 2p) or (n1*n2*batch, C, h/n1 + 2*p1, w/n2 + 2*p2)
def make_patches(x, n, p=0):

    size = x.size()

    # Only 1D and 2D supported
    assert len(size) == 3 or len(size) == 4

    if len(size) == 3:
        d = 1
    else:
        d = 2

    if isinstance(p, int):
        p = [p, p]

    # Pad
    if p[0] > 0 or p[1] > 0:
        if d == 1:
            x = paddle.nn.functional.pad(x, pad=p, mode="circular")
        else:
            x = paddle.nn.functional.pad(
                x, pad=[p[1], p[1], p[0], p[0]], mode="circular"
            )

    if isinstance(n, int):
        n = [n, n]

    if n[0] <= 1 and n[1] <= 1:
        return x

    # Patches must be equally sized
    for j in range(d):
        assert size[-(j + 1)] % n[-(j + 1)] == 0

    # Patch
    for j in range(d):
        patch_size = size[-(j + 1)] // n[-(j + 1)]
        x = x.unfold(-(2 * j + 1), patch_size + 2 * p[-(j + 1)], patch_size)

    x = x.transpose([0, 2, 3, 4, 5, 1])
    x = x.reshape(
        [
            size[0] * n[0] * n[1],
            size[-1] // n[-1] + 2 * p[-1],
            size[-2] // n[-2] + 2 * p[-2],
            size[1],
        ]
    )
    x = x.transpose([0, 3, 2, 1])

    return x
