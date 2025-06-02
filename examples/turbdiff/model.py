"""
TurbDiff model implementation for PaddleScience.
Based on the paper "From Zero to Turbulence: Generative Modeling for 3D Flow Simulation" 
by Marten Lienen, David Lüdke, Jan Hansen-Palmus, and Stephan Günnemann.
"""

import math
from dataclasses import dataclass
from functools import partial
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


@dataclass
class ModelPrediction:
    """Model prediction output container."""
    noise: paddle.Tensor
    x_start: paddle.Tensor
    mean: paddle.Tensor
    log_var: paddle.Tensor


# Small helper modules

class Residual(nn.Layer):
    """Residual connection wrapper for a layer."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def pad_to_multiple_of(x: paddle.Tensor, n: int, *, mode: str):
    """Pad tensor to be a multiple of n in each spatial dimension."""
    h, w, d = x.shape[-3:]
    h_pad = n - h % n if h % n != 0 else 0
    w_pad = n - w % n if w % n != 0 else 0
    d_pad = n - d % n if d % n != 0 else 0
    
    if min(h_pad, w_pad, d_pad) > 0:
        return F.pad(x, [0, d_pad, 0, w_pad, 0, h_pad], mode=mode), (
            h_pad,
            w_pad,
            d_pad,
        )
    else:
        return x, (0, 0, 0)


def unpad(x: paddle.Tensor, padding):
    """Remove padding from tensor."""
    h_pad, w_pad, d_pad = padding
    if min(padding) > 0:
        return x[..., :-h_pad if h_pad > 0 else None, 
                 :-w_pad if w_pad > 0 else None, 
                 :-d_pad if d_pad > 0 else None]
    else:
        return x


class PreNorm(nn.Layer):
    """Apply normalization before a function."""
    def __init__(self, norm: nn.Layer, fn: nn.Layer):
        super().__init__()
        self.norm = norm
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


# Sinusoidal positional embeddings

class SinusoidalPosEmb(nn.Layer):
    """Sinusoidal positional embedding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = paddle.exp(paddle.arange(half_dim, dtype=paddle.float32) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = paddle.concat([paddle.sin(emb), paddle.cos(emb)], axis=-1)
        return emb


class NyquistFrequencyEmbedding(nn.Layer):
    """
    Sine-cosine embedding for timesteps that scales from 1/8 to a (< 1) multiple of
    the Nyquist frequency.
    """
    def __init__(self, dim: int, timesteps: int):
        super().__init__()
        assert dim % 2 == 0

        T = timesteps
        k = dim // 2

        # Nyquist frequency for T samples per cycle
        nyquist_frequency = T / 2

        golden_ratio = (1 + np.sqrt(5)) / 2
        frequencies = np.geomspace(1 / 8, nyquist_frequency / (2 * golden_ratio), num=k)

        # Sample every frequency twice, once shifted by pi/2 to get cosine
        scale = np.repeat(2 * np.pi * frequencies / timesteps, 2)
        bias = np.tile(np.array([0, np.pi / 2]), k)

        self.scale = paddle.to_tensor(scale, dtype=paddle.float32)
        self.bias = paddle.to_tensor(bias, dtype=paddle.float32)

    def forward(self, t):
        # paddle equivalent of torch.addcmul
        phase = self.bias + self.scale * t[..., None]
        return paddle.sin(phase)


# Building block modules

class Block(nn.Layer):
    """Basic convolutional block with normalization and activation."""
    def __init__(
        self,
        dim,
        dim_out,
        actfn,
        norm_klass=None,
    ):
        super().__init__()
        self.conv = nn.Conv3D(dim, dim_out, 3, padding=1, padding_mode='replicate')
        self.norm = norm_klass(dim_out)
        self.act = actfn()

    def forward(self, x, scale_shift=None):
        x = self.conv(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = shift + (scale + 1) * x

        x = self.act(x)
        return x


class ResnetBlock(nn.Layer):
    """Residual block with conditioning."""
    def __init__(self, dim_in, dim_out, *, c_dim: int, actfn, norm_klass):
        super().__init__()

        self.project_onto_scale_shift = nn.Linear(c_dim, dim_out * 2)

        self.block1 = Block(dim_in, dim_out, actfn=actfn, norm_klass=norm_klass)
        self.block2 = Block(dim_out, dim_out, actfn=actfn, norm_klass=norm_klass)
        self.conv = nn.Conv3D(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, c):
        # Reshape conditioning output
        c = self.project_onto_scale_shift(c)
        c = paddle.reshape(c, shape=[*c.shape[:-1], c.shape[-1], 1, 1, 1])
        scale, shift = paddle.split(c, 2, axis=-4)

        h = self.block1(x, scale_shift=(scale, shift))
        h = self.block2(h)

        return h + self.conv(x)


class LinearAttention(nn.Layer):
    """Linear attention mechanism."""
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3D(dim, hidden_dim * 3, 1, bias_attr=False)

        self.to_out = nn.Sequential(
            nn.Conv3D(hidden_dim, dim, 1),
            nn.GroupNorm(1, dim)
        )

    def forward(self, x):
        b, c, h, w, d = x.shape
        qkv = self.to_qkv(x).chunk(3, axis=1)
        q, k, v = map(lambda t: paddle.reshape(
            t, [b, self.heads, -1, h * w * d]), qkv)

        q = q * self.scale

        k = paddle.transpose(k, [0, 1, 3, 2])
        context = paddle.matmul(k, v)

        out = paddle.matmul(context, q)
        out = paddle.reshape(out, [b, self.heads, -1, h, w, d])
        out = paddle.transpose(out, [0, 1, 2, 3, 4, 5])
        out = paddle.reshape(out, [b, -1, h, w, d])
        return self.to_out(out)


class Attention(nn.Layer):
    """Standard attention mechanism."""
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv3D(dim, hidden_dim * 3, 1, bias_attr=False)
        self.to_out = nn.Conv3D(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w, d = x.shape
        qkv = self.to_qkv(x).chunk(3, axis=1)
        q, k, v = map(lambda t: paddle.reshape(
            t, [b, self.heads, -1, h * w * d]), qkv)

        # Transpose for attention dot product
        q = paddle.transpose(q, [0, 1, 3, 2])
        v = paddle.transpose(v, [0, 1, 3, 2])

        # Scale dot-product attention
        dots = paddle.matmul(q, k) * (k.shape[-1] ** -0.5)
        attn = F.softmax(dots, axis=-1)
        out = paddle.matmul(attn, v)
        
        out = paddle.transpose(out, [0, 1, 3, 2])
        out = paddle.reshape(out, [b, -1, h, w, d])
        
        return self.to_out(out)


# U-Net model

class UNet(nn.Layer):
    """
    A general U-Net structure with interpolation instead of max-pool and transposed
    convolutions.
    """
    def __init__(
        self,
        downsampling_blocks,
        upsampling_blocks,
        center_block,
        *,
        downsampling_factor=2.0,
    ):
        super().__init__()

        assert len(downsampling_blocks) == len(upsampling_blocks)

        self.downsampling_blocks = nn.LayerList(downsampling_blocks)
        self.upsampling_blocks = nn.LayerList(upsampling_blocks)
        self.center_block = center_block
        self.downsampling_factor = downsampling_factor
        self.scale_factor = 1 / downsampling_factor

    def forward(self, x, *args, **kwargs):
        h = [x]

        # Downsample
        for block in self.downsampling_blocks:
            h.append(block(h[-1], *args, **kwargs))
            h[-1] = F.interpolate(
                h[-1],
                scale_factor=self.scale_factor,
                mode="trilinear",
                align_corners=False,
                data_format="NCDHW"
            )

        # Center block
        h[-1] = self.center_block(h[-1], *args, **kwargs)

        # Upsample
        for i, block in enumerate(self.upsampling_blocks):
            h[-1] = F.interpolate(
                h[-1],
                size=h[-i - 2].shape[-3:],
                mode="trilinear",
                align_corners=False,
                data_format="NCDHW"
            )
            h[-1] = block(paddle.concat([h[-1], h[-i - 2]], axis=1), *args, **kwargs)

        return h[-1]


class GeometryEmbedding(nn.Layer):
    """Extract geometry features from local conditioning."""
    def __init__(self, in_features, out_features, actfn):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.actfn = actfn

        self.extract_features = nn.Sequential(
            nn.Conv3D(in_features, out_features, kernel_size=5, stride=5),
            actfn(),
            nn.Conv3D(out_features, out_features, kernel_size=5, stride=1),
            actfn(),
            nn.Conv3D(out_features, out_features, kernel_size=5, stride=5),
        )

    def forward(self, c_local):
        # We pool the geometry embedding over all spatial dimensions
        # to get a global feature vector
        h = self.extract_features(c_local)
        return paddle.mean(h, axis=[-3, -2, -1])


class DenoisingModel(nn.Layer):
    """Core denoising model for the diffusion process."""
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        c_local_features: int,
        c_global_features: int,
        timesteps: int,
        dim: int,
        u_net_levels: int,
        actfn=nn.Silu,
        norm_type: str = "instance",
        with_geometry_embedding: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.c_local_features = c_local_features
        self.c_global_features = c_global_features
        self.dim = dim
        self.timesteps = timesteps
        self.u_net_levels = u_net_levels
        self.with_geometry_embedding = with_geometry_embedding

        # Set up normalization
        if norm_type == "instance":
            norm_klass = lambda dim: nn.GroupNorm(dim, dim)
        elif norm_type == "layer":
            norm_klass = lambda dim: nn.GroupNorm(1, dim)
        elif norm_type == "group":
            norm_klass = lambda dim: nn.GroupNorm(8, dim)
        else:
            raise RuntimeError(f"Unknown norm type {norm_type}")

        # Input encoding
        self.encode_x = nn.Conv3D(in_features, dim, 1)
        
        # Setup conditioning
        c_local_dim = 0
        if c_local_features > 0:
            self.encode_c_local = nn.Conv3D(c_local_features, dim, 1)
            c_local_dim += dim
            
        c_dim = dim
        self.encode_t = NyquistFrequencyEmbedding(dim, timesteps)
        
        if c_global_features > 0:
            self.encode_c_global = nn.Linear(c_global_features, dim)
            c_dim += dim
            
        if with_geometry_embedding and c_local_features > 0:
            self.geometry_embedding = GeometryEmbedding(c_local_features, dim, actfn)
            c_dim += dim

        # Conditioning processing
        self.process_c = nn.Sequential(
            nn.Linear(c_dim, 4 * c_dim),
            actfn(),
            nn.Linear(4 * c_dim, c_dim),
            actfn(),
        )

        # Decoder
        resnet_block = partial(
            ResnetBlock, c_dim=c_dim, actfn=actfn, norm_klass=norm_klass
        )

        self.decode = nn.Sequential(
            resnet_block(dim, dim),
            nn.Conv3D(dim, out_features, 1)
        )

        # U-Net architecture
        downsampling_blocks = [resnet_block(dim + c_local_dim, dim * 2)] + [
            resnet_block(dim * 2**i, dim * 2 ** (i + 1)) for i in range(1, u_net_levels)
        ]
        upsampling_blocks = [
            resnet_block(2 * dim * 2 ** (i + 1), dim * 2**i)
            for i in reversed(range(u_net_levels))
        ]
        center_dim = dim * 2**u_net_levels
        center_block = nn.Sequential(
            resnet_block(center_dim, center_dim),
            Residual(PreNorm(norm_klass(center_dim), Attention(center_dim))),
            resnet_block(center_dim, center_dim),
        )
        self.u_net = UNet(downsampling_blocks, upsampling_blocks, center_block)

    def forward(self, x, t, C):
        """
        x: Input tensor [B, C, H, W, D]
        t: Timestep tensor [B]
        C: Dictionary of conditioning tensors
        """
        # Encode input
        h = self.encode_x(x)
        
        # Process time embedding
        t_emb = self.encode_t(t)
        
        # Process conditioning
        cond_elements = [t_emb]
        
        # Process local conditioning (boundary conditions)
        c_local = None
        if self.c_local_features > 0 and "local" in C:
            c_local = C["local"]
            c_local_encoded = self.encode_c_local(c_local)
            h = paddle.concat([h, c_local_encoded], axis=1)
        
        # Process global conditioning
        if self.c_global_features > 0 and "global" in C:
            c_global = C["global"]
            c_global_encoded = self.encode_c_global(c_global)
            cond_elements.append(c_global_encoded)
        
        # Process geometry embedding
        if self.with_geometry_embedding and c_local is not None:
            geom_emb = self.geometry_embedding(c_local)
            cond_elements.append(geom_emb)
        
        # Combine all conditioning elements
        c = paddle.concat(cond_elements, axis=-1)
        c = self.process_c(c)
        
        # Apply U-Net
        h = self.u_net(h, c=c)
        
        # Decode
        output = self.decode(h)
        
        return output
