"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Basic blocks for the Perceiver architecture.

The code borrows elements from the following files:

    https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py
    https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py

These files are licenced under respectively the following two licences:

    MIT License

    Copyright (c) 2021 Phil Wang

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    MIT License

    Copyright (c) 2023 Anas Awadalla, Irena Gao, Joshua Gardner, Jack Hessel, Yusuf
    Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Jenia Jitsev,
    Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, Ludwig Schmidt.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

from __future__ import annotations

import einops
import paddle

from ppsci.arch.aurora.model.util import fp16_safe_scaled_dot_product_attention
from ppsci.arch.aurora.model.util import scaled_dot_product_attention

__all__ = ["MLP", "PerceiverResampler"]


class MLP(paddle.nn.Layer):
    """A simple one-hidden-layer MLP."""

    def __init__(self, dim: int, hidden_features: int, dropout: float = 0.0) -> None:
        """Initialise.

        Args:
            dim (int): Input dimensionality.
            hidden_features (int): Width of the hidden layer.
            dropout (float, optional): Drop-out rate. Defaults to no drop-out.
        """
        super().__init__()
        self.net = paddle.nn.Sequential(
            paddle.nn.Linear(dim, hidden_features),
            paddle.nn.GELU(),
            paddle.nn.Linear(hidden_features, dim),
            paddle.nn.Dropout(dropout),
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Run the MLP."""
        return self.net(x)


class PerceiverAttention(paddle.nn.Layer):
    """Cross attention module from the Perceiver architecture."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        head_dim: int = 64,
        num_heads: int = 8,
        ln_k_q: bool = False,
    ) -> None:
        """Initialise.

        Args:
            latent_dim (int): Dimensionality of the latent features given as input.
            context_dim (int): Dimensionality of the context features also given as input.
            head_dim (int): Attention head dimensionality.
            num_heads (int): Number of heads.
            ln_k_q (bool): Apply an extra layer norm. to the keys and queries.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = head_dim * num_heads
        self.to_q = paddle.nn.Linear(latent_dim, self.inner_dim, bias_attr=False)
        self.to_kv = paddle.nn.Linear(context_dim, self.inner_dim * 2, bias_attr=False)
        self.to_out = paddle.nn.Linear(self.inner_dim, latent_dim, bias_attr=False)
        self.use_fp16_safe_attention = False
        if ln_k_q:
            self.ln_k = paddle.nn.LayerNorm(num_heads * head_dim)
            self.ln_q = paddle.nn.LayerNorm(num_heads * head_dim)
        else:
            self.ln_k = lambda x: x
            self.ln_q = lambda x: x

    def forward(self, latents: paddle.Tensor, x: paddle.Tensor) -> paddle.Tensor:
        """Run the cross-attention module.

        Args:
            latents (:class:`paddle.Tensor`): Latent features of shape `(B, L1, Latent_D)`
                where typically `L1 < L2` and `Latent_D <= Context_D`. `Latent_D` is equal to
                `self.latent_dim`.
            x (:class:`paddle.Tensor`): Context features of shape `(B, L2, Context_D)`.

        Returns:
            :class:`paddle.Tensor`: Latent values of shape `(B, L1, Latent_D)`.
        """
        h = self.num_heads
        q = self.to_q(latents)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        k = self.ln_k(k)
        q = self.ln_q(q)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b l (h d) -> b h l d", h=h), (q, k, v)
        )
        sdpa = (
            fp16_safe_scaled_dot_product_attention
            if self.use_fp16_safe_attention
            else scaled_dot_product_attention
        )
        out = sdpa(q, k, v)
        out = einops.rearrange(out, "B H L1 D -> B L1 (H D)")
        return self.to_out(out)


class PerceiverResampler(paddle.nn.Layer):
    """Perceiver Resampler module from the Flamingo paper."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        depth: int = 1,
        head_dim: int = 64,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        residual_latent: bool = True,
        ln_eps: float = 1e-05,
        ln_k_q: bool = False,
    ) -> None:
        """Initialise.

        Args:
            latent_dim (int): Dimensionality of the latent features given as input.
            context_dim (int): Dimensionality of the context features also given as input.
            depth (int, optional): Number of attention layers.
            head_dim (int, optional): Attention head dimensionality. Defaults to `64`.
            num_heads (int, optional): Number of heads. Defaults to `16`
            mlp_ratio (float, optional): Rimensionality of the hidden layer divided by that of the
                input for all MLPs. Defaults to `4.0`.
            drop (float, optional): Drop-out rate. Defaults to no drop-out.
            residual_latent (bool, optional): Use residual attention w.r.t. the latent features.
                Defaults to `True`.
            ln_eps (float, optional): Epsilon in the layer normalisation layers. Defaults to
                `1e-5`.
            ln_k_q (bool, optional): Apply an extra layer norm. to the keys and queries of the first
                resampling layer. Defaults to `False`.
        """
        super().__init__()
        self.residual_latent = residual_latent
        self.layers = paddle.nn.LayerList([])
        mlp_hidden_dim = int(latent_dim * mlp_ratio)
        for i in range(depth):
            self.layers.append(
                paddle.nn.LayerList(
                    [
                        PerceiverAttention(
                            latent_dim=latent_dim,
                            context_dim=context_dim,
                            head_dim=head_dim,
                            num_heads=num_heads,
                            ln_k_q=ln_k_q if i == 0 else False,
                        ),
                        MLP(
                            dim=latent_dim, hidden_features=mlp_hidden_dim, dropout=drop
                        ),
                        paddle.nn.LayerNorm(latent_dim, epsilon=ln_eps),
                        paddle.nn.LayerNorm(latent_dim, epsilon=ln_eps),
                    ]
                )
            )

    def forward(self, latents: paddle.Tensor, x: paddle.Tensor) -> paddle.Tensor:
        """Run the module.

        Args:
            latents (:class:`paddle.Tensor`): Latent features of shape `(B, L1, D1)`.
            x (:class:`paddle.Tensor`): Context features of shape `(B, L2, D1)`.

        Returns:
            paddle.Tensor: Latent features of shape `(B, L1, D1)`.
        """
        for attn, ff, ln1, ln2 in self.layers:
            attn_out = ln1(attn(latents, x))
            latents = attn_out + latents if self.residual_latent else attn_out
            if not self.training:
                del attn_out
            latents = ln2(ff(latents)) + latents
        return latents
