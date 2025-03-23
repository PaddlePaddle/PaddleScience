# import math
from collections import OrderedDict
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from einops import rearrange
from timm.models.layers import drop_path
from timm.models.layers import to_2tuple
from timm.models.layers import trunc_normal_
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

# from timm.models.registry import register_model
from tqdm import tqdm
from utils.eval import LossGenerator

# from pdb import set_trace as bp


def trunc_normal_(tensor, mean=0.0, std=1.0):  # noqa
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    # 'pretrain_videomae_small_patch16_224',
    # "pretrain_videomae_base_patch16_224",
    # 'pretrain_videomae_large_patch16_224',
    # 'pretrain_videomae_huge_patch16_224',
]


def _cfg(url="", **kwargs):
    return {
        # 'url': url,
        # 'num_classes': 400,
        "input_size": (3, 512, 512),  # TODO:
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        **kwargs,
    }


def build_vmae(params):
    """Builds model from parameter file.

    General recipe is to build the spatial and temporal modules separately and then
    combine them in a model. Eventually the "stem" and "destem" should
    also be parameterized.
    """
    # space_time_block = build_spacetime_block(params)
    # processor_blocks=params.processor_blocks,
    # n_states=params.n_states,
    # override_block=space_time_block,)
    model = PretrainVisionTransformer(
        img_size=params.input_size,
        patch_size=params.patch_size,
        encoder_embed_dim=params.encoder_embed_dim,
        encoder_depth=12,
        decoder_depth=params.decoder_depth,
        encoder_num_heads=params.encoder_num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        encoder_num_classes=0,
        decoder_num_classes=params.decoder_num_classes,
        tubelet_size=params.tubelet_size,
        decoder_embed_dim=params.decoder_embed_dim,
        decoder_num_heads=params.decoder_num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_frames=params.n_steps,
        num_demos=params.num_demos if hasattr(params, "num_demos") else 0,
        # drop_path_rate=params.drop_path_rate, # TODO:
        # n_states=params.n_states, # TODO:
    )
    model.default_cfg = _cfg()
    if params.vmae_pretrained:
        checkpoint = paddle.load(params.vmae_pretrained)
        if "model" in checkpoint.keys():
            model.load_state_dict(checkpoint["model"])
        elif "model_state" in checkpoint.keys():
            # model.load_state_dict(checkpoint["model_state"])
            # state = {key[7:] if 'module' in key else key: value for key, value in checkpoint["model_state"].items()}
            # model.load_state_dict(state)

            new_state_dict = OrderedDict()
            for key, val in checkpoint["model_state"].items():
                name = key[7:] if "module" in key else key
                new_state_dict[name] = val
            state = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {
                k: v
                for k, v in new_state_dict.items()
                if k in state and state[k].size() == new_state_dict[k].size()
            }
            # 2. overwrite entries in the existing state dict
            state.update(pretrained_dict)
            # 3. load the new state dict
            # message = model.load_state_dict(state)
            # self.model.load_state_dict(new_state_dict)
            unload_keys = [k for k in new_state_dict.keys() if k not in pretrained_dict]
            if len(unload_keys) > 0:
                import warnings

                warnings.warn(
                    "Warning: missing keys during restoring checkpoint: %s"
                    % (str(unload_keys))
                )

    return model


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Mlp(nn.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = paddle.base.framework.EagerParamBase.from_tensor(
                paddle.zeros([all_head_dim])
            )
            self.v_bias = paddle.base.framework.EagerParamBase.from_tensor(
                paddle.zeros([all_head_dim])
            )
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = paddle.concat(
                (self.q_bias, paddle.zeros_like(self.v_bias), self.v_bias)
            )
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape([B, N, 3, self.num_heads, -1]).permute([2, 0, 3, 1, 4])
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape([B, N, -1])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_head_dim=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_head_dim=attn_head_dim,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values > 0:
            gamma_1 = init_values * paddle.ones((dim))
            gamma_1.stop_gradient = False
            self.gamma_1 = paddle.base.framework.EagerParamBase.from_tensor(gamma_1)
            gamma_2 = init_values * paddle.ones((dim))
            gamma_2.stop_gradient = False
            self.gamma_2 = paddle.base.framework.EagerParamBase.from_tensor(gamma_2)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Layer):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        num_frames=16,
        tubelet_size=2,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (num_frames // self.tubelet_size)
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3D(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
            stride=(self.tubelet_size, patch_size[0], patch_size[1]),
        )

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # BCTHW -> BC'T'H'W' -> BC'(T'H'W')
        return x


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""
    # TODO: make it with paddle instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return paddle.to_tensor(
        sinusoid_table, dtype=paddle.float32, stop_gradient=True
    ).unsqueeze(0)


class PretrainVisionTransformerEncoder(nn.Layer):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=512,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        tubelet_size=2,
        use_checkpoint=False,
        use_learnable_pos_emb=False,
        num_frames=16,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            tubelet_size=tubelet_size,
            num_frames=num_frames,
        )
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint

        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = paddle.base.framework.EagerParamBase.from_tensor(
                paddle.zeros([1, num_patches + 1, embed_dim])
            )
        else:
            # sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [
            x.item() for x in paddle.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.LayerList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init_xaiverUniform = paddle.nn.initializer.XavierUniform()
            init_xaiverUniform(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init_constant = paddle.nn.initializer.Constant(value=0)
                init_constant(m.bias)
        elif isinstance(m, nn.LayerNorm):
            init_constant = paddle.nn.initializer.Constant(value=0)
            init_constant(m.bias)
            init_constant = paddle.nn.initializer.Constant(value=1.0)
            init_constant(m.weight)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x, mask=None):
        _, _, T, _, _ = x.shape
        x = self.patch_embed(x)

        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        if mask is not None:
            x_vis = x[~mask].reshape([B, -1, C])  # ~mask means visible
        else:
            x_vis = x.reshape([B, -1, C])

        if self.use_checkpoint:
            for blk in self.blocks:
                x_vis = paddle.distributed.fleet.utils.recompute(blk, x_vis)
        else:
            for blk in self.blocks:
                x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask=None):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x


class PretrainVisionTransformerDecoder(nn.Layer):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        patch_size=16,
        num_classes=768,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        num_patches=196,
        tubelet_size=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 3 * tubelet_size * patch_size**2
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint

        dpr = [
            x.item() for x in paddle.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.LayerList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init_xaiverUniform = paddle.nn.initializer.XavierUniform()
            init_xaiverUniform(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init_constant = paddle.nn.initializer.Constant(value=0)
                init_constant(m.bias)
        elif isinstance(m, nn.LayerNorm):
            init_constant = paddle.nn.initializer.Constant(value=0)
            init_constant(m.bias)
            init_constant = paddle.nn.initializer.Constant(value=1.0)
            init_constant(m.weight)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward(self, x, return_token_num=0):
        if self.use_checkpoint:
            for blk in self.blocks:
                x = paddle.distributed.fleet.utils.recompute(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)

        if return_token_num > 0:
            x = self.head(
                self.norm(x[:, -return_token_num:])
            )  # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x


class PretrainVisionTransformer(nn.Layer):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=512,
        patch_size=16,
        encoder_in_chans=3,
        encoder_num_classes=0,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_num_classes=1536,  #  decoder_num_classes=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=0.0,
        use_learnable_pos_emb=False,
        use_checkpoint=False,
        tubelet_size=2,
        num_classes=0,  # avoid the error from create_fn in timm
        in_chans=0,  # avoid the error from create_fn in timm
        num_frames=16,
        num_demos=0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_learnable_pos_emb=use_learnable_pos_emb,
            num_frames=num_frames,
        )

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
        )

        self.encoder_to_decoder = nn.Linear(
            encoder_embed_dim, decoder_embed_dim, bias=False
        )

        self.mask_token = paddle.base.framework.EagerParamBase.from_tensor(
            paddle.zeros([1, 1, decoder_embed_dim])
        )

        self.pos_embed = get_sinusoid_encoding_table(
            self.encoder.patch_embed.num_patches, decoder_embed_dim
        )

        trunc_normal_(self.mask_token, std=0.02)

        self.num_demos = num_demos
        if self.num_demos > 0:
            self.lossgen = LossGenerator(dx=1 / 256, kernel_size=3)  # TODO:
            # self.lossgen = LossGenerator(dx=2.0*math.pi/2048.0, kernel_size=3) # TODO:

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init_xaiverUniform = paddle.nn.initializer.XavierUniform()
            init_xaiverUniform(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init_constant = paddle.nn.initializer.Constant(value=0)
                init_constant(m.bias)
        elif isinstance(m, nn.LayerNorm):
            init_constant = paddle.nn.initializer.Constant(value=0)
            init_constant(m.bias)
            init_constant = paddle.nn.initializer.Constant(value=1.0)
            init_constant(m.weight)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "mask_token"}

    def forward(self, x, mask=None):
        """
        x: T, B, C, H, W
        """
        T_in, B, C_in, H, W = x.shape
        x = x.permute(1, 2, 0, 3, 4)
        _, _, T, _, _ = x.shape
        x_vis = self.encoder(x, mask)  # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = (
            self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        )
        if mask is not None:
            pos_emd_vis = expand_pos_embed[~mask].reshape([B, -1, C])
            if mask.sum() == 0:
                # mask_ratio = 0: all tokens are visible
                x_full = x_vis + pos_emd_vis  # [B, N, C_d]
                x = self.decoder(x_full)  # [B, :, 3 * 16 * 16]
            else:
                pos_emd_mask = expand_pos_embed[mask].reshape([B, -1, C])
                x_full = paddle.concat(
                    [x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], axis=1
                )  # [B, N, C_d]
                x = self.decoder(
                    x_full, pos_emd_mask.shape[1]
                )  # [B, N_mask, 3 * 16 * 16]
            return x
        else:
            x = self.decoder(x_vis)
            x = rearrange(
                x,
                "b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)",
                t=T_in // self.tubelet_size,
                h=H // self.patch_size,
                w=W // self.patch_size,
                p0=self.tubelet_size,
                p1=self.patch_size,
                p2=self.patch_size,
                c=C_in,
            )
            x = x.permute(2, 0, 1, 3, 4)
            return x[-1]

    def forward_icl(self, x, demo_xs, demo_ys):
        """
        x: T, B, C, H, W
        demo_xs: T, J, C, H, W
        demo_ys: J, H, W
        """
        _, B, _, H, W = x.shape
        # J = demo_xs.shape[1]
        pred = self.forward(x)  # B, 3, H, W
        C = pred.shape[1]
        # div = self.lossgen.get_div_loss(pred)
        demo_pred = []
        idx = 0
        demo_div = []
        while idx < demo_xs.shape[1]:
            _x = demo_xs[:, idx : idx + B]
            _pred = self.forward(_x)
            demo_pred.append(_pred)
            idx += _x.shape[1]
            _div = self.lossgen.get_div_loss(_pred)
            demo_div.append(_div)
        demo_pred = paddle.concat(demo_pred, axis=0)
        demo_div = paddle.concat(demo_div, axis=0)

        demo_pred_flat = demo_pred.permute(0, 2, 3, 1).view([1, -1, C])
        # demo_div_flat = demo_div.view([1, -1])
        y_nn = paddle.zeros([B, C, H, W])
        gap_nn = paddle.zeros([B, H, W])
        batch_b = 1
        _b = 0
        batch_h = 16
        _h = 0
        batch_w = 16
        _w = 0
        # topk1 = round(0.2 * H * W * J) # TODO:
        # topk1 = 20  # TODO:
        # topk = round(0.02 * H * W * J) # TODO:
        topk = 10  # TODO:
        pbar = tqdm(
            total=np.ceil(B / batch_b) * np.ceil(H / batch_h) * np.ceil(W / batch_w)
        )
        while _b < B:
            _h = 0
            while _h < H:
                _w = 0
                while _w < W:
                    pbar.set_description("_b %d, _h %d, _w %d" % (_b, _h, _w))
                    pbar.update(1)
                    pred_flat = pred[
                        _b : _b + batch_b, :, _h : _h + batch_h, _w : _w + batch_w
                    ]
                    __b, _, __h, __w = pred_flat.shape
                    pred_flat = pred_flat.permute(0, 2, 3, 1).view([-1, 1, C])
                    gap = paddle.linalg.norm(
                        (pred_flat - demo_pred_flat).pow(2) / pred_flat.pow(2), axis=-1
                    )
                    gap_re = gap.view([__b, __h, __w, -1])
                    gap_nn[
                        _b : _b + batch_b, _h : _h + batch_h, _w : _w + batch_w
                    ] = paddle.mean(paddle.sort(gap_re, -1)[0][:, :, :, :topk], -1)
                    index = paddle.argsort(paddle.abs(gap_re), -1)[
                        :, :, :, :topk
                    ]  # TODO: spatial index of ascending sort by pred gap
                    # index1 = paddle.argsort(paddle.abs(gap_re), -1)[
                    #     :, :, :, :topk1
                    # ]  # TODO: spatial index of ascending sort by pred gap

                    # div_flat = div[_b:_b+batch_b, :, _h:_h+batch_h, _w:_w+batch_w].view(-1, 1)
                    # gap_div = (div_flat - demo_div_flat).pow(2) / div_flat.pow(2)
                    # gap_re_div = torch.take_along_dim(gap_div, index1.view(__b*__h*__w, -1), dim=1).view(__b, __h, __w, -1)
                    # index = torch.take_along_dim(index1.view(__b*__h*__w, -1), torch.argsort(torch.abs(gap_re_div), -1)[:, :, :, :topk].view(__b*__h*__w, -1), dim=-1).view(__b, __h, __w, -1)

                    _y_nn = 0
                    for _k in range(topk):
                        _y_nn += (
                            paddle.take_along_dim(
                                demo_ys.permute(0, 2, 3, 1).view([-1, C]),
                                index[:, :, :, _k].view([-1, 1]),
                                dim=0,
                            )
                            .view([__b, __h, __w, C])
                            .permute(0, 3, 1, 2)
                        )
                    _y_nn /= topk
                    # spatial_dims = (2, 3)
                    # print("pred:", (((self.target[:, :, _h:_h+batch_h, _w:_w+batch_w] - pred[:, :, _h:_h+batch_h, _w:_w+batch_w]).pow(2).mean(spatial_dims, keepdim=True) / (1e-7 + self.target[:, :, _h:_h+batch_h, _w:_w+batch_w].pow(2).mean(spatial_dims, keepdim=True))).sqrt()).mean().item(), "    ICL:", (((self.target[:, :, _h:_h+batch_h, _w:_w+batch_w] - _y_nn).pow(2).mean(spatial_dims, keepdim=True) / (1e-7 + self.target[:, :, _h:_h+batch_h, _w:_w+batch_w].pow(2).mean(spatial_dims, keepdim=True))).sqrt()).mean().item())
                    y_nn[
                        _b : _b + batch_b, :, _h : _h + batch_h, _w : _w + batch_w
                    ] = _y_nn
                    # np.set_printoptions(precision=5)
                    # print(_h, _w, sum(pred[_b, :2, _h+1, _w+1]-y_nn[_b, :2, _h+1, _w+1]).item(), np.round(pred[_b, :, _h+1, _w+1].detach().cpu().numpy(), 5).tolist(), np.round(y_nn[_b, :, _h+1, _w+1].detach().cpu().numpy(), 5).tolist())

                    _w += batch_w
                _h += batch_h
            _b += batch_b
        # bp()
        print(y_nn.mean(), demo_ys.mean())
        # return y_nn
        # return (y_nn + pred) / 2
        mask = (paddle.clip(gap_nn, 0, 1) ** 0.5 > 0.1).astype(paddle.float32)  # TODO:
        return (1 - mask) * y_nn + mask * pred
