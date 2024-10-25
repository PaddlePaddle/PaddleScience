import numpy as np
import paddle
import yaml
from easydict import EasyDict

# from paddle.vision.models import vision_transformer
from paddleclas import ViT_base_patch16_224


class LayerNorm(paddle.nn.LayerNorm):
    """Subclass paddle's LayerNorm to handle fp16."""

    def forward(self, x: paddle.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.astype("float32"))
        return ret.astype(orig_type)


class QuickGELU(paddle.nn.Layer):
    def forward(self, x: paddle.Tensor):
        return x * paddle.nn.functional.sigmoid(x=1.702 * x)


class ResidualAttentionBlock(paddle.nn.Layer):
    def __init__(self, d_model: int, n_head: int, attn_mask: paddle.Tensor = None):
        super().__init__()
        self.attn = paddle.nn.MultiHeadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = paddle.nn.Sequential(
            *[
                (
                    "c_fc",
                    paddle.nn.Linear(in_features=d_model, out_features=d_model * 4),
                ),
                ("gelu", QuickGELU()),
                (
                    "c_proj",
                    paddle.nn.Linear(in_features=d_model * 4, out_features=d_model),
                ),
            ]
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: paddle.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.place)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: paddle.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(paddle.nn.Layer):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: paddle.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = paddle.nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: paddle.Tensor):
        return self.resblocks(x)


class ULIP_WITH_IMAGE(paddle.nn.Layer):
    def __init__(self, point_encoder, **kwargs):
        super().__init__()
        kwargs = EasyDict(kwargs)
        self.context_length = kwargs.context_length
        self.vision_width = kwargs.vision_width
        self.visual = kwargs.vision_model
        self.transformer = Transformer(
            width=kwargs.transformer_width,
            layers=kwargs.transformer_layers,
            heads=kwargs.transformer_heads,
            attn_mask=self.build_attention_mask(),
        )
        self.vocab_size = kwargs.vocab_size
        self.token_embedding = paddle.nn.Embedding(
            num_embeddings=kwargs.vocab_size, embedding_dim=kwargs.transformer_width
        )
        out_0 = paddle.create_parameter(
            shape=paddle.empty(
                shape=[self.context_length, kwargs.transformer_width]
            ).shape,
            dtype=paddle.empty(shape=[self.context_length, kwargs.transformer_width])
            .numpy()
            .dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.empty(shape=[self.context_length, kwargs.transformer_width])
            ),
        )
        out_0.stop_gradient = not True
        self.positional_embedding = out_0
        self.ln_final = LayerNorm(kwargs.transformer_width)
        out_1 = paddle.create_parameter(
            shape=paddle.empty(shape=[kwargs.vision_width, kwargs.embed_dim]).shape,
            dtype=paddle.empty(shape=[kwargs.vision_width, kwargs.embed_dim])
            .numpy()
            .dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.empty(shape=[kwargs.vision_width, kwargs.embed_dim])
            ),
        )
        out_1.stop_gradient = not True
        self.image_projection = out_1
        out_2 = paddle.create_parameter(
            shape=paddle.empty(
                shape=[kwargs.transformer_width, kwargs.embed_dim]
            ).shape,
            dtype=paddle.empty(shape=[kwargs.transformer_width, kwargs.embed_dim])
            .numpy()
            .dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.empty(shape=[kwargs.transformer_width, kwargs.embed_dim])
            ),
        )
        out_2.stop_gradient = not True
        self.text_projection = out_2
        out_3 = paddle.create_parameter(
            shape=(paddle.ones(shape=[]) * np.log(1 / 0.07)).shape,
            dtype=(paddle.ones(shape=[]) * np.log(1 / 0.07)).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.ones(shape=[]) * np.log(1 / 0.07)
            ),
        )
        out_3.stop_gradient = not True
        self.logit_scale = out_3
        self.initialize_parameters()
        self.point_encoder = point_encoder
        out_4 = paddle.create_parameter(
            shape=paddle.empty(shape=[kwargs.pc_feat_dims, 512]).shape,
            dtype=paddle.empty(shape=[kwargs.pc_feat_dims, 512]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.empty(shape=[kwargs.pc_feat_dims, 512])
            ),
        )
        out_4.stop_gradient = not True
        self.pc_projection = out_4
        init_Normal = paddle.nn.initializer.Normal(std=512**-0.5)
        init_Normal(self.pc_projection)

    def encode_image(self, image):
        x = self.visual(image)
        x = x @ self.image_projection
        return x

    def encode_text(self, text):
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.transpose(perm=[1, 0, 2])
        x = self.transformer(x)
        x = x.transpose(perm=[1, 0, 2])
        x = self.ln_final(x)
        x = (
            x[paddle.arange(end=tuple(x.shape)[0]), text.argmax(axis=-1)]
            @ self.text_projection
        )
        return x

    def build_attention_mask(self):
        mask = paddle.empty(shape=[self.context_length, self.context_length])
        mask.fill_(value=float("-inf"))
        mask.triu_(diagonal=1)
        return mask

    def initialize_parameters(self):
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.token_embedding.weight)
        init_Normal = paddle.nn.initializer.Normal(std=0.01)
        init_Normal(self.positional_embedding)
        proj_std = (
            self.transformer.width**-0.5 * (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            init_Normal = paddle.nn.initializer.Normal(std=attn_std)
            init_Normal(block.attn.q_proj.weight)
            init_Normal = paddle.nn.initializer.Normal(std=proj_std)
            init_Normal(block.attn.out_proj.weight)
            init_Normal = paddle.nn.initializer.Normal(std=fc_std)
            init_Normal(block.mlp.c_fc.weight)
            init_Normal = paddle.nn.initializer.Normal(std=proj_std)
            init_Normal(block.mlp.c_proj.weight)
        init_Normal = paddle.nn.initializer.Normal(std=self.vision_width**-0.5)
        init_Normal(self.image_projection)
        init_Normal = paddle.nn.initializer.Normal(std=self.transformer.width**-0.5)
        init_Normal(self.text_projection)

    def encode_pc(self, pc):
        pc_feat = self.point_encoder(pc)
        pc_embed = pc_feat @ self.pc_projection
        return pc_embed

    def forward(self, pc, text, image=None):
        text_embed_all = []
        for i in range(tuple(text.shape)[0]):
            text_for_one_sample = text[i]
            text_embed = self.encode_text(text_for_one_sample)
            text_embed = text_embed / text_embed.norm(axis=-1, keepdim=True)
            text_embed = text_embed.mean(axis=0)
            text_embed = text_embed / text_embed.norm(axis=-1, keepdim=True)
            text_embed_all.append(text_embed)
        text_embed_all = paddle.stack(x=text_embed_all)
        pc_embed = self.encode_pc(pc)
        if image is not None:
            image_embed = self.encode_image(image)
            return {
                "text_embed": text_embed_all,
                "pc_embed": pc_embed,
                "image_embed": image_embed,
                "logit_scale": self.logit_scale.exp(),
            }
        else:
            return {
                "text_embed": text_embed_all,
                "pc_embed": pc_embed,
                "logit_scale": self.logit_scale.exp(),
            }


def ULIP_PointBERT(args):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = ViT_base_patch16_224(pretrained=True, num_classes=0)
    from geom.models.pointbert.point_encoder import PointTransformer

    config_addr = "./geom/models/pointbert/PointTransformer_8192point.yaml"

    def merge_new_config(config, new_config):
        for key, val in new_config.items():
            if not isinstance(val, dict):
                if key == "_base_":
                    with open(new_config["_base_"], "r") as f:
                        try:
                            val = yaml.load(f, Loader=yaml.FullLoader)
                        except Exception:
                            val = yaml.load(f)
                    config[key] = EasyDict()
                    merge_new_config(config[key], val)
                else:
                    config[key] = val
                    continue
            if key not in config:
                config[key] = EasyDict()
            merge_new_config(config[key], val)
        return config

    def cfg_from_yaml_file(cfg_file):
        config = EasyDict()
        with open(cfg_file, "r") as f:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        merge_new_config(config=config, new_config=new_config)
        return config

    config = cfg_from_yaml_file(config_addr)
    point_encoder = PointTransformer(config.model, args=args)
    pc_feat_dims = 768
    model = ULIP_WITH_IMAGE(
        embed_dim=512,
        vision_width=768,
        point_encoder=point_encoder,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        pc_feat_dims=pc_feat_dims,
    )
    return model


def ULIP_PN_NEXT(args):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = ViT_base_patch16_224(pretrained=True, num_classes=0)
    from geom.models.pointnext.pointnext import PointNEXT

    point_encoder = PointNEXT()
    pc_feat_dims = 256
    model = ULIP_WITH_IMAGE(
        embed_dim=512,
        vision_width=768,
        point_encoder=point_encoder,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        pc_feat_dims=pc_feat_dims,
    )
    return model
