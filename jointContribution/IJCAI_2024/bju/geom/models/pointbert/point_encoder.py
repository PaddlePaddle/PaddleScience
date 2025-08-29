import paddle
import utils.paddle_aux  # NOQA
from geom.models.pointbert.checkpoint import get_missing_parameters_message
from geom.models.pointbert.checkpoint import get_unexpected_parameters_message
from geom.models.pointbert.dvae import Encoder
from geom.models.pointbert.dvae import Group
from geom.models.pointbert.logger import print_log


class Mlp(paddle.nn.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=paddle.nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = paddle.nn.Linear(
            in_features=in_features, out_features=hidden_features
        )
        self.act = act_layer()
        self.fc2 = paddle.nn.Linear(
            in_features=hidden_features, out_features=out_features
        )
        self.drop = paddle.nn.Dropout(p=drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = paddle.nn.Linear(
            in_features=dim, out_features=dim * 3, bias_attr=qkv_bias
        )
        self.attn_drop = paddle.nn.Dropout(p=attn_drop)
        self.proj = paddle.nn.Linear(in_features=dim, out_features=dim)
        self.proj_drop = paddle.nn.Dropout(p=proj_drop)

    def forward(self, x):
        B, N, C = tuple(x.shape)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .transpose(perm=[2, 0, 3, 1, 4])
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = k
        perm_8 = list(range(x.ndim))
        perm_8[-2] = -1
        perm_8[-1] = -2
        attn = q @ x.transpose(perm=perm_8) * self.scale
        attn = paddle.nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        perm_9 = list(range(x.ndim))
        perm_9[1] = 2
        perm_9[2] = 1
        x = x.transpose(perm=perm_9).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DropPath(paddle.nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob is None or self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
        binary_tensor = paddle.floor(random_tensor)
        output = x.divide(keep_prob) * binary_tensor
        return output


class Block(paddle.nn.Layer):
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
        act_layer=paddle.nn.GELU,
        norm_layer=paddle.nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else paddle.nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(paddle.nn.Layer):
    """Transformer Encoder without hierarchical structure"""

    def __init__(
        self,
        embed_dim=768,
        depth=4,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.blocks = paddle.nn.LayerList(
            sublayers=[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate[i]
                    if isinstance(drop_path_rate, list)
                    else drop_path_rate,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class PointTransformer(paddle.nn.Layer):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.args = kwargs["args"]
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder_dims = config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.reduce_dim = paddle.nn.Linear(
            in_features=self.encoder_dims, out_features=self.trans_dim
        )
        out_6 = paddle.create_parameter(
            shape=paddle.zeros(shape=[1, 1, self.trans_dim]).shape,
            dtype=paddle.zeros(shape=[1, 1, self.trans_dim]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.zeros(shape=[1, 1, self.trans_dim])
            ),
        )
        out_6.stop_gradient = not True
        self.cls_token = out_6
        out_7 = paddle.create_parameter(
            shape=paddle.randn(shape=[1, 1, self.trans_dim]).shape,
            dtype=paddle.randn(shape=[1, 1, self.trans_dim]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.randn(shape=[1, 1, self.trans_dim])
            ),
        )
        out_7.stop_gradient = not True
        self.cls_pos = out_7
        self.pos_embed = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=3, out_features=128),
            paddle.nn.GELU(),
            paddle.nn.Linear(in_features=128, out_features=self.trans_dim),
        )
        dpr = [
            x.item()
            for x in paddle.linspace(start=0, stop=self.drop_path_rate, num=self.depth)
        ]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )
        self.norm = paddle.nn.LayerNorm(normalized_shape=self.trans_dim)

    def build_loss_func(self):
        self.loss_ce = paddle.nn.CrossEntropyLoss()

    def get_loss_acc(self, pred, gt, smoothing=True):
        gt = gt.view(-1).astype(dtype="int64")
        if smoothing:
            eps = 0.2
            n_class = pred.shape[1]
            one_hot = paddle.zeros_like(x=pred).put_along_axis(
                axis=1, indices=gt.view(-1, 1), values=1
            )
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = paddle.nn.functional.log_softmax(x=pred, axis=1)
            loss = -(one_hot * log_prb).sum(axis=1).mean()
        else:
            loss = self.loss_ce(pred, gt.astype(dtype="int64"))
        pred = pred.argmax(axis=-1)
        acc = (pred == gt).sum() / float(gt.shape[0])
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = paddle.load(path=bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt["base_model"].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith("transformer_q") and not k.startswith(
                "transformer_q.cls_head"
            ):
                base_ckpt[k[len("transformer_q.") :]] = base_ckpt[k]
            elif k.startswith("base_model"):
                base_ckpt[k[len("base_model.") :]] = base_ckpt[k]
            del base_ckpt[k]
        incompatible = self.set_state_dict(
            state_dict=base_ckpt, use_structured_name=False
        )
        if incompatible.missing_keys:
            print_log("missing_keys", logger="Transformer")
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger="Transformer",
            )
        if incompatible.unexpected_keys:
            print_log("unexpected_keys", logger="Transformer")
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger="Transformer",
            )
        print_log(
            f"[Transformer] Successful Loading the ckpt from {bert_ckpt_path}",
            logger="Transformer",
        )

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)
        group_input_tokens = self.reduce_dim(group_input_tokens)
        cls_tokens = self.cls_token.expand(shape=[group_input_tokens.shape[0], -1, -1])
        cls_pos = self.cls_pos.expand(shape=[group_input_tokens.shape[0], -1, -1])
        pos = self.pos_embed(center)
        x = paddle.concat(x=(cls_tokens, group_input_tokens), axis=1)
        pos = paddle.concat(x=(cls_pos, pos), axis=1)
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = paddle.concat(x=[x[:, 0], x[:, 1:].max(1)[0]], axis=-1)
        return concat_f
