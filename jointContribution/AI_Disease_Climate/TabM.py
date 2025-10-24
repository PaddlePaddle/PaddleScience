import math
from typing import Optional, Literal, Tuple
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader
from paddle.vision.models import resnet18
import matplotlib.pyplot as plt

# =========================
# TabM 组件（Paddle 实现）
# =========================
def init_rsqrt_uniform_(w: paddle.Tensor) -> paddle.Tensor:
    bound = 1.0 / math.sqrt(w.shape[-1])
    noise = paddle.uniform(w.shape, min=-bound, max=bound, dtype=w.dtype)
    w.set_value(noise); return w

def init_random_signs_(w: paddle.Tensor) -> paddle.Tensor:
    with paddle.no_grad():
        p = paddle.full(w.shape, 0.5, dtype='float32')
        s = paddle.bernoulli(p) * 2.0 - 1.0
        s = paddle.cast(s, w.dtype)
        w.set_value(s)
    return w

class NLinear(nn.Layer):
    """PackedEnsemble: K 份 Linear 打包 → 输入 (B,K,D), 权重 (K, I, O)"""
    def __init__(self, k: int, in_f: int, out_f: int, bias: bool = True):
        super().__init__()
        self.k, self.in_f, self.out_f = k, in_f, out_f
        self.weight = self.create_parameter(shape=[k, in_f, out_f])
        self.bias_e = self.create_parameter(shape=[k, out_f]) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        init_rsqrt_uniform_(self.weight)
        if self.bias_e is not None:
            init_rsqrt_uniform_(self.bias_e)

    def forward(self, x):              # x: (B,K,I)
        xk = paddle.transpose(x, [1, 0, 2])      # (K,B,I)
        yk = paddle.bmm(xk, self.weight)         # (K,B,O)
        y  = paddle.transpose(yk, [1, 0, 2])     # (B,K,O)
        if self.bias_e is not None:
            y = y + self.bias_e
        return y

class ScaleEnsemble(nn.Layer):
    def __init__(self, k: int, d: int, init='ones'):
        super().__init__()
        self.k, self.d = k, d
        self.weight = self.create_parameter(shape=[k, d])
        self.init = init; self.reset_parameters()
    def reset_parameters(self):
        if self.init == 'ones':
            self.weight.set_value(paddle.ones_like(self.weight))
        else:
            init_random_signs_(self.weight)
    def forward(self, x):              # (B,K,D)
        return x * self.weight

class LinearBE(nn.Layer):
    """BatchEnsemble Linear:
       y_e = ((x * r_e) @ W) * s_e + b_e
       x: (B,K,I) → y: (B,K,O)
    """
    def __init__(self, in_f: int, out_f: int, k: int, scale_init='ones', bias: bool = True):
        super().__init__()
        self.k, self.in_f, self.out_f = k, in_f, out_f
        self.weight = self.create_parameter(shape=[in_f, out_f])   # 共享权重
        self.r = self.create_parameter(shape=[k, in_f])
        self.s = self.create_parameter(shape=[k, out_f])
        self.use_bias = bias
        self.bias_e = self.create_parameter(shape=[k, out_f]) if bias else None
        self.scale_init = scale_init
        self.reset_parameters()

    def reset_parameters(self):
        init_rsqrt_uniform_(self.weight)
        if self.scale_init == 'ones':
            self.r.set_value(paddle.ones_like(self.r))
            self.s.set_value(paddle.ones_like(self.s))
        else:
            init_random_signs_(self.r); init_random_signs_(self.s)
        if self.use_bias:
            init_rsqrt_uniform_(self.bias_e)

    def forward(self, x):              # (B,K,I)
        xr = x * self.r                                # (B,K,I)
        y  = paddle.matmul(xr, self.weight)            # (B,K,O)
        y  = y * self.s
        if self.use_bias:
            y = y + self.bias_e
        return y

class MLPBlock(nn.Layer):
    def __init__(self, d_in, d_hid, dropout, act='ReLU'):
        super().__init__()
        Act = getattr(nn, act)
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid),
            Act(),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class BackboneMLP(nn.Layer):
    def __init__(self, n_blocks: int, d_in: int, d_hidden: int, dropout: float):
        super().__init__()
        blocks = []
        for i in range(n_blocks):
            blocks.append(MLPBlock(d_in if i==0 else d_hidden, d_hidden, dropout))
        self.blocks = nn.LayerList(blocks)
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

def _get_parent_by_path(root: nn.Layer, path_list):
    cur = root
    for p in path_list:
        if hasattr(cur, p):
            cur = getattr(cur, p)
        else:
            sub_layers = getattr(cur, "_sub_layers", None)
            if sub_layers is None or p not in sub_layers:
                raise AttributeError(f"Cannot locate sublayer '{p}' under '{type(cur).__name__}'")
            cur = sub_layers[p]
    return cur

def _replace_linear(module: nn.Layer, k: int, mode: Literal['be','packed']):
    to_replace = []
    for full_name, layer in module.named_sublayers(include_self=False):
        if isinstance(layer, nn.Linear):
            parts = full_name.split('.')
            parent_path, child_name = parts[:-1], parts[-1]
            parent = _get_parent_by_path(module, parent_path) if parent_path else module
            in_f  = layer.weight.shape[0]
            out_f = layer.weight.shape[1]
            if mode == 'be':
                new_layer = LinearBE(in_f, out_f, k)
                with paddle.no_grad():
                    new_layer.weight.set_value(layer.weight.clone())
                    if layer.bias is not None and new_layer.bias_e is not None:
                        b = layer.bias.reshape([1,-1]).tile([k,1])
                        new_layer.bias_e.set_value(b)
            else:  # packed
                new_layer = NLinear(k, in_f, out_f, bias=layer.bias is not None)
                with paddle.no_grad():
                    w = layer.weight.unsqueeze(0).tile([k,1,1])
                    new_layer.weight.set_value(w)
                    if layer.bias is not None and new_layer.bias_e is not None:
                        b = layer.bias.unsqueeze(0).tile([k,1])
                        new_layer.bias_e.set_value(b)
            to_replace.append((parent, child_name, new_layer))
    for parent, child_name, new_layer in to_replace:
        if hasattr(parent, child_name):
            setattr(parent, child_name, new_layer)
        else:
            parent._sub_layers[child_name] = new_layer

class TabMFeatureExtractor(nn.Layer):
    """arch_type: 'plain' | 'tabm' | 'tabm-mini' | 'tabm-packed'"""
    def __init__(self,
                 num_features: int,
                 arch_type: Literal['plain','tabm','tabm-mini','tabm-packed']='tabm',
                 k: int = 32,
                 backbone_cfg: Optional[dict] = None,
                 reduce: bool = True):
        super().__init__()
        if arch_type == 'plain':
            k = 1
        self.k = k
        self.reduce = reduce
        cfg = backbone_cfg or dict(n_blocks=3, d_hidden=512, dropout=0.1)
        self.d_hidden = cfg["d_hidden"]
        self.backbone = BackboneMLP(**cfg, d_in=num_features)

        if arch_type == 'tabm':
            _replace_linear(self.backbone, k, mode='be')
            self.min_adapter = None
        elif arch_type == 'tabm-mini':
            self.min_adapter = ScaleEnsemble(k, num_features, init='random-signs')
        elif arch_type == 'tabm-packed':
            _replace_linear(self.backbone, k, mode='packed')
            self.min_adapter = None
        else:
            self.min_adapter = None

    def forward(self, x_num: paddle.Tensor):      # x_num: (B, D)
        if self.k > 1:
            x = x_num.unsqueeze(1).tile([1, self.k, 1])  # (B,K,D)
        else:
            x = x_num.unsqueeze(1)                        # (B,1,D)
        if self.min_adapter is not None:
            x = self.min_adapter(x)
        feats = self.backbone(x)                          # (B,K,H)
        return feats.mean(axis=1) if self.reduce else feats  # (B,H) 或 (B,K,H)