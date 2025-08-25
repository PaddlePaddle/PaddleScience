import paddle
import numpy as np
import math
from ppsci.arch import base

class LatentNO(base.Arch):
    def Attention_Vanilla(q, k, v):
        score = paddle.nn.functional.softmax(paddle.matmul(q, paddle.transpose(k, perm=[0, 1, 3, 2])) / math.sqrt(k.shape[-1]), axis=-1)
        r = paddle.matmul(score, v)
        return r

    class LatentMLP(paddle.nn.Layer):
        def __init__(self, input_dim, hidden_dim, output_dim, n_layer):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.n_layer = n_layer
            self.act = paddle.nn.GELU()
            self.input = paddle.nn.Linear(self.input_dim, self.hidden_dim)
            self.hidden = paddle.nn.LayerList([paddle.nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_layer)])
            self.output = paddle.nn.Linear(self.hidden_dim, self.output_dim)
            
        def forward(self, x):
            r = self.act(self.input(x))
            for i in range(0, self.n_layer):
                r = r + self.act(self.hidden[i](r))
            r = self.output(r)
            return r
    
    class SelfAttention(paddle.nn.Layer):
        def __init__(self, n_mode, n_dim, n_head, attn):
            super().__init__()
            self.n_mode = n_mode
            self.n_dim = n_dim
            self.n_head = n_head
            self.Wq = paddle.nn.Linear(self.n_dim, self.n_dim)
            self.Wk = paddle.nn.Linear(self.n_dim, self.n_dim)
            self.Wv = paddle.nn.Linear(self.n_dim, self.n_dim)
            self.attn = attn
            self.proj = paddle.nn.Linear(self.n_dim, self.n_dim)
            
        
        def forward(self, x):
            B, N, D = tuple(x.shape)
            q = self.Wq(x)
            q = paddle.reshape(q, (B, N, self.n_head, D // self.n_head))
            q = paddle.transpose(q, [0, 2, 1, 3])
            k = self.Wk(x)
            k = paddle.reshape(k, (B, N, self.n_head, D // self.n_head))
            k = paddle.transpose(k, [0, 2, 1, 3])
            v = self.Wv(x)
            v = paddle.reshape(v, (B, N, self.n_head, D // self.n_head))
            v = paddle.transpose(v, [0, 2, 1, 3])
            r = self.attn(q, k, v)
            r = paddle.transpose(r, [0, 2, 1, 3])
            r = paddle.reshape(r, (B, N, D))
            r = self.proj(r)
            return r

    
    class AttentionBlock(paddle.nn.Layer):
        def __init__(self, n_mode, n_dim, n_head):
            super().__init__()
            self.n_mode = n_mode
            self.n_dim = n_dim
            self.n_head = n_head
            
            self.self_attn = LatentNO.SelfAttention(self.n_mode, self.n_dim, self.n_head, LatentNO.Attention_Vanilla)
            
            self.ln1 = paddle.nn.LayerNorm(self.n_dim)
            self.ln2 = paddle.nn.LayerNorm(self.n_dim)
            
            self.mlp = paddle.nn.Sequential(
                paddle.nn.Linear(self.n_dim, self.n_dim*2),
                paddle.nn.GELU(),
                paddle.nn.Linear(self.n_dim*2, self.n_dim),
            )

        def forward(self, y):
            y1 = self.ln1(y)
            y = y + self.self_attn(y1)
            y2 = self.ln2(y)
            y = y + self.mlp(y2)
            return y

        
    def __init__(self, n_block, n_mode, n_dim, n_head, n_layer, trunk_dim, branch_dim, out_dim):
        super().__init__()
        self.input_keys = ["x_y1"]   # 和 Dataset 对齐
        self.output_keys = ["y2"]
        self.trunk_dim = trunk_dim
        self.trunk_mlp = LatentNO.LatentMLP(trunk_dim, n_dim, n_dim, n_layer)
        self.branch_mlp = LatentNO.LatentMLP(branch_dim, n_dim, n_dim, n_layer)
        self.out_mlp = LatentNO.LatentMLP(n_dim, n_dim, out_dim, n_layer)
        self.mode_mlp = LatentNO.LatentMLP(n_dim, n_dim, n_mode, n_layer)

        self.attn_blocks = paddle.nn.Sequential(*[LatentNO.AttentionBlock(n_mode, n_dim, n_head) for _ in range(n_block)])

        # 手动 Kaiming_Uniform 初始化
        for module in self.sublayers():
            if isinstance(module, paddle.nn.Linear):
                bound = 1 / math.sqrt(module.weight.shape[0])
                module.weight.set_value(
                    paddle.to_tensor(np.random.uniform(-bound, bound, module.weight.shape).astype("float32"))
                )
                module.bias.set_value(
                    paddle.to_tensor(np.random.uniform(-bound, bound, module.bias.shape).astype("float32"))
                )
            elif isinstance(module, paddle.nn.LayerNorm):
                module.weight.set_value(paddle.ones_like(module.weight))
                module.bias.set_value(paddle.zeros_like(module.bias))

    def forward(self, inputs):

        x = inputs["x"]   # (B, N, trunk_dim)
        y = inputs["y1"]  # (B, N, branch_dim)

        x = self.trunk_mlp(x)
        y = self.branch_mlp(y)

        score = self.mode_mlp(x)
        score_encode = paddle.nn.functional.softmax(score, axis=1)
        score_decode = paddle.nn.functional.softmax(score, axis=-1)

        z = paddle.matmul(paddle.transpose(score_encode, perm=[0, 2, 1]), y)
        for block in self.attn_blocks:
            z = block(z)

        r = paddle.matmul(score_decode, z)
        r = self.out_mlp(r)

        return {"y2": r}   # 必须返回 dict，key 对齐 Dataset