# -*- coding: utf-8 -*-
import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader
# 选中第 0 张 GPU；如有多卡改成 'gpu:1' 等
# paddle.set_device('gpu:0')

# ====================== 工具：正弦位置编码 ======================
class SinusoidalPositionalEncoding(nn.Layer):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = np.zeros((max_len, d_model), dtype="float32")
        position = np.arange(0, max_len, dtype="float32")[:, None]
        div_term = np.exp(np.arange(0, d_model, 2, dtype="float32") * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.register_buffer("pe", paddle.to_tensor(pe), persistable=False)
    def forward(self, x):  # (B,T,D)
        T = x.shape[1]
        return x + self.pe[:T, :]

# ====================== TabM（占位，可换你的实现） ======================
class TabMFeatureExtractor(nn.Layer):
    def __init__(self, num_features: int, d_hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
        )
        self.d_hidden = d_hidden
    def forward(self, x_num: paddle.Tensor):
        return self.net(x_num)

# ====================== 3D ResNet-18 体数据特征抽取 ======================
class BasicBlock3D(nn.Layer):
    expansion = 1
    def __init__(self, in_planes, planes, stride=(1,1,1), downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3D(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias_attr=False)
        self.bn1   = nn.BatchNorm3D(planes)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv3D(planes, planes, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn2   = nn.BatchNorm3D(planes)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out

class ResNet3D(nn.Layer):
    def __init__(self, block, layers, in_channels=20, base_width=64):
        super().__init__()
        self.in_planes = base_width
        self.conv1 = nn.Conv3D(in_channels, self.in_planes,
                               kernel_size=(3,7,7), stride=(1,2,2),
                               padding=(1,3,3), bias_attr=False)
        self.bn1   = nn.BatchNorm3D(self.in_planes)
        self.relu  = nn.ReLU()
        self.maxpool = nn.MaxPool3D(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.layer1 = self._make_layer(block, base_width,  layers[0], stride=(1,1,1))
        self.layer2 = self._make_layer(block, base_width*2, layers[1], stride=(2,2,2))
        self.layer3 = self._make_layer(block, base_width*4, layers[2], stride=(2,2,2))
        self.layer4 = self._make_layer(block, base_width*8, layers[3], stride=(2,2,2))
        self.out_dim = base_width*8  # 512
        self.pool = nn.AdaptiveAvgPool3D(output_size=1)
    def _make_layer(self, block, planes, blocks, stride=(1,1,1)):
        downsample = None
        if stride != (1,1,1) or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3D(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm3D(planes * block.expansion),
            )
        layers = [block(self.in_planes, planes, stride=stride, downsample=downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):  # (B, C, D, H, W)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)           # (B, 512, 1,1,1)
        x = paddle.flatten(x, 1)   # (B, 512)
        return x

class Volume3DEncoder(nn.Layer):
    def __init__(self, in_channels: int = 20, base: int = 64, dropout: float = 0.0):
        super().__init__()
        self.backbone = ResNet3D(BasicBlock3D, layers=[2,2,2,2], in_channels=in_channels, base_width=base)
        self.drop = nn.Dropout(dropout)
        self.out_dim = self.backbone.out_dim  # 512
    def forward(self, x):  # (B, C, D, H, W)
        x = self.backbone(x)
        x = self.drop(x)
        return x

# ====================== MoE（Top-k；gather_nd 选择专家） ======================
class ExpertFFN(nn.Layer):
    def __init__(self, d_model, d_ff, dropout=0.1, act='relu'):
        super().__init__()
        Act = getattr(F, act) if isinstance(act, str) else act
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
        self.act = Act
    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))

class MoEConfig:
    def __init__(self,
                 n_experts=8, top_k=1, d_ff=2048, dropout=0.1,
                 router_temp=0.5, balance_loss_w=0.005, entropy_reg_w=-0.005,
                 diversity_w=1e-3, sticky_w=0.0, sup_router_w=0.0, use_gumbel=True):
        self.n_experts = n_experts; self.top_k = top_k; self.d_ff = d_ff; self.dropout = dropout
        self.router_temp = router_temp; self.balance_loss_w = balance_loss_w
        self.entropy_reg_w = entropy_reg_w; self.diversity_w = diversity_w
        self.sticky_w = sticky_w; self.sup_router_w = sup_router_w; self.use_gumbel = use_gumbel

class MoE(nn.Layer):
    def __init__(self, d_model: int, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(d_model, cfg.n_experts)
        self.experts = nn.LayerList([ExpertFFN(d_model, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_experts)])
        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(cfg.dropout)
    def _router_probs(self, logits):
        if self.cfg.use_gumbel and self.training:
            u = paddle.uniform(logits.shape, min=1e-6, max=1-1e-6, dtype=logits.dtype)
            g = -paddle.log(-paddle.log(u)); logits = logits + g
        return F.softmax(logits / self.cfg.router_temp, axis=-1)
    def forward(self, x, domain_id=None):
        orig_shape = x.shape
        if len(orig_shape) == 3:
            B, T, D = orig_shape; X = x.reshape([B*T, D])
        else:
            X = x
        N, D = X.shape
        logits = self.router(X); probs = self._router_probs(logits)
        topk_val, topk_idx = paddle.topk(probs, k=self.cfg.top_k, axis=-1)
        all_out = paddle.stack([e(X) for e in self.experts], axis=1)  # (N,E,D)
        arangeN = paddle.arange(N, dtype='int64')
        picked_list = []
        for i in range(self.cfg.top_k):
            idx_i = topk_idx[:, i].astype('int64')
            idx_nd = paddle.stack([arangeN, idx_i], axis=1)
            picked_i = paddle.gather_nd(all_out, idx_nd)
            picked_list.append(picked_i)
        picked = paddle.stack(picked_list, axis=1)  # (N,k,D)
        w = topk_val / (paddle.sum(topk_val, axis=-1, keepdim=True) + 1e-9)
        Y = paddle.sum(picked * w.unsqueeze(-1), axis=1)
        Y = self.drop(Y); Y = self.ln(Y + X)
        aux = 0.0
        if self.cfg.balance_loss_w > 0:
            mean_prob = probs.mean(axis=0)
            target = paddle.full_like(mean_prob, 1.0 / self.cfg.n_experts)
            aux = aux + self.cfg.balance_loss_w * F.mse_loss(mean_prob, target)
        if self.cfg.entropy_reg_w != 0.0:
            ent = -paddle.sum(probs * (paddle.log(probs + 1e-9)), axis=1).mean()
            aux = aux + self.cfg.entropy_reg_w * ent
        if (domain_id is not None) and (self.cfg.sup_router_w > 0):
            dom = domain_id.reshape([-1])[:N] % self.cfg.n_experts
            aux = aux + self.cfg.sup_router_w * F.cross_entropy(logits, dom)
        if self.cfg.diversity_w > 0 and self.cfg.n_experts > 1:
            chosen = F.one_hot(topk_idx[:, 0], num_classes=self.cfg.n_experts).astype('float32')
            denom = chosen.sum(axis=0).clip(min=1.0).unsqueeze(-1)
            means = (all_out * chosen.unsqueeze(-1)).sum(axis=0) / denom
            sims = []
            for i in range(self.cfg.n_experts):
                for j in range(i+1, self.cfg.n_experts):
                    si = F.normalize(means[i:i+1], axis=-1)
                    sj = F.normalize(means[j:j+1], axis=-1)
                    sims.append((si*sj).sum())
            if sims:
                aux = aux + self.cfg.diversity_w * paddle.stack(sims).mean()
        if len(orig_shape) == 3:
            Y = Y.reshape([B, T, D])
        return Y, aux

class MoEHead(nn.Layer):
    def __init__(self, d_model=512, cfg: MoEConfig = None):
        super().__init__()
        self.moe = MoE(d_model, cfg or MoEConfig())
    def forward(self, tok, domain_id=None):
        y, aux = self.moe(tok.unsqueeze(1), domain_id=domain_id)
        return y.squeeze(1), aux

# ====================== Self-Attention Transformer（可 MoE） ======================
class TransformerEncoderLayerMoE(nn.Layer):
    def __init__(self, d_model=512, nhead=8, d_ff=1024, dropout=0.1,
                 use_moe: bool = True, moe_cfg: MoEConfig = None):
        super().__init__()
        self.use_moe = use_moe
        self.self_attn = nn.MultiHeadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model); self.do1 = nn.Dropout(dropout)
        if use_moe:
            self.moe = MoE(d_model, moe_cfg or MoEConfig(d_ff=d_ff, dropout=dropout))
        else:
            self.ffn = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            ); self.do2 = nn.Dropout(dropout)
    def forward(self, x, domain_id=None):  # (B,T,D)
        h = self.ln1(x)
        h = paddle.transpose(h, [1,0,2])
        sa = self.self_attn(h, h, h)
        sa = paddle.transpose(sa, [1,0,2])
        x = x + self.do1(sa)
        aux = 0.0
        if self.use_moe:
            x, aux = self.moe(x, domain_id=domain_id)
        else:
            x = x + self.do2(self.ffn(x))
        return x, aux

class TemporalTransformerFlexible(nn.Layer):
    def __init__(self, d_model=512, nhead=8, num_layers=2, d_ff=1024, dropout=0.1,
                 max_len=4096, use_moe: bool = True, moe_cfg: MoEConfig = None):
        super().__init__()
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.LayerList([
            TransformerEncoderLayerMoE(d_model, nhead, d_ff, dropout, use_moe=use_moe, moe_cfg=moe_cfg)
            for _ in range(num_layers)
        ])
    def forward(self, x, domain_id=None):
        x = self.pos(x); aux_total = 0.0
        for layer in self.layers:
            x, aux = layer(x, domain_id=domain_id); aux_total += aux
        return x, aux_total

# ====================== AFNO(1D) + MoE FFN ======================
class AFNO1DLayer(nn.Layer):
    """
    自适应傅里叶算子（时间 1D 版）：
    - 对 (B,T,D) 沿 T 做 rFFT → (B,D,F)
    - 仅保留前 K=modes 个频率，对每个频率在“通道组内”做两层复线性（W1,W2）+ GELU + Softshrink
    - 把频谱其余部分置零 → irFFT → 残差 + Dropout + (可选 LN)
    """
    def __init__(self, d_model: int, modes: int = 32, num_blocks: int = 8,
                 shrink: float = 0.01, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_blocks == 0, "d_model must be divisible by num_blocks"
        self.d_model = d_model
        self.modes = modes
        self.num_blocks = num_blocks
        self.block = d_model // num_blocks
        self.shrink = shrink
        # 复权重拆成实/虚：形状 (G, Cb, Cb)
        scale = 1.0 / math.sqrt(self.block)
        def param():
            return nn.initializer.Uniform(-scale, scale)
        self.w1r = self.create_parameter([num_blocks, self.block, self.block], default_initializer=param())
        self.w1i = self.create_parameter([num_blocks, self.block, self.block], default_initializer=param())
        self.w2r = self.create_parameter([num_blocks, self.block, self.block], default_initializer=param())
        self.w2i = self.create_parameter([num_blocks, self.block, self.block], default_initializer=param())
        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def _complex_linear(self, xr, xi, Wr, Wi):
        # xr, xi: (B, G, K, Cb); Wr/Wi: (G, Cb, Cb)
        # (a+ib)*(Wr+iWi) = (a@Wr - b@Wi) + i(a@Wi + b@Wr)

        out_r = paddle.einsum('ngkc,gcd->ngkd', xr, Wr) - paddle.einsum('ngkc,gcd->ngkd', xi, Wi)
        out_i = paddle.einsum('ngkc,gcd->ngkd', xr, Wi) + paddle.einsum('ngkc,gcd->ngkd', xi, Wr)
        # out_r = paddle.matmul(xr, Wr) - paddle.matmul(xi, Wi)
        # out_i = paddle.matmul(xr, Wi) + paddle.matmul(xi, Wr)
        return out_r, out_i

    def forward(self, x):  # x: (B,T,D)
        B, T, D = x.shape
        Kmax = T // 2 + 1
        K = min(self.modes, Kmax)

        h = self.ln(x)                            # PreNorm
        h_td = paddle.transpose(h, [0, 2, 1])     # (B,D,T)
        h_ft = paddle.fft.rfft(h_td)              # (B,D,F) complex64

        # reshape 通道为 G 组： (B,G,Cb,F)
        h_ft = h_ft.reshape([B, self.num_blocks, self.block, Kmax])
        # 仅前 K 频率： (B,G,Cb,K) → 交换到 (B,G,K,Cb) 方便 matmul
        xk = h_ft[:, :, :, :K].transpose([0,1,3,2])
        xr, xi = paddle.real(xk), paddle.imag(xk)  # (B,G,K,Cb)

        # 组内两层复线性 + GELU + Softshrink
        yr, yi = self._complex_linear(xr, xi, self.w1r, self.w1i)
        yr = F.gelu(yr); yi = F.gelu(yi)
        # Softshrink（稀疏化）
        # yr = F.softshrink(yr, lambd=self.shrink); yi = F.softshrink(yi, lambd=self.shrink)
        yr = F.softshrink(yr, threshold=self.shrink)
        yi = F.softshrink(yi, threshold=self.shrink)
        yr, yi = self._complex_linear(yr, yi, self.w2r, self.w2i)  # (B,G,K,Cb)




        # 放回谱： (B,G,K,Cb) → (B,G,Cb,K) → (B,D,K)
        yk = paddle.complex(yr, yi).transpose([0,1,3,2]).reshape([B, D, K])
        out_ft = paddle.zeros([B, D, Kmax], dtype='complex64')
        out_ft[:, :, :K] = yk

        # 反变换 & 残差
        out_td = paddle.fft.irfft(out_ft, n=T)    # (B,D,T)
        out = paddle.transpose(out_td, [0, 2, 1]) # (B,T,D)
        out = self.drop(out)
        return x + out

class AFNOTransformerFlexible(nn.Layer):
    """
    堆叠若干 AFNO1DLayer；随后接 MoE FFN（与 Self-Attn 分支同构）
    """
    def __init__(self, d_model=512, num_layers=2, modes=32, dropout=0.1,
                 d_ff=1024, use_moe: bool = True, moe_cfg: MoEConfig = None):
        super().__init__()
        self.layers = nn.LayerList([AFNO1DLayer(d_model, modes=modes, num_blocks=8, shrink=0.01, dropout=dropout)
                                    for _ in range(num_layers)])
        self.use_moe = use_moe
        if use_moe:
            self.moe = MoE(d_model, moe_cfg or MoEConfig(d_ff=d_ff, dropout=dropout))
        else:
            self.ffn = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )
            self.do = nn.Dropout(dropout)

    def forward(self, x, domain_id=None):  # (B,T,D)
        for layer in self.layers:
            x = layer(x)
        aux = 0.0
        if self.use_moe:
            x, aux = self.moe(x, domain_id=domain_id)
        else:
            x = x + self.do(self.ffn(x))
        return x, aux

# ====================== Cross-Attention 融合 ======================
class MultiHeadCrossAttention(nn.Layer):
    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_head = d_model // nhead; self.nhead = nhead
        self.Wq = nn.Linear(d_model, d_model); self.Wk = nn.Linear(d_model, d_model); self.Wv = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model); self.drop = nn.Dropout(dropout); self.ln = nn.LayerNorm(d_model)
    def forward(self, q, kv):
        B, Nq, D = q.shape
        def split(t): return t.reshape([B, -1, self.nhead, self.d_head]).transpose([0,2,1,3])
        qh = split(self.Wq(q)); kh = split(self.Wk(kv)); vh = split(self.Wv(kv))
        scores = paddle.matmul(qh, kh, transpose_y=True) / math.sqrt(self.d_head)
        attn = F.softmax(scores, axis=-1)
        ctx = paddle.matmul(attn, vh).transpose([0,2,1,3]).reshape([B, Nq, D])
        out = self.drop(self.proj(ctx))
        return self.ln(out + q)

class BiModalCrossFusion(nn.Layer):
    def __init__(self, d_model=512, nhead=8, dropout=0.1, fuse_hidden=512):
        super().__init__()
        self.ca_v_from_t = MultiHeadCrossAttention(d_model, nhead, dropout)
        self.ca_t_from_v = MultiHeadCrossAttention(d_model, nhead, dropout)
        self.fuse = nn.Sequential(nn.Linear(2*d_model, fuse_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.out_dim = fuse_hidden
    def forward(self, video_seq, tabm_tok):
        v_tok = video_seq.mean(axis=1, keepdim=True)
        t_tok = tabm_tok.unsqueeze(1)
        v_upd = self.ca_v_from_t(v_tok, t_tok)
        t_upd = self.ca_t_from_v(t_tok, video_seq)
        fused = paddle.concat([v_upd, t_upd], axis=-1).squeeze(1)
        return self.fuse(fused)

# ====================== 总模型：Self-Attn + AFNO 并行 ======================
class TwoModalMultiLabelModel(nn.Layer):
    def __init__(self,
                 # 视频模态
                 vid_channels=20, vid_h=20, vid_w=20, vid_frames=365, depth_n=24,
                 # 结构化模态
                 vec_dim=424,
                 # 维度与结构
                 d_model=512, nhead=4, n_trans_layers=2, trans_ff=1024,
                 tabm_hidden=512, dropout=0.1, num_labels=4,
                 # MoE 开关
                 moe_temporal_attn: bool = True,
                 moe_temporal_afno: bool = True,
                 moe_fused: bool = False,
                 moe_tabm: bool = False,
                 # AFNO 频率数
                 afno_modes: int = 32,
                 # MoE 超参
                 moe_cfg_temporal_attn: MoEConfig = None,
                 moe_cfg_temporal_afno: MoEConfig = None,
                 moe_cfg_fused: MoEConfig = None,
                 moe_cfg_tabm: MoEConfig = None):
        super().__init__()
        # 逐帧 3D ResNet18
        self.vol_encoder = Volume3DEncoder(in_channels=vid_channels, dropout=dropout)
        # Self-Attention Transformer
        self.trans_attn = TemporalTransformerFlexible(
            d_model=d_model, nhead=nhead, num_layers=n_trans_layers, d_ff=trans_ff, dropout=dropout,
            max_len=vid_frames, use_moe=moe_temporal_attn,
            moe_cfg=moe_cfg_temporal_attn or MoEConfig(
                n_experts=8, top_k=1, d_ff=max(2048, trans_ff), router_temp=0.5,
                balance_loss_w=0.005, entropy_reg_w=-0.005, diversity_w=1e-3
            )
        )
        # AFNO Transformer（1D）
        self.trans_afno = AFNOTransformerFlexible(
            d_model=d_model, num_layers=n_trans_layers, modes=afno_modes, dropout=dropout,
            d_ff=trans_ff, use_moe=moe_temporal_afno,
            moe_cfg=moe_cfg_temporal_afno or MoEConfig(
                n_experts=8, top_k=1, d_ff=max(2048, trans_ff), router_temp=0.5,
                balance_loss_w=0.005, entropy_reg_w=-0.005, diversity_w=1e-3
            )
        )
        # 两路拼接后投回 d_model
        self.video_merge = nn.Linear(2*d_model, d_model)

        # TabM
        self.tabm = TabMFeatureExtractor(vec_dim, d_hidden=tabm_hidden, dropout=dropout)
        self.tabm_proj = nn.Linear(tabm_hidden, d_model)
        self.moe_tabm = moe_tabm
        if moe_tabm:
            self.tabm_moe = MoEHead(d_model=d_model, cfg=moe_cfg_tabm or MoEConfig(
                n_experts=6, top_k=1, d_ff=1024, router_temp=0.5,
                balance_loss_w=0.005, entropy_reg_w=-0.005, diversity_w=1e-3
            ))

        # 融合
        self.fusion = BiModalCrossFusion(d_model=d_model, nhead=nhead, dropout=dropout, fuse_hidden=d_model)
        self.moe_fused = moe_fused
        if moe_fused:
            self.fused_moe = MoEHead(d_model=d_model, cfg=moe_cfg_fused or MoEConfig(
                n_experts=6, top_k=1, d_ff=1024, router_temp=0.5,
                balance_loss_w=0.005, entropy_reg_w=-0.005, diversity_w=1e-3
            ))

        # 分类头
        self.head = nn.Linear(self.fusion.out_dim, num_labels)

        self.vid_frames = vid_frames; self.depth_n = depth_n

    # 导出融合前 512 表示（用于检索库）
    def encode(self, x_video, x_vec, domain_id=None):
        """
        x_video: (B,T,C,H,W,N)  —— N 为体深度（24）
        """
        B, T, C, H, W, N = x_video.shape
        assert N == self.depth_n, f"N mismatch: {N} vs {self.depth_n}"
        xvt = x_video.transpose([0,1,2,5,3,4]).reshape([B*T, C, N, H, W])
        f_frame = self.vol_encoder(xvt)               # (B*T,512)
        seq = f_frame.reshape([B, T, -1])             # (B,T,512)

        z_attn, _  = self.trans_attn(seq, domain_id=domain_id)      # (B,T,512)
        z_afno, _  = self.trans_afno(seq, domain_id=domain_id)      # (B,T,512)
        z_vid = self.video_merge(paddle.concat([z_attn, z_afno], axis=-1))  # (B,T,512)

        z_tabm = self.tabm(x_vec); z_tabm = self.tabm_proj(z_tabm)  # (B,512)
        if self.moe_tabm:
            z_tabm, _ = self.tabm_moe(z_tabm, domain_id=domain_id)

        fused = self.fusion(z_vid, z_tabm)             # (B,512)
        if self.moe_fused:
            fused, _ = self.fused_moe(fused, domain_id=domain_id)
        return fused

    # def forward(self, x_video, x_vec, domain_id=None):
    def forward(self, input_dict, domain_id=None):
        x_video, x_vec = input_dict["video"], input_dict["vec"]
        fused = self.encode(x_video, x_vec, domain_id=domain_id)
        logits = self.head(fused)
        # (B,4)
        return {"y": logits}

# ====================== 检索增强（cos / l2；k 邻居软加权；概率融合） ======================
class Retriever:
    def __init__(self, sim_metric: str = 'cos', k: int = 8, alpha: float = 0.3, tau: float = 0.5):
        assert sim_metric in ['cos', 'l2']
        self.sim_metric = sim_metric; self.k = k; self.alpha = alpha; self.tau = tau
        self.keys = None; self.labels = None
    @paddle.no_grad()
    def build(self, model: nn.Layer, loader: DataLoader):
        model.eval()
        feats, labs = [], []
        for x_vid, x_vec, y in loader:
            f = model.encode(x_vid.astype('float32'), x_vec.astype('float32'))
            feats.append(f.numpy()); labs.append(y.numpy())
        self.keys = paddle.to_tensor(np.concatenate(feats, 0)).astype('float32')
        self.labels = paddle.to_tensor(np.concatenate(labs, 0)).astype('float32')
        self.keys_norm = F.normalize(self.keys, axis=-1)
    @paddle.no_grad()
    def query_and_fuse(self, model_probs: paddle.Tensor, test_feat: paddle.Tensor) -> paddle.Tensor:
        B, D = test_feat.shape
        if self.sim_metric == 'cos':
            q = F.normalize(test_feat, axis=-1)
            sim = paddle.matmul(q, self.keys_norm, transpose_y=True)
            w = F.softmax(sim / self.tau, axis=-1)
        else:
            q2 = paddle.sum(test_feat * test_feat, axis=-1, keepdim=True)
            k2 = paddle.sum(self.keys * self.keys, axis=-1, keepdim=True).transpose([1,0])
            dot = paddle.matmul(test_feat, self.keys, transpose_y=True)
            dist2 = q2 + k2 - 2.0 * dot
            w = F.softmax(-dist2 / self.tau, axis=-1)
        topk_val, topk_idx = paddle.topk(w, k=min(self.k, w.shape[1]), axis=-1)
        picked_labels = paddle.gather(self.labels, topk_idx.reshape([-1]), axis=0)
        C = self.labels.shape[1]
        picked_labels = picked_labels.reshape([B, -1, C])
        w_norm = topk_val / (paddle.sum(topk_val, axis=-1, keepdim=True) + 1e-9)
        p_knn = paddle.sum(picked_labels * w_norm.unsqueeze(-1), axis=1)
        p_final = (1.0 - self.alpha) * model_probs + self.alpha * p_knn
        return p_final.clip(1e-6, 1-1e-6)