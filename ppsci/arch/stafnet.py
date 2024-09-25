import paddle
import paddle.nn as nn
import os
import numpy as np
import math
from math import sqrt
# from Embed import DataEmbedding
import argparse
from ppsci.arch import base
from tqdm import tqdm
from pgl.nn.conv import GATv2Conv

class Inception_Block_V1(paddle.nn.Layer):

    def __init__(self, in_channels, out_channels, num_kernels=6,
        init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(paddle.nn.Conv2D(in_channels=in_channels,
                out_channels=out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = paddle.nn.LayerList(sublayers=kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.sublayers():
            if isinstance(m, paddle.nn.Conv2D):

                init_kaimingNormal = paddle.nn.initializer.KaimingNormal(fan_in=None,
                            negative_slope=0.0, nonlinearity='relu')
                init_kaimingNormal(m.weight)
                
                if m.bias is not None:
                    init_Constant = paddle.nn.initializer.Constant(value=0)
                    init_Constant(m.bias)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = paddle.stack(x=res_list, axis=-1).mean(axis=-1)
        return res

class AttentionLayer(paddle.nn.Layer):

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None
        ):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or d_model // n_heads
        d_values = d_values or d_model // n_heads
        self.inner_attention = attention
        self.query_projection = paddle.nn.Linear(in_features=d_model,
            out_features=d_keys * n_heads)
        self.key_projection = paddle.nn.Linear(in_features=d_model,
            out_features=d_keys * n_heads)
        self.value_projection = paddle.nn.Linear(in_features=d_model,
            out_features=d_values * n_heads)
        self.out_projection = paddle.nn.Linear(in_features=d_values *
            n_heads, out_features=d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = tuple(queries.shape)
        _, S, _ = tuple(keys.shape)
        H = self.n_heads
        queries = self.query_projection(queries).reshape((B, L, H, -1))
        keys = self.key_projection(keys).reshape((B, S, H, -1))
        values = self.value_projection(values).reshape((B, S, H, -1))
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.reshape((B, L, -1))
        return self.out_projection(out), attn

class ProbAttention(paddle.nn.Layer):

    def __init__(self, mask_flag=True, factor=5, scale=None,
        attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = paddle.nn.Dropout(p=attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = tuple(K.shape)
        _, _, L_Q, _ = tuple(Q.shape)
        K_expand = K.unsqueeze(axis=-3).expand(shape=[B, H, L_Q, L_K, E])
        index_sample = paddle.randint(low=0, high=L_K, shape=[L_Q, sample_k])
        # index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, paddle.arange(end=L_Q).unsqueeze(axis=1), index_sample, :]

        x = K_sample
        perm_5 = list(range(x.ndim))
        perm_5[-2] = -1
        perm_5[-1] = -2
        Q_K_sample = paddle.matmul(x=Q.unsqueeze(axis=-2), y=x.transpose(
            perm=perm_5)).squeeze()
        M = Q_K_sample.max(-1)[0] - paddle.divide(x=Q_K_sample.sum(axis=-1),
            y=paddle.to_tensor(L_K,dtype ='float32') )
        M_top = M.topk(k=n_top, sorted=False)[1]
        Q_reduce = Q[paddle.arange(end=B)[:, None, None], paddle.arange(end
            =H)[None, :, None], M_top, :]
        x = K
        perm_6 = list(range(x.ndim))
        perm_6[-2] = -1
        perm_6[-1] = -2
        Q_K = paddle.matmul(x=Q_reduce, y=x.transpose(perm=perm_6))
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = tuple(V.shape)
        if not self.mask_flag:
            V_sum = V.mean(axis=-2)
            contex = V_sum.unsqueeze(axis=-2).expand(shape=[B, H, L_Q,
                tuple(V_sum.shape)[-1]]).clone()
        else:
            assert L_Q == L_V
            contex = V.cumsum(axis=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = tuple(V.shape)
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.place)
            scores.masked_fill_(mask=attn_mask.mask, value=-np.inf)
        attn = paddle.nn.functional.softmax(x=scores, axis=-1)
        context_in[paddle.arange(end=B)[:, None, None], paddle.arange(end=H
            )[None, :, None], index, :] = paddle.matmul(x=attn, y=V).astype(
            dtype=context_in.dtype)
        if self.output_attention:
            attns = (paddle.ones(shape=[B, H, L_Q, L_V]) / L_V).astype(dtype
                =attn.dtype).to(attn.place)
            attns[paddle.arange(end=B)[:, None, None], paddle.arange(end=H)
                [None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = tuple(queries.shape)
        _, L_K, _, _ = tuple(keys.shape)
        x = queries
        perm_7 = list(range(x.ndim))
        perm_7[2] = 1
        perm_7[1] = 2
        queries = x.transpose(perm=perm_7)
        x = keys
        perm_8 = list(range(x.ndim))
        perm_8[2] = 1
        perm_8[1] = 2
        keys = x.transpose(perm=perm_8)
        x = values
        perm_9 = list(range(x.ndim))
        perm_9[2] = 1
        perm_9[1] = 2
        values = x.transpose(perm=perm_9)
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part,
            n_top=u)
        scale = self.scale or 1.0 / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top,
            index, L_Q, attn_mask)
        return context, attn


def FFT_for_Period(x, k=2):
    xf = paddle.fft.rfft(x=x, axis=1)
    frequency_list = paddle.abs(xf).mean(axis=0).mean(axis=-1)
    frequency_list[0] = 0
    _, top_list = paddle.topk(k=k, x=frequency_list)
    top_list = top_list.detach().cpu().numpy()
    period = tuple(x.shape)[1] // top_list
    return period, paddle.index_select(paddle.abs(xf).mean(axis=-1), paddle.to_tensor(top_list), axis=1)


class TimesBlock(paddle.nn.Layer):

    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.conv = paddle.nn.Sequential(Inception_Block_V1(configs.d_model,
            configs.d_ff, num_kernels=configs.num_kernels), paddle.nn.GELU(
            ), Inception_Block_V1(configs.d_ff, configs.d_model,
            num_kernels=configs.num_kernels))

    def forward(self, x):
        B, T, N = tuple(x.shape)
        period_list, period_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = ((self.seq_len + self.pred_len) // period + 1
                    ) * period
                padding = paddle.zeros(shape=[tuple(x.shape)[0], length - (
                    self.seq_len + self.pred_len), tuple(x.shape)[2]])
                out = paddle.concat(x=[x, padding], axis=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            out = out.reshape((B, length // period, period, N)).transpose(perm=[0, 3, 1, 2])
            out = self.conv(out)
            out = out.transpose(perm=[0, 2, 3, 1]).reshape((B, -1, N))
            res.append(out[:, :self.seq_len + self.pred_len, :])
        res = paddle.stack(x=res, axis=-1)
        period_weight_raw = period_weight
        period_weight = paddle.nn.functional.softmax(x=period_weight, axis=1)
        # print(period_weight.unsqueeze(axis=1).unsqueeze(axis=1).shape)
        # period_weight = period_weight.unsqueeze(axis=1).unsqueeze(axis=1
        #     ).repeat(1, T, N, 1)
        period_weight = paddle.tile(period_weight.unsqueeze(axis=1).unsqueeze(axis=1
            ), (1, T, N, 1))
        res = paddle.sum(x=res * period_weight, axis=-1)
        res = res + x
        return res, period_list, period_weight_raw




def compared_version(ver1, ver2):
    """
    :param ver1
    :param ver2
    :return: ver1< = >ver2 False/True
    """
    list1 = str(ver1).split('.')
    list2 = str(ver2).split('.')
    for i in (range(len(list1)) if len(list1) < len(list2) else range(len(
        list2))):
        if int(list1[i]) == int(list2[i]):
            pass
        elif int(list1[i]) < int(list2[i]):
            return -1
        else:
            return 1
    if len(list1) == len(list2):
        return True
    elif len(list1) < len(list2):
        return False
    else:
        return True


class PositionalEmbedding(paddle.nn.Layer):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = paddle.zeros(shape=[max_len, d_model]).astype(dtype='float32')
        pe.stop_gradient = True
        position = paddle.arange(start=0, end=max_len).astype(dtype='float32'
            ).unsqueeze(axis=1)
        div_term = (paddle.arange(start=0, end=d_model, step=2).astype(
            dtype='float32') * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = paddle.sin(x=position * div_term)
        pe[:, 1::2] = paddle.cos(x=position * div_term)
        pe = pe.unsqueeze(axis=0)
        self.register_buffer(name='pe', tensor=pe)

    def forward(self, x):
        return self.pe[:, :x.shape[1]]


class TokenEmbedding(paddle.nn.Layer):

    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if compared_version(paddle.__version__, '1.5.0') else 2
        self.tokenConv = paddle.nn.Conv1D(in_channels=c_in, out_channels=
            d_model, kernel_size=3, padding=padding, padding_mode=
            'circular', bias_attr=False)
        for m in self.sublayers():
            if isinstance(m, paddle.nn.Conv1D):
                init_KaimingNormal = paddle.nn.initializer.KaimingNormal(
                    nonlinearity='leaky_relu')
                init_KaimingNormal(m.weight)

    def forward(self, x):
        x = self.tokenConv(x.transpose(perm=[0, 2, 1]))
        perm_13 = list(range(x.ndim))
        perm_13[1] = 2
        perm_13[2] = 1
        x = x.transpose(perm=perm_13)
        return x


class FixedEmbedding(paddle.nn.Layer):

    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        w = paddle.zeros(shape=[c_in, d_model]).astype(dtype='float32')
        w.stop_gradient = True
        position = paddle.arange(start=0, end=c_in).astype(dtype='float32'
            ).unsqueeze(axis=1)
        div_term = (paddle.arange(start=0, end=d_model, step=2).astype(
            dtype='float32') * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = paddle.sin(x=position * div_term)
        w[:, 1::2] = paddle.cos(x=position * div_term)
        self.emb = paddle.nn.Embedding(num_embeddings=c_in, embedding_dim=
            d_model)
        out_3 = paddle.create_parameter(shape=w.shape, dtype=w.numpy().
            dtype, default_initializer=paddle.nn.initializer.Assign(w))
        out_3.stop_gradient = not False
        self.emb.weight = out_3

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(paddle.nn.Layer):

    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        minute_size = 4
        hour_size = 24
        weeknum_size = 53
        weekday_size = 7
        day_size = 32
        month_size = 13
        Embed = (FixedEmbedding if embed_type == 'fixed' else paddle.nn.
            Embedding)
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.weeknum_embed = Embed(weeknum_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
        self.Temporal_feature = ['month', 'day', 'week', 'weekday', 'hour']

    def forward(self, x):
        x = x.astype(dtype='int64')
        for idx, freq in enumerate(self.Temporal_feature):
            if freq == 'year':
                pass
            elif freq == 'month':
                month_x = self.month_embed(x[:, :, idx])
            elif freq == 'day':
                day_x = self.day_embed(x[:, :, idx])
            elif freq == 'week':
                weeknum_x = self.weeknum_embed(x[:, :, idx])
            elif freq == 'weekday':
                weekday_x = self.weekday_embed(x[:, :, idx])
            elif freq == 'hour':
                hour_x = self.hour_embed(x[:, :, idx])
        return hour_x + weekday_x + weeknum_x + day_x + month_x


class TimeFeatureEmbedding(paddle.nn.Layer):

    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3,
            'b': 3}
        d_inp = freq_map[freq]
        self.embed = paddle.nn.Linear(in_features=d_inp, out_features=
            d_model, bias_attr=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(paddle.nn.Layer):

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1
        ):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model,
            embed_type=embed_type, freq=freq
            ) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=
            d_model, embed_type=embed_type, freq=freq)
        self.dropout = paddle.nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark
            ) + self.position_embedding(x)
        return self.dropout(x)



class DataEmbedding_wo_pos(paddle.nn.Layer):

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1
        ):
        super(DataEmbedding_wo_pos, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model,
            embed_type=embed_type, freq=freq
            ) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=
            d_model, embed_type=embed_type, freq=freq)
        self.dropout = paddle.nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)

class GAT_Encoder(paddle.nn.Layer):

    def __init__(self, input_dim, hid_dim, edge_dim, gnn_embed_dim, dropout):
        super(GAT_Encoder, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.conv1 = GATv2Conv(input_dim, hid_dim, )
        self.conv2 = GATv2Conv(hid_dim, hid_dim * 2, )
        self.conv3 = GATv2Conv(hid_dim * 2, gnn_embed_dim,)

    def forward(self,graph, feature, ):
        x = self.conv1(graph, feature )
        x = self.relu(x)
        # x = x.relu()
        x = self.conv2(graph, x)
        x = self.relu(x)
        x = self.conv3(graph, x)
        x = self.dropout(x)

        return x


class STAFNet(base.Arch):

    def __init__(self, configs, **kwargs):
        super(STAFNet, self).__init__()
        # configs = argparse.Namespace(**configs)
        self.device = str('cuda').replace('cuda', 'gpu')
        self.configs = configs
        self.task_name = configs.task_name
        if hasattr(configs, 'output_attention'):
            self.output_attention = configs.output_attention
        else:
            self.output_attention = False
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.dec_in = configs.dec_in
        self.gat_embed_dim = configs.enc_in
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model,
            configs.embed, configs.freq, configs.dropout)
        self.aq_gat_node_num = configs.aq_gat_node_num
        self.aq_gat_node_features = configs.aq_gat_node_features
        self.aq_GAT = GAT_Encoder(configs.aq_gat_node_features, configs.
            gat_hidden_dim, configs.gat_edge_dim, self.gat_embed_dim,
            configs.dropout).to(self.device)
        self.mete_gat_node_num = configs.mete_gat_node_num
        self.mete_gat_node_features = configs.mete_gat_node_features
        self.mete_GAT = GAT_Encoder(configs.mete_gat_node_features, configs
            .gat_hidden_dim, configs.gat_edge_dim, self.gat_embed_dim,
            configs.dropout).to(self.device)
        self.pos_fc = paddle.nn.Linear(in_features=2, out_features=configs.
            gat_embed_dim, bias_attr=True)
        self.fusion_Attention = AttentionLayer(ProbAttention(False, configs
            .factor, attention_dropout=configs.dropout, output_attention=
            self.output_attention), configs.gat_embed_dim, configs.n_heads)
        self.model = paddle.nn.LayerList(sublayers=[TimesBlock(configs) for
            _ in range(configs.e_layers)])
        self.layer = configs.e_layers
        self.layer_norm = paddle.nn.LayerNorm(normalized_shape=configs.d_model)
        self.predict_linear = paddle.nn.Linear(in_features=self.seq_len,
            out_features=self.pred_len + self.seq_len)
        self.projection = paddle.nn.Linear(in_features=configs.d_model,
            out_features=configs.c_out, bias_attr=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(axis=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = paddle.sqrt(x=paddle.var(x=x_enc, axis=1, keepdim=True,
            unbiased=False) + 1e-05)
        x_enc /= stdev
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.transpose(perm=[0, 2, 1])
            ).transpose(perm=[0, 2, 1])
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)
        # dec_out = dec_out * stdev[:, 0, :].unsqueeze(axis=1).repeat(1, self
        #     .pred_len + self.seq_len, 1)
        # dec_out = dec_out + means[:, 0, :].unsqueeze(axis=1).repeat(1, self
        #     .pred_len + self.seq_len, 1)
        
        dec_out = paddle.tile(dec_out * stdev[:, 0, :].unsqueeze(axis=1), (1, self.pred_len + self.seq_len, 1))
        dec_out = paddle.tile(dec_out + means[:, 0, :].unsqueeze(axis=1), (1, self.pred_len + self.seq_len, 1))
        
        return dec_out

    def aq_gat(self, G):
        # x = G.x[:, -self.aq_gat_node_features:].to(self.device)
        # edge_index = G.edge_index.to(self.device)
        # edge_attr = G.edge_attr.to(self.device)
        g_batch = G.num_graph
        batch_size = int(g_batch/self.seq_len)
        gat_output = self.aq_GAT(G, G.node_feat["feature"][:, -self.aq_gat_node_features:])
        gat_output = gat_output.reshape((batch_size, self.seq_len, self.
            aq_gat_node_num, self.gat_embed_dim))
        gat_output = paddle.flatten(x=gat_output, start_axis=0, stop_axis=1)
        return gat_output

    def mete_gat(self, G):
        # x = G.x[:, -self.mete_gat_node_features:].to(self.device)
        # edge_index = G.edge_index.to(self.device)
        # edge_attr = G.edge_attr.to(self.device)
        g_batch = G.num_graph
        batch_size = int(g_batch/self.seq_len)
        gat_output = self.mete_GAT(G, G.node_feat["feature"][:, -self.mete_gat_node_features:])
        gat_output = gat_output.reshape((batch_size, self.seq_len, self.
            mete_gat_node_num, self.gat_embed_dim))
        gat_output = paddle.flatten(x=gat_output, start_axis=0, stop_axis=1)
        return gat_output

    def norm_pos(self, A, B):
        # paddle.mean(x)
        A_mean = paddle.mean(A, axis=0)
        A_std = paddle.std(A, axis=0)

        A_norm = (A - A_mean) / A_std
        B_norm = (B - A_mean) / A_std
        return A_norm, B_norm

    def forward(self, Data, mask=None):
        aq_G = Data['aq_G']
        mete_G = Data['mete_G']
        aq_gat_output = self.aq_gat(aq_G)
        mete_gat_output = self.mete_gat(mete_G)
        aq_pos, mete_pos = self.norm_pos(aq_G.node_feat["pos"], mete_G.node_feat["pos"])
        # aq_pos = paddle.view(self.pos_fc(aq_pos),[-1, self.aq_gat_node_num, self.gat_embed_dim])
        #
        # mete_pos = paddle.view(self.pos_fc(mete_pos),[-1, self.mete_gat_node_num, self.gat_embed_dim])
        aq_pos = self.pos_fc(aq_pos).reshape((-1, self.aq_gat_node_num, self.
                                          gat_embed_dim))
        mete_pos = self.pos_fc(mete_pos).reshape((-1, self.mete_gat_node_num,
                                              self.gat_embed_dim))
        fusion_out, attn = self.fusion_Attention(aq_pos, mete_pos, mete_gat_output, attn_mask=None)
        aq_gat_output = aq_gat_output + fusion_out
        aq_gat_output = aq_gat_output.reshape((-1, self.seq_len, self.
            aq_gat_node_num, self.gat_embed_dim))
        x = aq_gat_output
        perm_0 = list(range(x.ndim))
        perm_0[1] = 2
        perm_0[2] = 1
        aq_gat_output = paddle.transpose(x=x, perm=perm_0)
        aq_gat_output = paddle.flatten(x=aq_gat_output, start_axis=0,
            stop_axis=1)
        train_data = Data['aq_train_data']
        x = train_data
        perm_1 = list(range(x.ndim))
        perm_1[1] = 2
        perm_1[2] = 1
        train_data = paddle.transpose(x=x, perm=perm_1)
        train_data = paddle.flatten(x=train_data, start_axis=0, stop_axis=1)
        x_enc = train_data[:, :self.seq_len, -self.dec_in:]
        x_mark_enc = train_data[:, :self.seq_len, 1:6]
        # x_dec = paddle.zeros_like(x=train_data[:, -self.pred_len:, -self.
        #     dec_in:]).astype(dtype='float32')
        # x_dec = paddle.concat(x=[train_data[:, self.seq_len - self.
        #     label_len:self.seq_len, -self.dec_in:], x_dec], axis=1).astype(
        #     dtype='float32')  #.to(self.device)
        # x_mark_dec = train_data[:, -self.pred_len - self.label_len:, 1:6]
        means = x_enc.mean(axis=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = paddle.sqrt(x=paddle.var(x=x_enc, axis=1, keepdim=True,
            unbiased=False) + 1e-05)
        x_enc /= stdev
        enc_out = self.enc_embedding(aq_gat_output, x_mark_enc)
        enc_out = self.predict_linear(enc_out.transpose(perm=[0, 2, 1])
            ).transpose(perm=[0, 2, 1])
        for i in range(self.layer):
            enc_out, period_list, period_weight = self.model[i](enc_out)
            enc_out = self.layer_norm(enc_out)
        dec_out = self.projection(enc_out)
        dec_out = dec_out * paddle.tile( stdev[:, 0, :].unsqueeze(axis=1), (1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out +  paddle.tile(means[:, 0, :].unsqueeze(axis=1), (1, self.pred_len + self.seq_len, 1))
        # dec_out = dec_out * stdev[:, 0, :].unsqueeze(axis=1).repeat(1, self
        #     .pred_len + self.seq_len, 1)
        # dec_out = dec_out + means[:, 0, :].unsqueeze(axis=1).repeat(1, self
        #     .pred_len + self.seq_len, 1)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], None
        else:
            return dec_out[:, -self.pred_len:, :], period_list, period_weight
