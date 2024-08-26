# STAFNet

## 1. 背景介绍

近些年，全球城市化和工业化不可避免地导致了严重的空气污染问题。心脏病、哮喘和肺癌等非传染性疾病的高发与暴露于空气污染直接相关。因此，空气质量预测已成为公共卫生、国民经济和城市管理的研究热点。目前已经建立了大量监测站来监测空气质量，并将其地理位置和历史观测数据合并为时空数据。然而，由于空气污染形成和扩散的高度复杂性，空气质量预测仍然面临着一些挑战。

首先，空气中污染物的排放和扩散会导致邻近地区的空气质量迅速恶化，这一现象在托布勒地理第一定律中被描述为空间依赖关系，建立空间关系模型对于预测空气质量至关重要。然而，由于空气监测站的地理分布稀疏，要捕捉数据中内在的空间关联具有挑战性。其次，空气质量受到多源复杂因素的影响，尤其是气象条件。例如，长时间的小风或静风会抑制空气污染物的扩散，而自然降雨则在清除和冲刷空气污染物方面发挥作用。然而，空气质量站和气象站位于不同区域，导致多模态特征不对齐。融合不对齐的多模态特征并获取互补信息以准确预测空气质量是另一个挑战。最后但并非最不重要的一点是，空气质量的变化具有明显的多周期性特征。利用这一特点对提高空气质量预测的准确性非常重要，但也具有挑战性。

针对空气质量预测提出了许多研究。早期的方法侧重于学习单个观测站观测数据的时间模式，而放弃了观测站之间的空间关系。最近，由于图神经网络（GNN）在处理非欧几里得图结构方面的有效性，越来越多的方法采用 GNN 来模拟空间依赖关系。这些方法将车站位置作为上下文特征，隐含地建立空间依赖关系模型，没有充分利用车站位置和车站之间关系所包含的宝贵空间信息。此外，现有的时空 GNN 缺乏在错位图中融合多个特征的能力。因此，大多数方法都需要额外的插值算法，以便在早期阶段将气象特征与 AQ 特征进行对齐和连接。这种方法消除了空气质量站和气象站之间的空间和结构信息，还可能引入噪声导致误差累积。此外，在空气质量预测中利用多周期性的问题仍未得到探索。

## 2. 模型原理

STAFNet是一个新颖的多模式预报框架--时空感知融合网络来预测空气质量。STAFNet 由三个主要部分组成：空间感知 GNN、跨图融合关注机制和 TimesNet 。具体来说，为了捕捉各站点之间的空间关系，我们首先引入了空间感知 GNN，将空间信息明确纳入信息传递和节点表示中。为了全面表示气象影响，我们随后提出了一种基于交叉图融合关注机制的多模态融合策略，在不同类型站点的数量和位置不一致的情况下，将气象数据整合到 AQ 数据中。受多周期分析的启发，我们采用 TimesNet 将时间序列数据分解为不同频率的周期信号，并分别提取时间特征。

本章节仅对 STAFNet的模型原理进行简单地介绍，详细的理论推导请阅读 STAFNet: Spatiotemporal-Aware Fusion Network for Air Quality Prediction

模型的总体结构如图所示：

![image-20240530165151443](C:\Users\pal\AppData\Roaming\Typora\typora-user-images\image-20240530165151443.png)

<div align = "center">STAFNet网络模型</div>

STAFNet 包含三个模块，分别将空间信息、气象信息和历史信息融合到空气质量特征表征中。首先模型的输入：过去T个时刻的**空气质量**数据和**气象**数据，使用两个空间感知 GNN（SAGNN），利用监测站之间的空间关系分别提取空气质量和气象信息。然后，跨图融合注意（CGF）将气象信息融合到空气质量表征中。最后，我们采用 TimesNet 模型来描述空气质量序列的时间动态，并生成多步骤预测。这一推理过程可表述如下，

![image-20240531173333183](C:\Users\pal\AppData\Roaming\Typora\typora-user-images\image-20240531173333183.png)

## 3. 模型构建

### 3.1 数据集介绍

数据集采用了STAFNet处理好的北京空气质量数据集。数据集都包含：

（1）空气质量观测值（即 PM2.5、PM10、O3、NO2、SO2 和 CO）；

（2）气象观测值（即温度、气压、湿度、风速和风向）；

（3）站点位置（即经度和纬度）。

所有空气质量和气象观测数据每小时记录一次。数据集的收集时间为 2021 年 1 月 24 日至 2023 年 1 月 19 日，按 9:1的比例将数据分为训练集和测试集。空气质量观测数据来自国家城市空气质量实时发布平台，气象观测数据来自中国气象局。数据集的具体细节如下表所示，

<div>			<!--块级封装-->     <center>	<!--将图片和文字居中-->     <img src="C:\Users\pal\AppData\Roaming\Typora\typora-user-images\image-20240530104042194.png" alt="image-20240530104042194" style="zoom: 25%;" />     <br>		<!--换行-->     北京空气质量数据集	<!--标题-->     </center> </div>

数据加载代码如下：

```python

def gat_lstmcollate_fn(data):
    aq_train_data = []
    mete_train_data = []
    aq_g_list = []
    mete_g_list = []
    edge_index = []
    edge_attr = []
    pos = []
    label = []
    for unit in data:
        aq_train_data.append(unit[0]['aq_train_data'])
        mete_train_data.append(unit[0]['mete_train_data'])
        aq_g_list = aq_g_list + unit[0]['aq_g_list']
        mete_g_list = mete_g_list + unit[0]['mete_g_list']
        label.append(unit[1])
    label = paddle.stack(x=label)
    x = label
    perm_1 = list(range(x.ndim))
    perm_1[1] = 2
    perm_1[2] = 1
    label = paddle.transpose(x=x, perm=perm_1)
    label = paddle.flatten(x=label, start_axis=0, stop_axis=1)
    return {'aq_train_data': paddle.stack(x=aq_train_data),
        'mete_train_data': paddle.stack(x=mete_train_data), 
        'aq_G': pgl.graph.Graph.batch(aq_g_list), 
        'mete_G': pgl.graph.Graph.batch(mete_g_list)
        }, label


class pygmmdataLoader(DataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, args, data_dir, batch_size, shuffle=True,
        num_workers=1, training=True, T=24, t=12, collate_fn=gat_lstmcollate_fn
        ):
        self.T = T
        self.t = t
        self.dataset = AQGDataset(args=args, data_path=data_dir)
        # if get_world_size() > 1:
        #     sampler = paddle.io.DistributedBatchSampler(dataset=self.
        #         dataset, shuffle=shuffle, batch_size=1)
        # else:
        #     sampler = None
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers = num_workers, collate_fn=collate_fn)

class AQGDataset(Dataset):

    def __init__(self, args, data_path, root='/PROTEINS_full', filepath=
        '/PROTEINS_full/raw', name='custom', use_edge_attr=True, transform=
        None, pre_transform=None, pre_filter=None):
        """
        root: 数据集保存的地方。
        会产生两个文件夹：
          raw_dir(downloaded dataset) 和 processed_dir(processed data)。
        """
        self.name = name
        self.root = root
        self.filepath = filepath
        self.use_edge_attr = use_edge_attr
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        
        super().__init__( )
        if data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                self.data = pandas.read_pickle(f)
        self.metedata = self.data['metedata']
        self.AQdata = self.data['AQdata']
        self.AQStation_imformation = self.data['AQStation_imformation']
        self.meteStation_imformation = self.data['meteStation_imformation']
        mete_coords = np.array(self.meteStation_imformation.loc[:, ['经度','纬度']]).astype('float64')
        AQ_coords = np.array(self.AQStation_imformation.iloc[:, -2:]).astype('float64')
        self.aq_edge_index, self.aq_edge_attr, self.aq_node_coords = (self.get_edge_attr(np.array(self.AQStation_imformation.iloc[:, -2:]).astype('float64')))
        (self.mete_edge_index, self.mete_edge_attr, self.mete_node_coords) = (self.get_edge_attr(np.array(self.meteStation_imformation.loc[:, ['经度', '纬度']]).astype('float64')))

        self.lut = self.find_nearest_point(AQ_coords, mete_coords)
        # self.AQdata = np.concatenate((self.AQdata, self.metedata[:, self.lut, -7:]), axis=2)


    def __len__(self):
        return len(self.AQdata) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        input_data = {}
        aq_train_data = paddle.to_tensor(data=self.AQdata[idx:idx + self.seq_len + self.pred_len]).astype(dtype='float32')
        mete_train_data = paddle.to_tensor(data=self.metedata[idx:idx + self.seq_len + self.pred_len]).astype(dtype='float32')
        aq_g_list = [pgl.Graph(num_nodes=s.shape[0],
            edges=self.aq_edge_index,
            node_feat={
                "feature": s,
                "pos": self.aq_node_coords.astype(dtype='float32')
            },
            edge_feat={
                "edge_feature": self.aq_edge_attr.astype(dtype='float32')
            })for s in aq_train_data[:self.seq_len]]
        
        mete_g_list = [pgl.Graph(num_nodes=s.shape[0],
            edges=self.mete_edge_index,
            node_feat={
                "feature": s,
                "pos": self.mete_node_coords.astype(dtype='float32')
            },
            edge_feat={
                "edge_feature": self.mete_edge_attr.astype(dtype='float32')
            })for s in mete_train_data[:self.seq_len]]
        
        # aq_g_list = [pgl.Graph(x=s, edge_index=self.aq_edge_index, edge_attr=self.aq_edge_attr.astype(dtype='float32'), pos=self.aq_node_coords.astype(dtype='float32')) for s in aq_train_data[:self.seq_len]]
        # mete_g_list = [pgl.Graph((x=s, edge_index=self.
        #     mete_edge_index, edge_attr=self.mete_edge_attr.astype(dtype=
        #     'float32'), pos=self.mete_node_coords.astype(dtype='float32')) for
        #     s in mete_train_data[:self.seq_len]]
        
        
        label = aq_train_data[-self.pred_len:, :, -7:]
        data = {'aq_train_data': aq_train_data, 'mete_train_data':
            mete_train_data, 'aq_g_list': aq_g_list, 'mete_g_list': mete_g_list
            }
        return data, label

    def get_edge_attr(self, node_coords, threshold=0.2):
        node_coords = paddle.to_tensor(data=node_coords)
        dist_matrix = cdist(node_coords, node_coords)
        edge_index = np.where(dist_matrix < threshold)
        edge_index = paddle.to_tensor(data=edge_index, dtype='int64')
        start_nodes, end_nodes = edge_index
        edge_lengths = dist_matrix[start_nodes, end_nodes]
        edge_directions = node_coords[end_nodes] - node_coords[start_nodes]
        edge_attr = paddle.to_tensor(data=np.concatenate((edge_lengths[:,
            np.newaxis], edge_directions), axis=1))
        return edge_index, edge_attr, node_coords

    def find_nearest_point(self, A, B):
        nearest_indices = []
        for a in A:
            distances = [np.linalg.norm(a - b) for b in B]
            nearest_indices.append(np.argmin(distances))
        return nearest_indices

```

```python
dataloader = pygmmdataLoader(args,**dataLoader_args)
valid_loader = pygmmdataLoader(args,**valid_loader_args)
```



### 3.2 模型构建

在STAFNet模型中，输入过去72小时35个站点的空气质量数据，预测这35个站点未来48小时的空气质量。

模型代码：

```python
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
        K_sample = K_expand[:, :, paddle.arange(end=L_Q).unsqueeze(axis=1),
            index_sample, :]
        x = K_sample
        perm_5 = list(range(x.ndim))
        perm_5[-2] = -1
        perm_5[-1] = -2
        Q_K_sample = paddle.matmul(x=Q.unsqueeze(axis=-2), y=x.transpose(
            perm=perm_5)).squeeze()
        M = Q_K_sample.max(-1)[0] - paddle.divide(x=Q_K_sample.sum(axis=-1),
            y=paddle.to_tensor(L_K))
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
        period_weight = paddle.tile(period_weight.unsqueeze(axis=1).unsqueeze(axis=1), repeat_times=(1, T, N, 1))
        # period_weight = period_weight.unsqueeze(axis=1).unsqueeze(axis=1
        #     ).repeat(1, T, N, 1)
        res = paddle.sum(x=res * period_weight, axis=-1)
        res = res + x
        return res, period_list, period_weight_raw


class Gat_TimesNet_mm(paddle.nn.Layer):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs, **kwargs):
        super(Gat_TimesNet_mm, self).__init__()
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
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(axis=1).repeat(1, self
            .pred_len + self.seq_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(axis=1).repeat(1, self
            .pred_len + self.seq_len, 1)
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
        # fusion_out, attn = self.fusion_Attention(aq_pos, mete_pos, mete_gat_output, attn_mask=None)
        # aq_gat_output = aq_gat_output + fusion_out
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
        dec_out = dec_out * paddle.tile(stdev[:, 0, :].unsqueeze(axis=1), (1, self
            .pred_len + self.seq_len, 1))
        # dec_out = dec_out * stdev[:, 0, :].unsqueeze(axis=1).repeat(1, self
        #     .pred_len + self.seq_len, 1)
        dec_out = dec_out + paddle.tile(means[:, 0, :].unsqueeze(axis=1), (1, self
            .pred_len + self.seq_len, 1))
        # dec_out = dec_out + means[:, 0, :].unsqueeze(axis=1).repeat(1, self
        #     .pred_len + self.seq_len, 1)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :]
        else:
            return dec_out[:, -self.pred_len:, :], period_list, period_weight



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
```

```python
args=  {
                "task_name": "forecast",
                "output_attention": True,
                "seq_len": 72,
                "label_len": 24,
                "pred_len": 48,

                "aq_gat_node_features" : 7,
                "aq_gat_node_num": 35,

                "mete_gat_node_features" : 7,
                "mete_gat_node_num": 18,

                "gat_hidden_dim": 32,
                "gat_edge_dim": 3,
                "gat_embed_dim": 32,
 
                "e_layers": 1,
                "enc_in": 32,
                "dec_in": 7,
                "c_out": 7,
                "d_model": 16 ,
                "embed": "fixed",
                "freq": "t",
                "dropout": 0.05,
                "factor": 3,
                "n_heads": 4,

                "d_ff": 32 ,
                "num_kernels": 6,
                "top_k": 4
            }
model = Gat_TimesNet_mm(args)
```

这样我们就实例化出了一个 STAFNet模型。

### 3.3  模型训练评估、可视化

本案例训练过程中会按照一定的训练轮数间隔，使用验证集评估当前模型的训练情况。在这里使用了自定义的评价指标分别是 `MAE`和`RMSE`，代码如下：

```python
def AQI_RMSE(output, target):
    return paddle.sqrt(x=paddle.nn.functional.mse_loss(input=output[:, :, 0
        ], label=target[:, :, 0]))


def AQI_RMSE_112(output, target):
    return paddle.sqrt(x=paddle.nn.functional.mse_loss(input=output[:, :12,
        0], label=target[:, :12, 0]))


def AQI_RMSE_1324(output, target):
    return paddle.sqrt(x=paddle.nn.functional.mse_loss(input=output[:, 12:
        24, 0], label=target[:, 12:24, 0]))


def AQI_RMSE_2548(output, target):
    return paddle.sqrt(x=paddle.nn.functional.mse_loss(input=output[:, 24:,
        0], label=target[:, 24:, 0]))


def AQI_MAE(output, target):
    return paddle.nn.functional.l1_loss(input=output[:, :, 0], label=target
        [:, :, 0])


def AQI_MAE_112(output, target):
    return paddle.nn.functional.l1_loss(input=output[:, :12, 0], label=
        target[:, :12, 0])


def AQI_MAE_1324(output, target):
    return paddle.nn.functional.l1_loss(input=output[:, 12:24, 0], label=
        target[:, 12:24, 0])


def AQI_MAE_2548(output, target):
    return paddle.nn.functional.l1_loss(input=output[:, 24:, 0], label=
        target[:, 24:, 0])
```



## 4. 完整代码

```python
class Trainer:
    def __init__(self,model,data_loader,criterion, optimizer, device, 
                 num_epochs,metric_ftns,valid_data_loader=None,checkpoint_dir='checkpoints/'):
        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.start_epoch = 1
        self.epochs = num_epochs

        self.save_period =1
        self.start_save_epoch=20
        self.checkpoint_dir = checkpoint_dir

        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)

        
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.writer = None
        self.train_metrics = MetricTracker('loss', writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.device = device

    def train(self):
        loss_history = []
        epoch_loss = 0

        for epoch in range(self.start_epoch, self.epochs + 1):
            log = self._train_epoch(epoch)
            best = False
            print(log)
            # for key, value in log.items():
            #     print(' {:s}: {}'.format(str(key), value))    

        # 保存模型
            if (epoch % self.save_period == 0 and epoch > self.start_save_epoch):
                self._save_checkpoint(epoch, save_best=best)
                # filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
                filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
                paddle.save(model, path=filename)
            
        print("Training complete.")
        return loss_history
    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        # arch = type(self.model).__name__
        # state = {'arch': arch, 'epoch': epoch, 'state_dict': {k.replace(
        #     'module.', ''): v for k, v in self.model.state_dict().items()},
        #     'optimizer': self.optimizer.state_dict(), 'monitor_best': self.
        #     mnt_best, 'config': self.config}
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        paddle.save(obj=self.model, path=filename)
        # self.logger.info('Saving checkpoint: {} ...'.format(filename))
        # if save_best:
        #     best_path = str(self.checkpoint_dir / 'model_best.pth')
        #     paddle.save(obj=state, path=best_path)
        #     self.logger.info('Saving current best: model_best.pth ...')

    def _train_epoch(self, epoch): 
        self.model.train()
        self.train_metrics.reset()
        pbar = tqdm(enumerate(self.data_loader), total=len(self.data_loader))
        for batch_idx, (data, target) in pbar:
            # for key, value in data.items():
            #     if paddle.is_tensor(x=value):
            #         data[key] = value.to(self.device)
            # target = target.to(self.device)

            self.optimizer.clear_grad()
            output = self.model(data)
            loss = self.criterion(output[:, :], target[:, :, :1])
            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())
            pbar.set_description('Train Epoch: {} {} '.format(epoch, self._progress(batch_idx + 1)))
            pbar.set_postfix(train_loss=loss.item())
            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            outputs, targets = self._valid_epoch(epoch)
            # outputs = [output.to(self.device) for output in outputs]
            # targets = [target.to(self.device) for target in targets]
            # for met in self.metric_ftns:
            #     self.valid_metrics.update(met.__name__, met(paddle.
            #         concat(x=outputs, axis=0), paddle.concat(x=targets,
            #         axis=0)))
            # val_loss = self.criterion(output[:, :], target[:, :, :1])
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(outputs, 
                                                            targets))
            val_log = self.valid_metrics.result()
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            # pbar.set_description('Val Epoch: {} {} '.format(epoch, self._progress(batch_idx + 1)))
            # pbar.set_postfix(train_loss=val_loss.item())
        return log
    def _valid_epoch(self, epoch):
        self.model.eval()
        outputs = []
        targets = []
        with paddle.no_grad():
            pbar = tqdm(enumerate(self.valid_data_loader), total=len(self.valid_data_loader))
            for batch_idx, (data, target) in pbar:
                output = self.model(data)
                outputs.append(output.clone()[:, :])
                targets.append(target.clone())
        pbar.set_description('Val Epoch: {} {} '.format(epoch, self._progress(batch_idx + 1)))

        return paddle.concat(x=outputs, axis=0), paddle.concat(x=targets, axis=0)
                


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    
criterion = paddle.nn.MSELoss()
optimizer = paddle.optimizer.Adam(learning_rate=lr,
                                parameters=model.parameters())
num_epochs = 40
metric_ftns = [getattr(module_metric, met) for met in metrics]

trainer = Trainer(model,dataloader,criterion, optimizer, device, num_epochs,metric_ftns,valid_loader)
trainer.train()
```

## 5. 相关问题

1. 目前使用的paddlepaddle版本为2.4.2，STAFNet模型中跨图融合注意力部分涉及到**对高维变量的多维同时索引**的操作，2.4.2版本paddlepaddle还不支持这样的操作，目前最新版paddlepaddle==2.6.1已经支持。

2. 但STAFNet模型中图网络部分使用paddlepaddle框架下图网络框架pgl实现，pgl底层中使用了2.4paddlepaddle版本中的接口fluid，在后续的版本被废弃了。

解决方案：使用最新版paddlepaddle==2.6.1，使用PGL时需要把调用那么的fluid换成base。

## 6. 参考资料

STAFNet: Spatiotemporal-Aware Fusion Network for Air Quality Prediction

