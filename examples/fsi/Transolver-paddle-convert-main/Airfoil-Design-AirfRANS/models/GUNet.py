import paddle
import torch_geometric.nn as nng
import random


def DownSample(id, x, edge_index, pos_x, pool, pool_ratio, r, max_neighbors):
    y = x.clone()
    n = int(x.shape[0])
    if pool is not None:
        y, _, _, _, id_sampled, _ = pool(y, edge_index)
    else:
        k = int((pool_ratio * paddle.to_tensor(data=n, dtype='float32')).ceil()
            )
        id_sampled = random.sample(range(n), k)
        id_sampled = paddle.to_tensor(data=id_sampled, dtype='int64')
        y = y[id_sampled]
    pos_x = pos_x[id_sampled]
    id.append(id_sampled)
    edge_index_sampled = nng.radius_graph(x=pos_x.detach(), r=r, loop=True,
        max_num_neighbors=max_neighbors)
    return y, edge_index_sampled


def UpSample(x, pos_x_up, pos_x_down):
    cluster = nng.nearest(pos_x_up, pos_x_down)
    x_up = x[cluster]
    return x_up


class GUNet(paddle.nn.Layer):

    def __init__(self, hparams, encoder, decoder):
        super(GUNet, self).__init__()
        self.L = hparams['nb_scale']
        self.layer = hparams['layer']
        self.pool_type = hparams['pool']
        self.pool_ratio = hparams['pool_ratio']
        self.list_r = hparams['list_r']
        self.size_hidden_layers = hparams['size_hidden_layers']
        self.size_hidden_layers_init = hparams['size_hidden_layers']
        self.max_neighbors = hparams['max_neighbors']
        self.dim_enc = hparams['encoder'][-1]
        self.bn_bool = hparams['batchnorm']
        self.res = hparams['res']
        self.head = 2
        self.activation = paddle.nn.ReLU()
        self.encoder = encoder
        self.decoder = decoder
        self.down_layers = paddle.nn.LayerList()
        if self.pool_type != 'random':
            self.pool = paddle.nn.LayerList()
        else:
            self.pool = None
        if self.layer == 'SAGE':
            self.down_layers.append(nng.SAGEConv(in_channels=self.dim_enc,
                out_channels=self.size_hidden_layers))
            bn_in = self.size_hidden_layers
        elif self.layer == 'GAT':
            self.down_layers.append(nng.GATConv(in_channels=self.dim_enc,
                out_channels=self.size_hidden_layers, heads=self.head,
                add_self_loops=False, concat=True))
            bn_in = self.head * self.size_hidden_layers
        if self.bn_bool == True:
            self.bn = paddle.nn.LayerList()
            self.bn.append(nng.BatchNorm(in_channels=bn_in,
                track_running_stats=False))
        else:
            self.bn = None
        for n in range(1, self.L):
            if self.pool_type != 'random':
                self.pool.append(nng.TopKPooling(in_channels=self.
                    size_hidden_layers, ratio=self.pool_ratio[n - 1],
                    nonlinearity=paddle.nn.functional.sigmoid))
            if self.layer == 'SAGE':
                self.down_layers.append(nng.SAGEConv(in_channels=self.
                    size_hidden_layers, out_channels=2 * self.
                    size_hidden_layers))
                self.size_hidden_layers = 2 * self.size_hidden_layers
                bn_in = self.size_hidden_layers
            elif self.layer == 'GAT':
                self.down_layers.append(nng.GATConv(in_channels=self.head *
                    self.size_hidden_layers, out_channels=self.
                    size_hidden_layers, heads=2, add_self_loops=False,
                    concat=True))
            if self.bn_bool == True:
                self.bn.append(nng.BatchNorm(in_channels=bn_in,
                    track_running_stats=False))
        self.up_layers = paddle.nn.LayerList()
        if self.layer == 'SAGE':
            self.up_layers.append(nng.SAGEConv(in_channels=3 * self.
                size_hidden_layers_init, out_channels=self.dim_enc))
            self.size_hidden_layers_init = 2 * self.size_hidden_layers_init
        elif self.layer == 'GAT':
            self.up_layers.append(nng.GATConv(in_channels=2 * self.head *
                self.size_hidden_layers, out_channels=self.dim_enc, heads=2,
                add_self_loops=False, concat=False))
        if self.bn_bool == True:
            self.bn.append(nng.BatchNorm(in_channels=self.dim_enc,
                track_running_stats=False))
        for n in range(1, self.L - 1):
            if self.layer == 'SAGE':
                self.up_layers.append(nng.SAGEConv(in_channels=3 * self.
                    size_hidden_layers_init, out_channels=self.
                    size_hidden_layers_init))
                bn_in = self.size_hidden_layers_init
                self.size_hidden_layers_init = 2 * self.size_hidden_layers_init
            elif self.layer == 'GAT':
                self.up_layers.append(nng.GATConv(in_channels=2 * self.head *
                    self.size_hidden_layers, out_channels=self.
                    size_hidden_layers, heads=2, add_self_loops=False,
                    concat=True))
            if self.bn_bool == True:
                self.bn.append(nng.BatchNorm(in_channels=bn_in,
                    track_running_stats=False))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        id = []
        edge_index_list = [edge_index.clone()]
        pos_x_list = []
        z = self.encoder(x)
        if self.res:
            z_res = z.clone()
        z = self.down_layers[0](z, edge_index)
        if self.bn_bool == True:
            z = self.bn[0](z)
        z = self.activation(z)
        z_list = [z.clone()]
        for n in range(self.L - 1):
            pos_x = x[:, :2] if n == 0 else pos_x[id[n - 1]]
            pos_x_list.append(pos_x.clone())
            if self.pool_type != 'random':
                z, edge_index = DownSample(id, z, edge_index, pos_x, self.
                    pool[n], self.pool_ratio[n], self.list_r[n], self.
                    max_neighbors)
            else:
                z, edge_index = DownSample(id, z, edge_index, pos_x, None,
                    self.pool_ratio[n], self.list_r[n], self.max_neighbors)
            edge_index_list.append(edge_index.clone())
            z = self.down_layers[n + 1](z, edge_index)
            if self.bn_bool == True:
                z = self.bn[n + 1](z)
            z = self.activation(z)
            z_list.append(z.clone())
        pos_x_list.append(pos_x[id[-1]].clone())
        for n in range(self.L - 1, 0, -1):
            z = UpSample(z, pos_x_list[n - 1], pos_x_list[n])
            z = paddle.concat(x=[z, z_list[n - 1]], axis=1)
            z = self.up_layers[n - 1](z, edge_index_list[n - 1])
            if self.bn_bool == True:
                z = self.bn[self.L + n - 1](z)
            z = self.activation(z) if n != 1 else z
        del (z_list, pos_x_list, edge_index_list)
        if self.res:
            z = z + z_res
        z = self.decoder(z)
        return z
