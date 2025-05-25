import paddle
from paddle.nn import Layer, Linear
from paddle_geometric.graphgym.config import cfg
from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.inits import glorot, zeros
from paddle_geometric.utils import add_remaining_self_loops, scatter


class GeneralConvLayer(MessagePassing):
    r"""A general GNN layer."""
    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, **kwargs):
        super().__init__(aggr=cfg.gnn.agg, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = cfg.gnn.normalize_adj

        self.weight = self.create_parameter(shape=[in_channels, out_channels])
        if cfg.gnn.self_msg == 'concat':
            self.weight_self = self.create_parameter(shape=[in_channels, out_channels])

        if bias:
            self.bias = self.create_parameter(shape=[out_channels], is_bias=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if cfg.gnn.self_msg == 'concat':
            glorot(self.weight_self)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = paddle.ones([edge_index.shape[1]], dtype=dtype)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter(edge_weight, row, 0, num_nodes, reduce='sum')
        deg_inv_sqrt = paddle.pow(deg, -0.5)
        deg_inv_sqrt[paddle.isinf(deg_inv_sqrt)] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None, edge_feature=None):
        if cfg.gnn.self_msg == 'concat':
            x_self = paddle.matmul(x, self.weight_self)
        x = paddle.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.shape[1] != self.cached_num_edges:
                raise RuntimeError(
                    f'Cached {self.cached_num_edges} number of edges, but found {edge_index.shape[1]}.'
                    ' Disable caching by setting `cached=False`.')

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.shape[1]
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.shape[self.node_dim], edge_weight, self.improved, x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        x_msg = self.propagate(edge_index, x=x, norm=norm, edge_feature=edge_feature)
        if cfg.gnn.self_msg == 'none':
            return x_msg
        elif cfg.gnn.self_msg == 'add':
            return x_msg + x
        elif cfg.gnn.self_msg == 'concat':
            return x_msg + x_self
        else:
            raise ValueError(f'self_msg {cfg.gnn.self_msg} not defined')

    def message(self, x_j, norm, edge_feature):
        if edge_feature is None:
            return norm.unsqueeze(-1) * x_j if norm is not None else x_j
        else:
            return norm.unsqueeze(-1) * (x_j + edge_feature) if norm is not None else (x_j + edge_feature)

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels})'


class GeneralEdgeConvLayer(MessagePassing):
    r"""General GNN layer, with edge features."""
    def __init__(self, in_channels, out_channels, edge_dim, improved=False,
                 cached=False, bias=True, **kwargs):
        super().__init__(aggr=cfg.gnn.agg, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = cfg.gnn.normalize_adj
        self.msg_direction = cfg.gnn.msg_direction

        if self.msg_direction == 'single':
            self.linear_msg = Linear(in_channels + edge_dim, out_channels, bias_attr=False)
        else:
            self.linear_msg = Linear(in_channels * 2 + edge_dim, out_channels, bias_attr=False)

        if cfg.gnn.self_msg == 'concat':
            self.linear_self = Linear(in_channels, out_channels, bias_attr=False)

        if bias:
            self.bias = self.create_parameter(shape=[out_channels], is_bias=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = paddle.ones([edge_index.shape[1]], dtype=dtype)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter(edge_weight, row, 0, num_nodes, reduce='sum')
        deg_inv_sqrt = paddle.pow(deg, -0.5)
        deg_inv_sqrt[paddle.isinf(deg_inv_sqrt)] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None, edge_feature=None):
        if self.cached and self.cached_result is not None:
            if edge_index.shape[1] != self.cached_num_edges:
                raise RuntimeError(
                    f'Cached {self.cached_num_edges} number of edges, but found {edge_index.shape[1]}.'
                    ' Disable caching by setting `cached=False`.')

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.shape[1]
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.shape[self.node_dim], edge_weight, self.improved, x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        x_msg = self.propagate(edge_index, x=x, norm=norm, edge_feature=edge_feature)

        if cfg.gnn.self_msg == 'concat':
            x_self = self.linear_self(x)
            return x_self + x_msg
        elif cfg.gnn.self_msg == 'add':
            return x + x_msg
        else:
            return x_msg

    def message(self, x_i, x_j, norm, edge_feature):
        if self.msg_direction == 'both':
            x_j = paddle.concat([x_i, x_j, edge_feature], axis=-1)
        else:
            x_j = paddle.concat([x_j, edge_feature], axis=-1)
        x_j = self.linear_msg(x_j)
        return norm.unsqueeze(-1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels})'
