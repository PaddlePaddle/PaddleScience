import copy
from dataclasses import dataclass

import paddle
import paddle.nn.functional as F

import paddle_geometric as pyg
import paddle_geometric.graphgym.models.act
import paddle_geometric.graphgym.register as register
from paddle_geometric.graphgym.contrib.layer.generalconv import (
    GeneralConvLayer,
    GeneralEdgeConvLayer,
)
from paddle_geometric.graphgym.register import register_layer
from paddle_geometric.nn import Linear as Linear_pyg


@dataclass
class LayerConfig:
    has_batchnorm: bool = False
    bn_eps: float = 1e-5
    bn_mom: float = 0.1
    mem_inplace: bool = False
    dim_in: int = -1
    dim_out: int = -1
    edge_dim: int = -1
    dim_inner: int = None
    num_layers: int = 2
    has_bias: bool = True
    has_l2norm: bool = True
    dropout: float = 0.0
    has_act: bool = True
    final_act: bool = True
    act: str = 'relu'
    keep_edge: float = 0.5


def new_layer_config(
    dim_in: int,
    dim_out: int,
    num_layers: int,
    has_act: bool,
    has_bias: bool,
    cfg,
) -> LayerConfig:
    return LayerConfig(
        has_batchnorm=cfg.gnn.batchnorm,
        bn_eps=cfg.bn.eps,
        bn_mom=cfg.bn.mom,
        mem_inplace=cfg.mem.inplace,
        dim_in=dim_in,
        dim_out=dim_out,
        edge_dim=cfg.dataset.edge_dim,
        has_l2norm=cfg.gnn.l2norm,
        dropout=cfg.gnn.dropout,
        has_act=has_act,
        final_act=True,
        act=cfg.gnn.act,
        has_bias=has_bias,
        keep_edge=cfg.gnn.keep_edge,
        dim_inner=cfg.gnn.dim_inner,
        num_layers=num_layers,
    )


class GeneralLayer(paddle.nn.Layer):
    def __init__(self, name, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.has_l2norm = layer_config.has_l2norm
        has_bn = layer_config.has_batchnorm
        layer_config.has_bias = not has_bn
        self.layer = register.layer_dict[name](layer_config, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(
                paddle.nn.BatchNorm1D(
                    layer_config.dim_out,
                    epsilon=layer_config.bn_eps,
                    momentum=layer_config.bn_mom,
                ))
        if layer_config.dropout > 0:
            layer_wrapper.append(
                paddle.nn.Dropout(
                    p=layer_config.dropout,
                    axis=-1 if layer_config.mem_inplace else 0,
                ))
        if layer_config.has_act:
            layer_wrapper.append(register.act_dict[layer_config.act]())
        self.post_layer = paddle.nn.Sequential(*layer_wrapper)

    def forward(self, batch):
        batch = self.layer(batch)
        if isinstance(batch, paddle.Tensor):
            batch = self.post_layer(batch)
            if self.has_l2norm:
                batch = F.normalize(batch, p=2, axis=1)
        else:
            batch.x = self.post_layer(batch.x)
            if self.has_l2norm:
                batch.x = F.normalize(batch.x, p=2, axis=1)
        return batch


class GeneralMultiLayer(paddle.nn.Layer):
    def __init__(self, name, layer_config: LayerConfig, **kwargs):
        super().__init__()
        if layer_config.dim_inner:
            dim_inner = layer_config.dim_out
        else:
            dim_inner = layer_config.dim_inner

        for i in range(layer_config.num_layers):
            d_in = layer_config.dim_in if i == 0 else dim_inner
            d_out = layer_config.dim_out \
                if i == layer_config.num_layers - 1 else dim_inner
            has_act = layer_config.final_act \
                if i == layer_config.num_layers - 1 else True
            inter_layer_config = copy.deepcopy(layer_config)
            inter_layer_config.dim_in = d_in
            inter_layer_config.dim_out = d_out
            inter_layer_config.has_act = has_act
            layer = GeneralLayer(name, inter_layer_config, **kwargs)
            self.add_sublayer(f'Layer_{i}', layer)

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        return batch


@register_layer('linear')
class Linear(paddle.nn.Layer):
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = Linear_pyg(
            layer_config.dim_in,
            layer_config.dim_out,
            bias_attr=layer_config.has_bias,
        )

    def forward(self, batch):
        if isinstance(batch, paddle.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch


class BatchNorm1dNode(paddle.nn.Layer):
    def __init__(self, layer_config: LayerConfig):
        super().__init__()
        self.bn = paddle.nn.BatchNorm1D(
            layer_config.dim_in,
            epsilon=layer_config.bn_eps,
            momentum=layer_config.bn_mom,
        )

    def forward(self, batch):
        batch.x = self.bn(batch.x)
        return batch


class BatchNorm1dEdge(paddle.nn.Layer):
    def __init__(self, layer_config: LayerConfig):
        super().__init__()
        self.bn = paddle.nn.BatchNorm1D(
            layer_config.dim_in,
            epsilon=layer_config.bn_eps,
            momentum=layer_config.bn_mom,
        )

    def forward(self, batch):
        batch.edge_attr = self.bn(batch.edge_attr)
        return batch

@register.register_layer('mlp')
class MLP(paddle.nn.Layer):
    """A basic MLP model."""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        dim_inner = layer_config.dim_in if layer_config.dim_inner is None else layer_config.dim_inner
        layer_config.has_bias = True
        layers = []
        if layer_config.num_layers > 1:
            sub_layer_config = LayerConfig(
                num_layers=layer_config.num_layers - 1,
                dim_in=layer_config.dim_in,
                dim_out=dim_inner,
                dim_inner=dim_inner,
                final_act=True
            )
            layers.append(GeneralMLP(sub_layer_config))
            layer_config = replace(layer_config, dim_in=dim_inner)
            layers.append(Linear_pyg(layer_config.dim_in, layer_config.dim_out))
        else:
            layers.append(Linear_pyg(layer_config.dim_in, layer_config.dim_out))
        self.model = paddle.nn.Sequential(*layers)

    def forward(self, batch):
        if isinstance(batch, paddle.Tensor):
            return self.model(batch)
        batch.x = self.model(batch.x)
        return batch


@register.register_layer('gcnconv')
class GCNConv(paddle.nn.Layer):
    """A Graph Convolutional Network (GCN) layer."""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.GCNConv(
            layer_config.dim_in,
            layer_config.dim_out,
            bias_attr=layer_config.has_bias,
        )

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


@register.register_layer('sageconv')
class SAGEConv(paddle.nn.Layer):
    """A GraphSAGE layer."""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.SAGEConv(
            layer_config.dim_in,
            layer_config.dim_out,
            bias_attr=layer_config.has_bias,
        )

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


@register.register_layer('gatconv')
class GATConv(paddle.nn.Layer):
    """A Graph Attention Network (GAT) layer."""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.GATConv(
            layer_config.dim_in,
            layer_config.dim_out,
            bias_attr=layer_config.has_bias,
        )

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


@register.register_layer('ginconv')
class GINConv(paddle.nn.Layer):
    """A Graph Isomorphism Network (GIN) layer."""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        gin_nn = paddle.nn.Sequential(
            Linear_pyg(layer_config.dim_in, layer_config.dim_out),
            paddle.nn.ReLU(),
            Linear_pyg(layer_config.dim_out, layer_config.dim_out),
        )
        self.model = pyg.nn.GINConv(gin_nn)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


@register.register_layer('splineconv')
class SplineConv(paddle.nn.Layer):
    """A SplineCNN layer."""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.SplineConv(
            layer_config.dim_in,
            layer_config.dim_out,
            dim=1,
            kernel_size=2,
            bias_attr=layer_config.has_bias,
        )

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch


@register.register_layer('generalconv')
class GeneralConv(paddle.nn.Layer):
    """A general GNN layer."""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = GeneralConvLayer(
            layer_config.dim_in,
            layer_config.dim_out,
            bias=layer_config.has_bias,
        )

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


@register.register_layer('generaledgeconv')
class GeneralEdgeConv(paddle.nn.Layer):
    """A general GNN layer with edge feature support."""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = GeneralEdgeConvLayer(
            layer_config.dim_in,
            layer_config.dim_out,
            layer_config.edge_dim,
            bias=layer_config.has_bias,
        )

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, edge_feature=batch.edge_attr)
        return batch


@register.register_layer('generalsampleedgeconv')
class GeneralSampleEdgeConv(paddle.nn.Layer):
    """A general GNN layer that supports edge features and edge sampling."""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = GeneralEdgeConvLayer(
            layer_config.dim_in,
            layer_config.dim_out,
            layer_config.edge_dim,
            bias=layer_config.has_bias,
        )
        self.keep_edge = layer_config.keep_edge

    def forward(self, batch):
        edge_mask = paddle.rand([batch.edge_index.shape[1]]) < self.keep_edge
        edge_index = batch.edge_index[:, edge_mask]
        edge_feature = batch.edge_attr[edge_mask, :]
        batch.x = self.model(batch.x, edge_index, edge_feature=edge_feature)
        return batch