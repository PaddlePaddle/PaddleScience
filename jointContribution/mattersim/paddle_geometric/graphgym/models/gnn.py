import paddle
import paddle.nn.functional as F

import paddle_geometric.graphgym.register as register
from paddle_geometric.graphgym.config import cfg
from paddle_geometric.graphgym.init import init_weights
from paddle_geometric.graphgym.models.layer import (
    BatchNorm1dNode,
    GeneralLayer,
    GeneralMultiLayer,
    new_layer_config,
)
from paddle_geometric.graphgym.register import register_stage


def GNNLayer(dim_in: int, dim_out: int, has_act: bool = True) -> GeneralLayer:
    r"""Creates a GNN layer, given the specified input and output dimensions
    and the underlying configuration in :obj:`cfg`.
    """
    return GeneralLayer(
        cfg.gnn.layer_type,
        layer_config=new_layer_config(
            dim_in,
            dim_out,
            1,
            has_act=has_act,
            has_bias=False,
            cfg=cfg,
        ),
    )


def GNNPreMP(dim_in: int, dim_out: int, num_layers: int) -> GeneralMultiLayer:
    r"""Creates a NN layer used before message passing."""
    return GeneralMultiLayer(
        'linear',
        layer_config=new_layer_config(
            dim_in,
            dim_out,
            num_layers,
            has_act=False,
            has_bias=False,
            cfg=cfg,
        ),
    )


@register_stage('stack')
@register_stage('skipsum')
@register_stage('skipconcat')
class GNNStackStage(paddle.nn.Layer):
    r"""Stacks a number of GNN layers."""
    def __init__(self, dim_in, dim_out, num_layers):
        super().__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            if cfg.gnn.stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            else:
                d_in = dim_in if i == 0 else dim_out
            layer = GNNLayer(d_in, dim_out)
            self.add_sublayer(f'layer{i}', layer)

    def forward(self, batch):
        for i, layer in enumerate(self.children()):
            x = batch.x
            batch = layer(batch)
            if cfg.gnn.stage_type == 'skipsum':
                batch.x = x + batch.x
            elif (cfg.gnn.stage_type == 'skipconcat'
                  and i < self.num_layers - 1):
                batch.x = paddle.concat([x, batch.x], axis=1)
        if cfg.gnn.l2norm:
            batch.x = F.normalize(batch.x, p=2, axis=-1)
        return batch


class FeatureEncoder(paddle.nn.Layer):
    r"""Encodes node and edge features."""
    def __init__(self, dim_in: int):
        super().__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            NodeEncoder = register.node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(
                        cfg.gnn.dim_inner,
                        -1,
                        -1,
                        has_act=False,
                        has_bias=False,
                        cfg=cfg,
                    ))
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            EdgeEncoder = register.edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(
                        cfg.gnn.dim_inner,
                        -1,
                        -1,
                        has_act=False,
                        has_bias=False,
                        cfg=cfg,
                    ))

    def forward(self, batch):
        for module in self.sublayers():
            batch = module(batch)
        return batch


class GNN(paddle.nn.Layer):
    r"""A general Graph Neural Network (GNN) model."""
    def __init__(self, dim_in: int, dim_out: int, **kwargs):
        super().__init__()
        GNNStage = register.stage_dict[cfg.gnn.stage_type]
        GNNHead = register.head_dict[cfg.gnn.head]

        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner,
                                   cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner
        if cfg.gnn.layers_mp > 0:
            self.mp = GNNStage(dim_in=dim_in, dim_out=cfg.gnn.dim_inner,
                               num_layers=cfg.gnn.layers_mp)
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

        self.apply(init_weights)

    def forward(self, batch):
        for module in self.sublayers():
            batch = module(batch)
        return batch
