import os
import sys

import paddle
from NN.GNN.FiniteVolumeGN.EPDbackbone import EncoderProcesserDecoder
from NN.UNet.attention_unet import UNet3DWithSamplePoints

import ppsci

cur_path = os.path.split(__file__)[0]
sys.path.append(cur_path)
sys.path.append(os.path.join(cur_path, ".."))


class Model(paddle.nn.Layer):
    def __init__(
        self,
        space_dim=1,
        n_layers=5,
        n_hidden=256,
        dropout=0,
        n_head=8,
        act="gelu",
        mlp_ratio=1,
        fun_dim=1,
        out_dim=1,
        slice_num=32,
        ref=8,
        unified_pos=False,
        params=None,
    ):
        super(Model, self).__init__()
        self.unet_o_dim = n_hidden
        self.unet = UNet3DWithSamplePoints(
            in_channels=1,
            out_channels=self.unet_o_dim,
            hidden_channels=self.unet_o_dim,
            num_levels=4,
        )
        self.fvgn = EncoderProcesserDecoder(
            message_passing_num=params.message_passing_num,
            cell_input_size=params.cell_input_size,
            edge_input_size=params.edge_input_size,
            node_input_size=params.node_input_size,
            cell_output_size=params.cell_output_size,
            edge_output_size=params.edge_output_size,
            node_output_size=params.node_output_size,
            hidden_size=n_hidden,
            params=params,
        )
        self.last_layer = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=n_hidden * 2, out_features=n_hidden * 4),
            paddle.nn.GELU(),
            paddle.nn.Linear(in_features=n_hidden * 4, out_features=out_dim),
        )
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            m.weight = ppsci.utils.initializer.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, paddle.nn.Linear) and m.bias is not None:
                init_Constant = paddle.nn.initializer.Constant(value=0)
                init_Constant(m.bias)
        elif isinstance(m, (paddle.nn.LayerNorm, paddle.nn.BatchNorm1D)):
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(m.bias)
            init_Constant = paddle.nn.initializer.Constant(value=1.0)
            init_Constant(m.weight)

    def forward(
        self,
        x,
        graph_node=None,
        graph_edge=None,
        graph_cell=None,
        params=None,
        is_training=True,
    ):
        query_list = []
        for i in range(graph_cell.num_graph):
            # graph_cell.batch is a function in pgl
            # mask = i == graph_cell.batch
            # cur_query = graph_cell.query[mask]
            cur_query = graph_cell.query

            # use paddle<2.6 to support pgl so that cur_query[:, [2, 0, 1]] not work
            # cur_query = cur_query[:, [2, 0, 1]][None,]  # B, N, 3
            cur_query = paddle.stack(
                [cur_query[:, 2], cur_query[:, 0], cur_query[:, 1]], axis=-1
            )[
                None,
            ]

            cur_query = cur_query.unsqueeze(axis=2).unsqueeze(axis=2)
            query_list.append(cur_query)
        ufeatures = self.unet(graph_cell.voxel, query_list, half=False)
        graph_feature = self.fvgn(
            graph_node=graph_cell, params=params, is_training=is_training
        )
        fx = self.last_layer(paddle.concat(x=(ufeatures, graph_feature), axis=-1))
        return fx
