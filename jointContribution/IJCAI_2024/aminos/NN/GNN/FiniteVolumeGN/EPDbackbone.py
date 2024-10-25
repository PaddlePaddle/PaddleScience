import paddle
import pgl
from dataset.Load_mesh import CustomGraphData
from utils.normalization import Normalizer
from utils.paddle_aux import scatter_paddle
from utils.utilities import copy_geometric_data
from utils.utilities import decompose_and_trans_node_attr_to_cell_attr_graph

from .blocks import EdgeBlock
from .blocks import NodeBlock


def build_mlp(
    in_size, hidden_size, out_size, drop_out=True, lay_norm=True, dropout_prob=0.2
):
    if drop_out:
        module = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=in_size, out_features=hidden_size),
            paddle.nn.Dropout(p=dropout_prob),
            paddle.nn.GELU(),
            paddle.nn.Linear(in_features=hidden_size, out_features=hidden_size),
            paddle.nn.Dropout(p=dropout_prob),
            paddle.nn.GELU(),
            paddle.nn.Linear(in_features=hidden_size, out_features=out_size),
        )
    else:
        module = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=in_size, out_features=hidden_size),
            paddle.nn.GELU(),
            paddle.nn.Linear(in_features=hidden_size, out_features=hidden_size),
            paddle.nn.GELU(),
            paddle.nn.Linear(in_features=hidden_size, out_features=out_size),
        )
    if lay_norm:
        return paddle.nn.Sequential(
            module, paddle.nn.LayerNorm(normalized_shape=out_size)
        )
    return module


def build_mlp_test(
    in_size,
    hidden_size,
    out_size,
    drop_out=False,
    lay_norm=True,
    dropout_prob=0.2,
    specify_hidden_layer_num=2,
):
    layers = []
    layers.append(paddle.nn.Linear(in_features=in_size, out_features=hidden_size))
    if drop_out:
        layers.append(paddle.nn.Dropout(p=dropout_prob))
    layers.append(paddle.nn.GELU())
    for i in range(specify_hidden_layer_num - 1):
        layers.append(
            paddle.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        )
        if drop_out:
            layers.append(paddle.nn.Dropout(p=dropout_prob))
        layers.append(paddle.nn.GELU())
    layers.append(paddle.nn.Linear(in_features=hidden_size, out_features=out_size))
    if lay_norm:
        layers.append(paddle.nn.LayerNorm(normalized_shape=out_size))
    return paddle.nn.Sequential(*layers)


class Encoder(paddle.nn.Layer):
    def __init__(
        self,
        node_input_size=128,
        edge_input_size=128,
        cell_input_size=128,
        hidden_size=128,
        attention=False,
    ):
        super(Encoder, self).__init__()
        self.eb_encoder = build_mlp(
            edge_input_size, hidden_size, int(hidden_size), drop_out=False
        )
        self.nb_encoder = build_mlp(
            node_input_size, hidden_size, int(hidden_size), drop_out=False
        )
        self.attention = attention
        self.scale = paddle.sqrt(
            x=paddle.to_tensor(data=hidden_size, dtype=paddle.get_default_dtype())
        )

    def forward(self, graph_node, graph_cell):
        (
            node_attr,
            edge_index,
            edge_attr,
            face,
            _,
            _,
        ) = decompose_and_trans_node_attr_to_cell_attr_graph(
            graph_node, has_changed_node_attr_to_cell_attr=False
        )
        node_ = self.nb_encoder(node_attr)
        edge_ = self.eb_encoder(edge_attr)
        ret = CustomGraphData(
            x=node_,
            edge_attr=edge_,
            edge_index=edge_index,
            face=face,
            num_graphs=graph_node.num_graph,
            batch=graph_node.batch,
        )
        ret.keys = ["x", "num_graphs", "edge_index", "batch", "edge_attr"]
        return ret, edge_, node_


class GraphSCA3D(paddle.nn.Layer):
    def __init__(self, channel, reduction=2):
        super().__init__()
        self.channel_excitation = paddle.nn.Sequential(
            paddle.nn.Linear(
                in_features=channel, out_features=int(channel // reduction)
            ),
            paddle.nn.ReLU(),
            paddle.nn.Linear(
                in_features=int(channel // reduction), out_features=channel
            ),
        )
        self.spatial_se = pgl.nn.GCNConv(input_size=channel, output_size=1)

    def forward(self, x, batch, edge_index):
        BN, C = tuple(x.shape)
        chn_se = scatter_paddle(x, index=batch, dim=0, reduce="mean").view(-1, C)
        chn_se = paddle.nn.functional.sigmoid(x=self.channel_excitation(chn_se))
        chn_se = x * chn_se[batch]
        spa_se = paddle.nn.functional.sigmoid(x=self.spatial_se(x, edge_index))
        spa_se = x * spa_se
        net_out = spa_se + x + chn_se
        return net_out


class GnBlock(paddle.nn.Layer):
    def __init__(self, hidden_size=128, drop_out=False, attention=True, MultiHead=1):
        super(GnBlock, self).__init__()
        eb_input_dim = int(3 * hidden_size)
        nb_input_dim = int(hidden_size + hidden_size // 2.0)
        self.nb_module = NodeBlock(
            hidden_size,
            hidden_size,
            attention=attention,
            MultiHead=MultiHead,
            custom_func=build_mlp(
                nb_input_dim, hidden_size, int(hidden_size), drop_out=False
            ),
        )
        self.eb_module = EdgeBlock(
            input_size=hidden_size,
            custom_func=build_mlp(
                eb_input_dim, hidden_size, int(hidden_size), drop_out=False
            ),
        )

    def forward(self, graph_node, graph_cell=None):
        graph_node_last = copy_geometric_data(
            graph_node, has_changed_node_attr_to_cell_attr=True
        )
        graph_node = self.eb_module(graph_node, graph_cell=None)
        graph_node = self.nb_module(graph_node, graph_cell=None)
        x = graph_node.x + graph_node_last.x
        edge_attr = graph_node.edge_attr + graph_node_last.edge_attr
        ret = CustomGraphData(
            x=x,
            edge_attr=edge_attr,
            edge_index=graph_node.edge_index,
            face=graph_node.face,
            num_graphs=graph_node.num_graph,
            batch=graph_node.batch,
        )
        return ret


class Decoder(paddle.nn.Layer):
    def __init__(
        self,
        edge_hidden_size=128,
        cell_hidden_size=128,
        edge_output_size=3,
        cell_output_size=2,
        cell_input_size=2,
        node_output_size=2,
        attention=False,
    ):
        super(Decoder, self).__init__()
        self.node_decode_module = build_mlp_test(
            cell_hidden_size,
            cell_hidden_size,
            node_output_size,
            drop_out=False,
            lay_norm=False,
            specify_hidden_layer_num=2,
        )

    def forward(self, trans_feature=None, latent_graph_node=None):
        node_attr, _, _, _, _, _ = decompose_and_trans_node_attr_to_cell_attr_graph(
            latent_graph_node, has_changed_node_attr_to_cell_attr=True
        )
        node_decode_attr = self.node_decode_module(node_attr)
        return node_decode_attr


class EncoderProcesserDecoder(paddle.nn.Layer):
    def __init__(
        self,
        message_passing_num,
        cell_input_size,
        edge_input_size,
        node_input_size,
        cell_output_size,
        edge_output_size,
        node_output_size,
        drop_out=False,
        hidden_size=128,
        attention=False,
        params=None,
        MultiHead=1,
    ):
        super(EncoderProcesserDecoder, self).__init__()
        self.encoder = Encoder(
            node_input_size=node_input_size,
            edge_input_size=edge_input_size,
            cell_input_size=cell_input_size,
            hidden_size=hidden_size,
            attention=attention,
        )
        try:
            satistic_times = params.dataset_size // params.batch_size
        except Exception:
            satistic_times = 500
        self.node_norm = Normalizer(node_input_size, satistic_times)
        self.edge_norm = Normalizer(edge_input_size, satistic_times)
        GN_block_list = []
        for _ in range(message_passing_num):
            GN_block_list.append(
                GnBlock(
                    hidden_size=hidden_size,
                    drop_out=drop_out,
                    attention=attention,
                    MultiHead=MultiHead,
                )
            )
        self.GN_block_list = paddle.nn.LayerList(sublayers=GN_block_list)

    def forward(
        self,
        graph_node=None,
        graph_edge=None,
        graph_cell=None,
        params=None,
        is_training=True,
    ):
        graph_node.x = self.node_norm(graph_node.x)
        graph_node.edge_attr = self.edge_norm(graph_node.edge_attr)
        latent_graph_node, _, _ = self.encoder(graph_node, graph_cell=graph_cell)
        for model in self.GN_block_list:
            latent_graph_node = model(latent_graph_node, graph_cell=graph_cell)
        return latent_graph_node.x
