import paddle
import pgl
import utils.paddle_aux as paddle_aux
from dataset.Load_mesh import CustomGraphData
from utils.utilities import calc_cell_centered_with_node_attr
from utils.utilities import decompose_and_trans_node_attr_to_cell_attr_graph


class GraphSCA3D(paddle.nn.Layer):
    def __init__(self, channel, reduction=16):
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
        chn_se = paddle_aux.scatter_paddle(x, index=batch, dim=0, reduce="mean").view(
            -1, C
        )
        chn_se = paddle.nn.functional.sigmoid(x=self.channel_excitation(chn_se))
        chn_se = x * chn_se[batch]
        spa_se = paddle.nn.functional.sigmoid(x=self.spatial_se(x, edge_index))
        spa_se = x * spa_se
        net_out = spa_se + x + chn_se
        return net_out


class GraphCA3D(paddle.nn.Layer):
    def __init__(self, channel, reduction=16):
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

    def forward(self, x, batch):
        BN, C = tuple(x.shape)
        chn_se = paddle_aux.scatter_paddle(x, index=batch, dim=0, reduce="mean").view(
            -1, C
        )
        chn_se = paddle.nn.functional.sigmoid(x=self.channel_excitation(chn_se))
        chn_se = x * chn_se[batch]
        net_out = x + chn_se
        return net_out


class NodeBlock(paddle.nn.Layer):
    def __init__(
        self, input_size, attention_size, attention=True, MultiHead=1, custom_func=None
    ):
        super(NodeBlock, self).__init__()
        self.net = custom_func

    def forward(self, graph_node, graph_cell=None):
        (
            node_attr,
            edge_index,
            edge_attr,
            face,
            _,
            _,
        ) = decompose_and_trans_node_attr_to_cell_attr_graph(
            graph_node, has_changed_node_attr_to_cell_attr=True
        )
        """ cell-based two step message passing algorithm """
        if graph_cell is not None:
            senders_cell_idx, receivers_cell_idx = graph_cell.edge_index
            twoway_edge_attr = paddle.concat(
                x=paddle.chunk(x=edge_attr, chunks=2, axis=-1), axis=0
            )
            twoway_cell_connections_indegree = paddle.concat(
                x=[senders_cell_idx, receivers_cell_idx], axis=0
            )
            twoway_cell_connections_outdegree = paddle.concat(
                x=[receivers_cell_idx, senders_cell_idx], axis=0
            )

            cell_agg_received_edges = paddle_aux.scatter_paddle(
                twoway_edge_attr, twoway_cell_connections_indegree, dim=0, reduce="add"
            )
            cell_agg_neighbour_cell = paddle_aux.scatter_paddle(
                cell_agg_received_edges[twoway_cell_connections_indegree],
                twoway_cell_connections_outdegree,
                dim=0,
                reduce="add",
            )
            cells_node = graph_node.face[0]
            cells_index = graph_cell.face[0]
            cell_to_node = cell_agg_neighbour_cell[cells_index]
            node_agg_received_edges = paddle_aux.scatter_paddle(
                cell_to_node, index=cells_node, dim=0, reduce="mean"
            )
            x = self.net(paddle.concat(x=(node_agg_received_edges, node_attr), axis=1))
        else:
            """node-based two step message passing algorithm"""
            senders_node_idx, receivers_node_idx = edge_index
            twoway_node_connections_indegree = paddle.concat(
                x=[senders_node_idx, receivers_node_idx], axis=0
            )
            twoway_node_connections_outdegree = paddle.concat(
                x=[receivers_node_idx, senders_node_idx], axis=0
            )
            twoway_edge_attr = paddle.concat(
                x=paddle.chunk(x=edge_attr, chunks=2, axis=-1), axis=0
            )

            node_agg_received_edges = paddle_aux.scatter_paddle(
                twoway_edge_attr,
                twoway_node_connections_indegree,
                dim=0,
                out=paddle.zeros(shape=(node_attr.shape[0], twoway_edge_attr.shape[1])),
                reduce="add",
            )
            node_avg_neighbour_node = paddle_aux.scatter_paddle(
                node_agg_received_edges[twoway_node_connections_outdegree],
                twoway_node_connections_indegree,
                dim=0,
                out=paddle.zeros(shape=(node_attr.shape[0], twoway_edge_attr.shape[1])),
                reduce="mean",
            )
            x = self.net(paddle.concat(x=(node_avg_neighbour_node, node_attr), axis=1))
        ret = CustomGraphData(
            x=x,
            edge_attr=edge_attr,
            edge_index=edge_index,
            face=face,
            num_graphs=graph_node.num_graph,
            batch=graph_node.batch,
        )
        return ret


class EdgeBlock(paddle.nn.Layer):
    def __init__(self, input_size=None, custom_func=None):
        super(EdgeBlock, self).__init__()
        self.net = custom_func

    def forward(self, graph_node, graph_cell=None):
        (
            node_attr,
            edge_index,
            edge_attr,
            face,
            _,
            _,
        ) = decompose_and_trans_node_attr_to_cell_attr_graph(
            graph_node, has_changed_node_attr_to_cell_attr=True
        )
        edges_to_collect = []
        """ >>> node to cell and concancentendate to edge >>> """
        if graph_cell is not None:
            cells_node = graph_node.face
            cells_index = graph_cell.face
            cell_attr = calc_cell_centered_with_node_attr(
                node_attr=node_attr,
                cells_node=cells_node,
                cells_index=cells_index,
                reduce="sum",
                map=True,
            )
            senders_cell_idx, receivers_cell_idx = graph_cell.edge_index
            mask = paddle.logical_not(
                x=senders_cell_idx == receivers_cell_idx
            ).unsqueeze(axis=1)
            senders_attr = cell_attr[senders_cell_idx]
            receivers_attr = cell_attr[receivers_cell_idx]
            edges_to_collect.append(senders_attr)
            edges_to_collect.append(receivers_attr * mask.astype(dtype="int64"))
            edges_to_collect.append(edge_attr)
            """ <<< node to cell and concancentendate to edge <<< """
            collected_edges = paddle.concat(x=edges_to_collect, axis=1)
            edge_attr_ = self.net(collected_edges)
        else:
            """>>> only node concancentendate to edge >>>"""
            senders_node_idx, receivers_node_idx = edge_index
            twoway_node_connections_indegree = paddle.concat(
                x=[senders_node_idx, receivers_node_idx], axis=0
            )
            twoway_node_connections_outdegree = paddle.concat(
                x=[receivers_node_idx, senders_node_idx], axis=0
            )

            node_avg_neighbour_node = paddle_aux.scatter_paddle(
                node_attr[twoway_node_connections_outdegree],
                twoway_node_connections_indegree,
                dim=0,
                out=paddle.zeros(shape=(node_attr.shape[0], node_attr.shape[1])),
                reduce="add",
            )
            senders_attr = node_avg_neighbour_node[senders_node_idx]
            receivers_attr = node_avg_neighbour_node[receivers_node_idx]
            edges_to_collect.append(senders_attr)
            edges_to_collect.append(receivers_attr)
            edges_to_collect.append(edge_attr)
            """ >>>> only node concancentendate to edge >>> """
            collected_edges = paddle.concat(x=edges_to_collect, axis=1)
            edge_attr_ = self.net(collected_edges)
        ret = CustomGraphData(
            x=node_attr,
            edge_attr=edge_attr_,
            edge_index=edge_index,
            face=face,
            num_graphs=graph_node.num_graph,
            batch=graph_node.batch,
        )
        return ret
