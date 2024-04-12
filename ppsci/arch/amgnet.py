# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import functools
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import paddle
import paddle.nn as nn
from typing_extensions import Literal

try:
    import pgl
except ModuleNotFoundError:
    pass

try:
    import pyamg
except ModuleNotFoundError:
    pass

from paddle import sparse as pd_sparse
from scipy import sparse as sci_sparse


def _knn_interpolate(
    features: paddle.Tensor, coarse_nodes: paddle.Tensor, fine_nodes: paddle.Tensor
) -> paddle.Tensor:
    coarse_nodes_input = paddle.repeat_interleave(
        coarse_nodes.unsqueeze(0), fine_nodes.shape[0], axis=0
    )  # [6684,352,2]
    fine_nodes_input = paddle.repeat_interleave(
        fine_nodes.unsqueeze(1), coarse_nodes.shape[0], axis=1
    )  # [6684,352,2]
    dist_w = 1.0 / (
        paddle.norm(x=coarse_nodes_input - fine_nodes_input, p=2, axis=-1) + 1e-9
    )  # [6684,352]
    knn_value, knn_index = paddle.topk(dist_w, k=3, largest=True)  # [6684,3],[6684,3]
    weight = knn_value.unsqueeze(-2)
    features_input = features[knn_index]
    output = paddle.bmm(weight, features_input).squeeze(-2) / paddle.sum(
        knn_value, axis=-1, keepdim=True
    )
    return output


def _get_corse_node(latent_graph: "pgl.Graph") -> paddle.Tensor:
    row = latent_graph.edge_index[0].numpy()
    col = latent_graph.edge_index[1].numpy()
    data = paddle.ones(shape=[row.size]).numpy()
    A = sci_sparse.coo_matrix((data, (row, col))).tocsr()
    splitting = pyamg.classical.split.RS(A)
    index = np.array(np.nonzero(splitting))
    b = paddle.to_tensor(index)
    b = paddle.squeeze(b)
    return b


def StAS(
    index_A: paddle.Tensor,
    value_A: paddle.Tensor,
    index_S: paddle.Tensor,
    value_S: paddle.Tensor,
    N: int,
    kN: int,
    norm_layer: nn.Layer,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph Representations.
    Ranjan, E., Sanyal, S., Talukdar, P. (2020, April).  AAAI(2020)

    Args:
        index_A (paddle.Tensor): Indices of sparse matrix A.
        value_A (paddle.Tensor): Values of sparse matrix A.
        index_S (paddle.Tensor): Indices of sparse matrix S.
        value_S (paddle.Tensor): Values of sparse matrix S.
        N (int): Dimension N.
        kN (int): Dimension kN.
        norm_layer (nn.Layer): Normalization layer.

    Returns:
        Tuple[paddle.Tensor, paddle.Tensor]: Indices and values of result matrix E.
    """
    sp_x = pd_sparse.sparse_coo_tensor(index_A, value_A)
    sp_x = pd_sparse.coalesce(sp_x)
    index_A = sp_x.indices()
    value_A = sp_x.values()

    sp_s = pd_sparse.sparse_coo_tensor(index_S, value_S)
    sp_s = pd_sparse.coalesce(sp_s)
    index_S = sp_s.indices()
    value_S = sp_s.values()

    indices_A = index_A.numpy()
    values_A = value_A.numpy()
    coo_A = sci_sparse.coo_matrix(
        (values_A, (indices_A[0], indices_A[1])), shape=(N, N)
    )

    indices_S = index_S.numpy()
    values_S = value_S.numpy()
    coo_S = sci_sparse.coo_matrix(
        (values_S, (indices_S[0], indices_S[1])), shape=(N, kN)
    )

    ans = coo_A.dot(coo_S).tocoo()
    row = paddle.to_tensor(ans.row)
    col = paddle.to_tensor(ans.col)
    index_B = paddle.stack([row, col], axis=0)
    value_B = paddle.to_tensor(ans.data)

    indices_A = index_S
    values_A = value_S
    coo_A = pd_sparse.sparse_coo_tensor(indices_A, values_A)
    out = pd_sparse.transpose(coo_A, [1, 0])
    index_St = out.indices()
    value_St = out.values()

    sp_x = pd_sparse.sparse_coo_tensor(index_B, value_B)
    sp_x = pd_sparse.coalesce(sp_x)
    index_B = sp_x.indices()
    value_B = sp_x.values()

    indices_A = index_St.numpy()
    values_A = value_St.numpy()
    coo_A = sci_sparse.coo_matrix(
        (values_A, (indices_A[0], indices_A[1])), shape=(kN, N)
    )

    indices_S = index_B.numpy()
    values_S = value_B.numpy()
    coo_S = sci_sparse.coo_matrix(
        (values_S, (indices_S[0], indices_S[1])), shape=(N, kN)
    )

    ans = coo_A.dot(coo_S).tocoo()
    row = paddle.to_tensor(ans.row)
    col = paddle.to_tensor(ans.col)
    index_E = paddle.stack([row, col], axis=0)
    value_E = paddle.to_tensor(ans.data)

    # index_E排序
    sp_x = pd_sparse.sparse_coo_tensor(index_E, value_E)
    sp_x = pd_sparse.coalesce(sp_x)
    index_E = sp_x.indices()
    value_E = sp_x.values()

    return index_E, value_E


def FillZeros(
    index_E: paddle.Tensor, value_E: paddle.Tensor, standard_index, kN: int
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    shape = [kN, kN]
    row_E = index_E[0]
    col_E = index_E[1]
    DenseMatrix_E = sci_sparse.coo_matrix(
        (paddle.ones_like(value_E), (row_E, col_E)), shape
    ).toarray()

    row_S = standard_index[0]
    col_S = standard_index[1]
    DenseMatrix_S = sci_sparse.coo_matrix(
        (paddle.ones([row_S.shape[0]]), (row_S, col_S)), shape
    ).toarray()

    diff = DenseMatrix_S - DenseMatrix_E
    rows, cols = np.nonzero(diff)
    rows = paddle.to_tensor(rows, dtype="int32")
    cols = paddle.to_tensor(cols, dtype="int32")
    index = paddle.stack([rows, cols], axis=0)
    value = paddle.zeros([index.shape[1]])
    index_E = paddle.concat([index_E, index], axis=1)
    value_E = paddle.concat([value_E, value], axis=-1)

    sp_x = pd_sparse.sparse_coo_tensor(index_E, value_E)
    sp_x = pd_sparse.coalesce(sp_x)
    index_E = sp_x.indices()
    value_E = sp_x.values()

    return index_E, value_E


def remove_self_loops(
    edge_index: paddle.Tensor, edge_attr: Optional[paddle.Tensor] = None
) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
    # remove self-loop
    mask = edge_index[0] != edge_index[1]
    mask = mask.tolist()
    edge_index = edge_index.t()
    edge_index = edge_index[mask]
    edge_index = edge_index.t()
    if edge_attr is None:
        return edge_index, None
    else:
        return edge_index, edge_attr[mask]


def faster_graph_connectivity(perm, edge_index, edge_weight, score, pos, N, norm_layer):
    """Adapted from Ranjan, E., Sanyal, S., Talukdar, P. (2020, April). Asap: Adaptive structure aware pooling
    for learning hierarchical graph representations. AAAI(2020)"""

    kN = perm.shape[0]
    perm2 = perm.reshape((-1, 1))
    mask = (edge_index[0] == perm2).sum(axis=0).astype("bool")

    S0 = edge_index[1][mask].reshape((1, -1))
    S1 = edge_index[0][mask].reshape((1, -1))
    index_S = paddle.concat([S0, S1], axis=0)
    value_S = score[mask].detach().squeeze()
    n_idx = paddle.zeros([N], dtype=paddle.int64)
    n_idx[perm] = paddle.arange(perm.shape[0])
    index_S = index_S.astype("int64")
    index_S[1] = n_idx[index_S[1]]
    subgraphnode_pos = pos[perm]
    index_A = edge_index.clone()
    if edge_weight is None:
        value_A = value_S.new_ones(edge_index[0].shape[0])
    else:
        value_A = edge_weight.clone()

    value_A = paddle.squeeze(value_A)
    model_1 = paddle.nn.Sequential(
        ("l1", paddle.nn.Linear(128, 256)),
        ("act1", paddle.nn.ReLU()),
        ("l2", paddle.nn.Linear(256, 256)),
        ("act2", paddle.nn.ReLU()),
        ("l4", paddle.nn.Linear(256, 128)),
        ("act4", paddle.nn.ReLU()),
        ("l5", paddle.nn.Linear(128, 1)),
    )
    model_2 = paddle.nn.Sequential(
        ("l1", paddle.nn.Linear(1, 64)),
        ("act1", paddle.nn.ReLU()),
        ("l2", paddle.nn.Linear(64, 128)),
        ("act2", paddle.nn.ReLU()),
        ("l4", paddle.nn.Linear(128, 128)),
    )

    val_A = model_1(value_A)
    val_A = paddle.squeeze(val_A)
    index_E, value_E = StAS(index_A, val_A, index_S, value_S, N, kN, norm_layer)
    value_E = paddle.reshape(value_E, shape=[-1, 1])
    edge_weight = model_2(value_E)

    return index_E, edge_weight, subgraphnode_pos


def norm_graph_connectivity(perm, edge_index, edge_weight, score, pos, N, norm_layer):
    """
    come from Ranjan, E., Sanyal, S., Talukdar, P. (2020, April). Asap: Adaptive
    structure aware pooling for learning hierarchical graph representations. AAAI(2020)
    """

    kN = perm.shape[0]
    perm2 = perm.reshape((-1, 1))
    mask = (edge_index[0] == perm2).sum(axis=0).astype("bool")
    S0 = edge_index[1][mask].reshape((1, -1))
    S1 = edge_index[0][mask].reshape((1, -1))

    index_S = paddle.concat([S0, S1], axis=0)
    value_S = score[mask].detach().squeeze()
    n_idx = paddle.zeros([N], dtype=paddle.int64)
    n_idx[perm] = paddle.arange(perm.shape[0])

    index_S = index_S.astype("int64")
    index_S[1] = n_idx[index_S[1]]
    subgraphnode_pos = pos[perm]
    index_A = edge_index.clone()

    if edge_weight is None:
        value_A = value_S.new_ones(edge_index[0].shape[0])
    else:
        value_A = edge_weight.clone()

    value_A = paddle.squeeze(value_A)
    eps_mask = (value_S == 0).astype(paddle.get_default_dtype())
    value_S = paddle.full_like(value_S, 1e-4) * eps_mask + (1 - eps_mask) * value_S
    attrlist = []
    standard_index, _ = StAS(
        index_A,
        paddle.ones_like(value_A[:, 0]),
        index_S,
        paddle.ones_like(value_S),
        N,
        kN,
        norm_layer,
    )
    for i in range(128):
        mask = (value_A[:, i] == 0).astype(paddle.get_default_dtype())
        val_A = paddle.full_like(mask, 1e-4) * mask + (1 - mask) * value_A[:, i]
        index_E, value_E = StAS(index_A, val_A, index_S, value_S, N, kN, norm_layer)

        if index_E.shape[1] != standard_index.shape[1]:
            index_E, value_E = FillZeros(index_E, value_E, standard_index, kN)

        index_E, value_E = remove_self_loops(edge_index=index_E, edge_attr=value_E)
        attrlist.append(value_E)
    edge_weight = paddle.stack(attrlist, axis=1)

    return index_E, edge_weight, subgraphnode_pos


class GraphNetBlock(nn.Layer):
    """Multi-Edge Interaction Network with residual connections."""

    def __init__(
        self, model_fn, output_dim, message_passing_aggregator, attention=False
    ):
        super().__init__()
        self.edge_model = model_fn(output_dim, 384)
        self.node_model = model_fn(output_dim, 256)
        self.message_passing_aggregator = message_passing_aggregator

    def _update_edge_features(self, graph):
        """Aggregates node features, and applies edge function."""
        senders = graph.edge_index[0]
        receivers = graph.edge_index[1]
        sender_features = paddle.index_select(x=graph.x, index=senders, axis=0)
        receiver_features = paddle.index_select(x=graph.x, index=receivers, axis=0)
        features = [sender_features, receiver_features, graph.edge_attr]
        features = paddle.concat(features, axis=-1)
        return self.edge_model(features)

    def unsorted_segment_operation(self, data, segment_ids, num_segments, operation):
        """Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

        Args:
            data (paddle.Tensor): A tensor whose segments are to be summed.
            segment_ids (paddle.Tensor): The segment indices tensor.
            num_segments (int): The number of segments.
            operation (str): _description_

        Returns:
            paddle.Tensor: A tensor of same data type as the data argument.
        """
        if not all([i in data.shape for i in segment_ids.shape]):
            raise ValueError("segment_ids.shape should be a prefix of data.shape")

        if not (data.shape[0] == segment_ids.shape[0]):
            raise ValueError("data.shape and segment_ids.shape should be equal")

        shape = [num_segments] + list(data.shape[1:])
        result_shape = paddle.zeros(shape)
        if operation == "sum":
            result = paddle.scatter(result_shape, segment_ids, data, overwrite=False)
        return result

    def _update_node_features(self, node_features, edge_attr, edge_index):
        """Aggregates edge features, and applies node function."""
        num_nodes = node_features.shape[0]
        features = [node_features]
        features.append(
            self.unsorted_segment_operation(
                edge_attr,
                edge_index[1],
                num_nodes,
                operation=self.message_passing_aggregator,
            )
        )
        features = paddle.concat(features, axis=-1)
        return self.node_model(features)

    def forward(self, graph):
        """Applies GraphNetBlock and returns updated MultiGraph."""
        new_edge_features = self._update_edge_features(graph)
        new_node_features = self._update_node_features(
            graph.x, graph.edge_attr, graph.edge_index
        )

        new_node_features += graph.x
        new_edge_features += graph.edge_attr
        latent_graph = pgl.Graph(
            num_nodes=new_node_features.shape[0], edges=graph.edge_index
        )
        latent_graph.x = new_node_features
        latent_graph.edge_attr = new_edge_features
        latent_graph.pos = graph.pos
        latent_graph.edge_index = graph.edge_index
        return latent_graph


class Processor(nn.Layer):
    """This class takes the nodes with the most influential feature (sum of square)
    The the chosen numbers of nodes in each ripple will establish connection(features and distances) with the most influential nodes and this connection will be learned
    Then the result is add to output latent graph of encoder and the modified latent graph will be feed into original processor

    Args:
        make_mlp (Callable): Function to make MLP.
        output_dim (int): Number of dimension of output.
        message_passing_steps (int): Message passing steps.
        message_passing_aggregator (str): Message passing aggregator.
        attention (bool, optional): Whether use attention. Defaults to False.
        use_stochastic_message_passing (bool, optional): Whether use stochastic message passing. Defaults to False.
    """

    # Each mesh can be coarsened to have no fewer points than this value
    min_nodes = 2000

    def __init__(
        self,
        make_mlp: Callable,
        output_dim: int,
        message_passing_steps: int,
        message_passing_aggregator: str,
        attention: bool = False,
        use_stochastic_message_passing: bool = False,
    ):
        super().__init__()
        self.use_stochastic_message_passing = use_stochastic_message_passing
        self.graphnet_blocks = nn.LayerList()
        self.cofe_edge_blocks = nn.LayerList()
        self.pool_blocks = nn.LayerList()
        self.latent_dim = output_dim
        self.normalization = nn.LayerNorm(128)
        for index in range(message_passing_steps):
            self.graphnet_blocks.append(
                GraphNetBlock(
                    model_fn=make_mlp,
                    output_dim=output_dim,
                    message_passing_aggregator=message_passing_aggregator,
                    attention=attention,
                )
            )

            self.pool_blocks.append(
                GraphNetBlock(
                    model_fn=make_mlp,
                    output_dim=output_dim,
                    message_passing_aggregator=message_passing_aggregator,
                    attention=attention,
                )
            )

    def forward(self, latent_graph, speed, normalized_adj_mat=None):
        x = []
        pos = []
        new = []
        for graphnet_block, pool in zip(self.graphnet_blocks, self.pool_blocks):
            if latent_graph.x.shape[0] > self.min_nodes:
                pre_matrix = graphnet_block(latent_graph)
                x.append(pre_matrix)
                cofe_graph = pool(pre_matrix)
                coarsenodes = _get_corse_node(pre_matrix)
                nodesfeatures = cofe_graph.x[coarsenodes]
                if speed == "fast":
                    subedge_index, edge_weight, subpos = faster_graph_connectivity(
                        perm=coarsenodes,
                        edge_index=cofe_graph.edge_index,
                        edge_weight=cofe_graph.edge_attr,
                        score=cofe_graph.edge_attr[:, 0],
                        pos=cofe_graph.pos,
                        N=cofe_graph.x.shape[0],
                        norm_layer=self.normalization,
                    )
                elif speed == "norm":
                    subedge_index, edge_weight, subpos = norm_graph_connectivity(
                        perm=coarsenodes,
                        edge_index=cofe_graph.edge_index,
                        edge_weight=cofe_graph.edge_attr,
                        score=cofe_graph.edge_attr[:, 0],
                        pos=cofe_graph.pos,
                        N=cofe_graph.x.shape[0],
                        norm_layer=self.normalization,
                    )
                else:
                    raise ValueError(
                        f"Argument 'speed' should be 'sum' or 'fast', bot got {speed}."
                    )
                edge_weight = self.normalization(edge_weight)
                pos.append(subpos)
                latent_graph = pgl.Graph(
                    num_nodes=nodesfeatures.shape[0], edges=subedge_index
                )
                latent_graph.x = nodesfeatures
                latent_graph.edge_attr = edge_weight
                latent_graph.pos = subpos
                latent_graph.edge_index = subedge_index
            else:
                latent_graph = graphnet_block(latent_graph)
                new.append(latent_graph)
        if len(new):
            x.append(new[-1])
        return x, pos


class FullyConnectedLayer(nn.Layer):
    def __init__(self, input_dim: int, hidden_size: Tuple[int, ...]):
        super(FullyConnectedLayer, self).__init__()
        num_layers = len(hidden_size)
        self._layers_ordered_dict = {}
        self.in_dim = input_dim
        for index, output_dim in enumerate(hidden_size):
            self._layers_ordered_dict["linear_" + str(index)] = nn.Linear(
                self.in_dim, output_dim
            )
            if index < (num_layers - 1):
                self._layers_ordered_dict["relu_" + str(index)] = nn.ReLU()
            self.in_dim = output_dim

        self.layers = nn.LayerDict(self._layers_ordered_dict)

    def forward(self, input):
        for key in self.layers:
            layer = self.layers[key]
            output = layer(input)
            input = output
        return input


class Encoder(nn.Layer):
    """Encodes node and edge features into latent features."""

    def __init__(self, input_dim, make_mlp, latent_dim):
        super(Encoder, self).__init__()
        self._make_mlp = make_mlp
        self._latent_dim = latent_dim
        self.node_model = self._make_mlp(latent_dim, input_dim=input_dim)
        self.mesh_edge_model = self._make_mlp(latent_dim, input_dim=1)

    def forward(self, graph):
        node_latents = self.node_model(graph.x)
        edge_latent = self.mesh_edge_model(graph.edge_attr)

        graph.x = node_latents
        graph.edge_attr = edge_latent
        return graph


class Decoder(nn.Layer):
    """Decodes node features from graph.
    Encodes node and edge features into latent features.
    """

    def __init__(self, make_mlp, output_dim):
        super(Decoder, self).__init__()
        self.model = make_mlp(output_dim, 128)

    def forward(self, node_features):
        return self.model(node_features)


class AMGNet(nn.Layer):
    """A Multi-scale Graph neural Network model
    based on Encoder-Process-Decoder structure for flow field prediction.

    https://doi.org/10.1080/09540091.2022.2131737

    Code reference: https://github.com/baoshiaijhin/amgnet

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("input", ).
        output_keys (Tuple[str, ...]): Name of output keys, such as ("pred", ).
        input_dim (int): Number of input dimension.
        output_dim (int): Number of output dimension.
        latent_dim (int): Number of hidden(feature) dimension.
        num_layers (int): Number of layer(s).
        message_passing_aggregator (Literal["sum"]): Message aggregator method in graph.
            Only "sum" available now.
        message_passing_steps (int): Message passing steps in graph.
        speed (str): Whether use vanilla method or fast method for graph_connectivity
            computation.

    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.AMGNet(
        ...     ("input", ), ("pred", ), 5, 3, 64, 2, "sum", 6, "norm",
        ... )
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        input_dim: int,
        output_dim: int,
        latent_dim: int,
        num_layers: int,
        message_passing_aggregator: Literal["sum"],
        message_passing_steps: int,
        speed: Literal["norm", "fast"],
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self._latent_dim = latent_dim
        self.speed = speed
        self._output_dim = output_dim
        self._num_layers = num_layers

        self.encoder = Encoder(input_dim, self._make_mlp, latent_dim=self._latent_dim)
        self.processor = Processor(
            make_mlp=self._make_mlp,
            output_dim=self._latent_dim,
            message_passing_steps=message_passing_steps,
            message_passing_aggregator=message_passing_aggregator,
            use_stochastic_message_passing=False,
        )
        self.post_processor = self._make_mlp(self._latent_dim, 128)
        self.decoder = Decoder(
            make_mlp=functools.partial(self._make_mlp, layer_norm=False),
            output_dim=self._output_dim,
        )

    def forward(self, x: Dict[str, "pgl.Graph"]) -> Dict[str, paddle.Tensor]:
        graphs = x[self.input_keys[0]]
        latent_graph = self.encoder(graphs)
        x, p = self.processor(latent_graph, speed=self.speed)
        node_features = self._spa_compute(x, p)
        pred_field = self.decoder(node_features)
        return {self.output_keys[0]: pred_field}

    def _make_mlp(self, output_dim: int, input_dim: int = 5, layer_norm: bool = True):
        widths = (self._latent_dim,) * self._num_layers + (output_dim,)
        network = FullyConnectedLayer(input_dim, widths)
        if layer_norm:
            network = nn.Sequential(network, nn.LayerNorm(normalized_shape=widths[-1]))
        return network

    def _spa_compute(self, x: List["pgl.Graph"], p):
        j = len(x) - 1
        node_features = x[j].x

        for k in range(1, j + 1):
            pos = p[-k]
            fine_nodes = x[-(k + 1)].pos
            feature = _knn_interpolate(node_features, pos, fine_nodes)
            node_features = x[-(k + 1)].x + feature
            node_features = self.post_processor(node_features)

        return node_features
