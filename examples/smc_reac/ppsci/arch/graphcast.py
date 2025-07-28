# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING
from typing import Dict
from typing import Tuple

import paddle
import paddle.nn as nn

from ppsci.arch import base

if TYPE_CHECKING:
    import ppsci.data.dataset.atmospheric_dataset as atmospheric_dataset


class ResidualConnection(nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, inputs):
        return inputs + self.fn(inputs)


class GraphCastMLP(nn.Layer):
    def __init__(
        self, in_features, out_features, latent_features=None, layer_norm=True
    ):
        super().__init__()

        if latent_features is None:
            latent_features = out_features

        self.mlp = nn.Sequential(
            nn.Linear(in_features, latent_features, bias_attr=True),
            nn.Silu(),
            nn.Linear(latent_features, out_features, bias_attr=True),
        )
        self.layer_norm = layer_norm
        if layer_norm:
            self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, feat):
        if self.layer_norm:
            out = self.layer_norm(self.mlp(feat))
        else:
            out = self.mlp(feat)
        return out


class GraphCastGNN(nn.Layer):
    def __init__(
        self,
        grid_node_num: int,
        grid_node_emb_dim: int,
        mesh_node_num: int,
        mesh_node_emb_dim: int,
        mesh_edge_emb_dim: int,
        grid2mesh_edge_emb_dim: int,
        mesh2grid_edge_emb_dim: int,
        src_type: str = "mesh",
        dst_type: str = "mesh",
    ):
        super().__init__()

        self.src = src_type
        self.dst = dst_type
        self.grid_node_num = grid_node_num
        self.mesh_node_num = mesh_node_num
        self.edge_in_dim = grid_node_emb_dim + mesh_node_emb_dim

        if src_type == "mesh" and dst_type == "mesh":
            self.edge_in_dim += mesh_edge_emb_dim
            self.edge_out_dim = mesh_edge_emb_dim
            self.node_in_dim = mesh_node_emb_dim + mesh_edge_emb_dim
            self.node_out_dim = mesh_node_emb_dim
        elif src_type == "grid" and dst_type == "mesh":
            self.edge_in_dim += grid2mesh_edge_emb_dim
            self.edge_out_dim = grid2mesh_edge_emb_dim
            self.node_in_dim = mesh_node_emb_dim + grid2mesh_edge_emb_dim
            self.node_out_dim = mesh_node_emb_dim
        elif src_type == "mesh" and dst_type == "grid":
            self.edge_in_dim += mesh2grid_edge_emb_dim
            self.edge_out_dim = mesh2grid_edge_emb_dim
            self.node_in_dim = grid_node_emb_dim + mesh2grid_edge_emb_dim
            self.node_out_dim = grid_node_emb_dim
        else:
            raise ValueError

        self.edge_layer = GraphCastMLP(self.edge_in_dim, self.edge_out_dim)
        self.node_layer = GraphCastMLP(self.node_in_dim, self.node_out_dim)

    def forward(self, graph: "atmospheric_dataset.GraphGridMesh"):
        if self.src == "mesh" and self.dst == "mesh":
            edge_feats = graph.mesh_edge_feat
            src_node_feats = graph.mesh_node_feat
            dst_node_feats = graph.mesh_node_feat
            src_idx = graph.mesh2mesh_src_index
            dst_idx = graph.mesh2mesh_dst_index
            dst_node_num = self.mesh_node_num
        elif self.src == "grid" and self.dst == "mesh":
            edge_feats = graph.grid2mesh_edge_feat
            src_node_feats = graph.grid_node_feat
            dst_node_feats = graph.mesh_node_feat
            src_idx = graph.grid2mesh_src_index
            dst_idx = graph.grid2mesh_dst_index
            dst_node_num = self.mesh_node_num
        elif self.src == "mesh" and self.dst == "grid":
            edge_feats = graph.mesh2grid_edge_feat
            src_node_feats = graph.mesh_node_feat
            dst_node_feats = graph.grid_node_feat
            src_idx = graph.mesh2grid_src_index
            dst_idx = graph.mesh2grid_dst_index
            dst_node_num = self.grid_node_num

        # update edge features
        edge_feats_concat = paddle.concat(
            [
                edge_feats,
                paddle.gather(src_node_feats, src_idx),
                paddle.gather(dst_node_feats, dst_idx),
            ],
            axis=-1,
        )
        edge_feats_out = self.edge_layer(edge_feats_concat)

        _, batch_dim, _ = edge_feats_out.shape

        # update node features
        edge_feats_scatter = paddle.zeros([dst_node_num, batch_dim, self.edge_out_dim])
        node_feats_concat = paddle.concat(
            [
                dst_node_feats,
                paddle.scatter(
                    edge_feats_scatter, dst_idx, edge_feats_out, overwrite=False
                ),
            ],
            axis=-1,
        )
        node_feats_out = self.node_layer(node_feats_concat)

        if self.src == "mesh" and self.dst == "mesh":
            graph.mesh_edge_feat += edge_feats_out
            graph.mesh_node_feat += node_feats_out
        elif self.src == "grid" and self.dst == "mesh":
            graph.grid2mesh_edge_feat += edge_feats_out
            graph.mesh_node_feat += node_feats_out
        elif self.src == "mesh" and self.dst == "grid":
            graph.mesh2grid_edge_feat += edge_feats_out
            graph.grid_node_feat += node_feats_out

        return graph


class GraphCastEmbedding(nn.Layer):
    def __init__(
        self,
        grid_node_dim: int,
        grid_node_emb_dim: int,
        mesh_node_dim: int,
        mesh_node_emb_dim: int,
        mesh_edge_dim: int,
        mesh_edge_emb_dim: int,
        grid2mesh_edge_dim: int,
        grid2mesh_edge_emb_dim: int,
        mesh2grid_edge_dim: int,
        mesh2grid_edge_emb_dim: int,
    ):
        super().__init__()

        self.grid_node_embedding = GraphCastMLP(grid_node_dim, grid_node_emb_dim)
        self.mesh_node_embedding = GraphCastMLP(mesh_node_dim, mesh_node_emb_dim)
        self.mesh_edge_embedding = GraphCastMLP(mesh_edge_dim, mesh_edge_emb_dim)
        self.grid2mesh_edge_embedding = GraphCastMLP(
            grid2mesh_edge_dim, grid2mesh_edge_emb_dim
        )
        self.mesh2grid_edge_embedding = GraphCastMLP(
            mesh2grid_edge_dim, mesh2grid_edge_emb_dim
        )

    def forward(self, graph: "atmospheric_dataset.GraphGridMesh"):
        grid_node_emb = self.grid_node_embedding(graph.grid_node_feat)
        mesh_node_emb = self.mesh_node_embedding(graph.mesh_node_feat)
        mesh_edge_emb = self.mesh_edge_embedding(graph.mesh_edge_feat)
        grid2mesh_edge_emb = self.grid2mesh_edge_embedding(graph.grid2mesh_edge_feat)
        mesh2grid_edge_emb = self.mesh2grid_edge_embedding(graph.mesh2grid_edge_feat)

        graph.grid_node_feat = grid_node_emb
        graph.mesh_node_feat = mesh_node_emb
        graph.mesh_edge_feat = mesh_edge_emb
        graph.grid2mesh_edge_feat = grid2mesh_edge_emb
        graph.mesh2grid_edge_feat = mesh2grid_edge_emb

        return graph


class GraphCastGrid2Mesh(nn.Layer):
    def __init__(
        self,
        grid_node_num: int,
        grid_node_emb_dim: int,
        mesh_node_num: int,
        mesh_node_emb_dim: int,
        mesh_edge_emb_dim: int,
        grid2mesh_edge_emb_dim: int,
        mesh2grid_edge_emb_dim: int,
    ):
        super().__init__()
        self.grid2mesh_gnn = GraphCastGNN(
            grid_node_num=grid_node_num,
            grid_node_emb_dim=grid_node_emb_dim,
            mesh_node_num=mesh_node_num,
            mesh_node_emb_dim=mesh_node_emb_dim,
            mesh_edge_emb_dim=mesh_edge_emb_dim,
            grid2mesh_edge_emb_dim=grid2mesh_edge_emb_dim,
            mesh2grid_edge_emb_dim=mesh2grid_edge_emb_dim,
            src_type="grid",
            dst_type="mesh",
        )
        self.grid_node_layer = ResidualConnection(
            GraphCastMLP(grid_node_emb_dim, grid_node_emb_dim)
        )

    def forward(self, graph: "atmospheric_dataset.GraphGridMesh"):
        graph = self.grid2mesh_gnn(graph)
        graph.grid_node_feat = self.grid_node_layer(graph.grid_node_feat)
        return graph


class GraphCastMesh2Grid(nn.Layer):
    def __init__(
        self,
        grid_node_num: int,
        grid_node_emb_dim: int,
        mesh_node_num: int,
        mesh_node_emb_dim: int,
        mesh_edge_emb_dim: int,
        grid2mesh_edge_emb_dim: int,
        mesh2grid_edge_emb_dim: int,
    ):
        super().__init__()
        self.mesh2grid_gnn = GraphCastGNN(
            grid_node_num=grid_node_num,
            grid_node_emb_dim=grid_node_emb_dim,
            mesh_node_num=mesh_node_num,
            mesh_node_emb_dim=mesh_node_emb_dim,
            mesh_edge_emb_dim=mesh_edge_emb_dim,
            grid2mesh_edge_emb_dim=grid2mesh_edge_emb_dim,
            mesh2grid_edge_emb_dim=mesh2grid_edge_emb_dim,
            src_type="mesh",
            dst_type="grid",
        )
        self.mesh_node_layer = ResidualConnection(
            GraphCastMLP(mesh_node_emb_dim, mesh_node_emb_dim)
        )

    def forward(self, graph: "atmospheric_dataset.GraphGridMesh"):
        graph = self.mesh2grid_gnn(graph)
        graph.mesh_node_feat = self.mesh_node_layer(graph.mesh_node_feat)
        return graph


class GraphCastEncoder(nn.Layer):
    def __init__(
        self,
        grid_node_num: int,
        grid_node_dim: int,
        grid_node_emb_dim: int,
        mesh_node_num: int,
        mesh_node_dim: int,
        mesh_node_emb_dim: int,
        mesh_edge_dim: int,
        mesh_edge_emb_dim: int,
        grid2mesh_edge_dim: int,
        grid2mesh_edge_emb_dim: int,
        mesh2grid_edge_dim: int,
        mesh2grid_edge_emb_dim: int,
    ):
        super().__init__()
        self.embedding = GraphCastEmbedding(
            grid_node_dim=grid_node_dim,
            grid_node_emb_dim=grid_node_emb_dim,
            mesh_node_dim=mesh_node_dim,
            mesh_node_emb_dim=mesh_node_emb_dim,
            mesh_edge_dim=mesh_edge_dim,
            mesh_edge_emb_dim=mesh_edge_emb_dim,
            grid2mesh_edge_dim=grid2mesh_edge_dim,
            grid2mesh_edge_emb_dim=grid2mesh_edge_emb_dim,
            mesh2grid_edge_dim=mesh2grid_edge_dim,
            mesh2grid_edge_emb_dim=mesh2grid_edge_emb_dim,
        )
        self.grid2mesh_gnn = GraphCastGrid2Mesh(
            grid_node_num=grid_node_num,
            grid_node_emb_dim=grid_node_emb_dim,
            mesh_node_num=mesh_node_num,
            mesh_node_emb_dim=mesh_node_emb_dim,
            mesh_edge_emb_dim=mesh_edge_emb_dim,
            grid2mesh_edge_emb_dim=grid2mesh_edge_emb_dim,
            mesh2grid_edge_emb_dim=mesh2grid_edge_emb_dim,
        )

    def forward(self, graph: "atmospheric_dataset.GraphGridMesh"):
        graph = self.embedding(graph)
        graph = self.grid2mesh_gnn(graph)
        return graph


class GraphCastDecoder(nn.Layer):
    def __init__(
        self,
        grid_node_num: int,
        grid_node_emb_dim: int,
        mesh_node_num: int,
        mesh_node_emb_dim: int,
        mesh_edge_emb_dim: int,
        grid2mesh_edge_emb_dim: int,
        mesh2grid_edge_emb_dim: int,
        node_output_dim: int,
    ):
        super().__init__()
        self.mesh2grid_gnn = GraphCastMesh2Grid(
            grid_node_num=grid_node_num,
            grid_node_emb_dim=grid_node_emb_dim,
            mesh_node_num=mesh_node_num,
            mesh_node_emb_dim=mesh_node_emb_dim,
            mesh_edge_emb_dim=mesh_edge_emb_dim,
            grid2mesh_edge_emb_dim=grid2mesh_edge_emb_dim,
            mesh2grid_edge_emb_dim=mesh2grid_edge_emb_dim,
        )
        self.grid_node_layer = GraphCastMLP(
            grid_node_emb_dim,
            node_output_dim,
            latent_features=grid_node_emb_dim,
            layer_norm=False,
        )

    def forward(self, graph: "atmospheric_dataset.GraphGridMesh"):
        graph = self.mesh2grid_gnn(graph)
        graph.grid_node_feat = self.grid_node_layer(graph.grid_node_feat)
        return graph


class GraphCastProcessor(nn.Layer):
    def __init__(
        self,
        grid_node_num: int,
        grid_node_emb_dim: int,
        mesh_node_num: int,
        mesh_node_emb_dim: int,
        mesh_edge_emb_dim: int,
        grid2mesh_edge_emb_dim: int,
        mesh2grid_edge_emb_dim: int,
        gnn_msg_steps: int,
    ):
        super().__init__()

        self.processor = nn.Sequential()
        for idx in range(gnn_msg_steps):
            self.processor.add_sublayer(
                f"{idx}",
                GraphCastGNN(
                    grid_node_num=grid_node_num,
                    grid_node_emb_dim=grid_node_emb_dim,
                    mesh_node_num=mesh_node_num,
                    mesh_node_emb_dim=mesh_node_emb_dim,
                    mesh_edge_emb_dim=mesh_edge_emb_dim,
                    grid2mesh_edge_emb_dim=grid2mesh_edge_emb_dim,
                    mesh2grid_edge_emb_dim=mesh2grid_edge_emb_dim,
                    src_type="mesh",
                    dst_type="mesh",
                ),
            )

    def forward(self, graph: "atmospheric_dataset.GraphGridMesh"):
        graph = self.processor(graph)
        return graph


class GraphCastNet(base.Arch):
    """GraphCast Network

    Args:
        input_keys (Tuple[str, ...]): Name of input keys.
        output_keys (Tuple[str, ...]): Name of output keys.
        grid_node_num (int): Number of grid nodes.
        grid_node_dim (int): Dimension of grid nodes.
        grid_node_emb_dim (int): Dimension of emdding grid nodes.
        mesh_node_num (int): Number of mesh nodes.
        mesh_node_dim (int): Dimension of mesh nodes.
        mesh_node_emb_dim (int): Dimension of emdding mesh nodes.
        mesh_edge_dim (int): Dimension of mesh edges.
        mesh_edge_emb_dim (int): Dimension of emdding mesh edges.
        grid2mesh_edge_dim (int): Dimension of mesh edges in Grid2Mesh GNN.
        grid2mesh_edge_emb_dim (int): Dimension of emdding mesh edges in Grid2Mesh GNN.
        mesh2grid_edge_dim (int): Dimension of mesh edges in Mesh2Grid GNN.
        mesh2grid_edge_emb_dim (int): Dimension of emdding mesh edges in Mesh2Grid GNN.
        gnn_msg_steps (int): Step of gnn messages.
        node_output_dim (int): Dimension of output nodes.
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        grid_node_num: int,
        grid_node_dim: int,
        grid_node_emb_dim: int,
        mesh_node_num: int,
        mesh_node_dim: int,
        mesh_node_emb_dim: int,
        mesh_edge_dim: int,
        mesh_edge_emb_dim: int,
        grid2mesh_edge_dim: int,
        grid2mesh_edge_emb_dim: int,
        mesh2grid_edge_dim: int,
        mesh2grid_edge_emb_dim: int,
        gnn_msg_steps: int,
        node_output_dim: int,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.graphcast = nn.Sequential(
            (
                "encoder",
                GraphCastEncoder(
                    grid_node_num=grid_node_num,
                    grid_node_dim=grid_node_dim,
                    grid_node_emb_dim=grid_node_emb_dim,
                    mesh_node_num=mesh_node_num,
                    mesh_node_dim=mesh_node_dim,
                    mesh_node_emb_dim=mesh_node_emb_dim,
                    mesh_edge_dim=mesh_edge_dim,
                    mesh_edge_emb_dim=mesh_edge_emb_dim,
                    grid2mesh_edge_dim=grid2mesh_edge_dim,
                    grid2mesh_edge_emb_dim=grid2mesh_edge_emb_dim,
                    mesh2grid_edge_dim=mesh2grid_edge_dim,
                    mesh2grid_edge_emb_dim=mesh2grid_edge_emb_dim,
                ),
            ),
            (
                "processor",
                GraphCastProcessor(
                    grid_node_num=grid_node_num,
                    grid_node_emb_dim=grid_node_emb_dim,
                    mesh_node_num=mesh_node_num,
                    mesh_node_emb_dim=mesh_node_emb_dim,
                    mesh_edge_emb_dim=mesh_edge_emb_dim,
                    grid2mesh_edge_emb_dim=grid2mesh_edge_emb_dim,
                    mesh2grid_edge_emb_dim=mesh2grid_edge_emb_dim,
                    gnn_msg_steps=gnn_msg_steps,
                ),
            ),
            (
                "decoder",
                GraphCastDecoder(
                    grid_node_num=grid_node_num,
                    grid_node_emb_dim=grid_node_emb_dim,
                    mesh_node_num=mesh_node_num,
                    mesh_node_emb_dim=mesh_node_emb_dim,
                    mesh_edge_emb_dim=mesh_edge_emb_dim,
                    grid2mesh_edge_emb_dim=grid2mesh_edge_emb_dim,
                    mesh2grid_edge_emb_dim=mesh2grid_edge_emb_dim,
                    node_output_dim=node_output_dim,
                ),
            ),
        )

    def forward(
        self, x: Dict[str, "atmospheric_dataset.GraphGridMesh"]
    ) -> Dict[str, paddle.Tensor]:
        if self._input_transform is not None:
            x = self._input_transform(x)

        graph = x[self.input_keys[0]]
        y = self.graphcast(graph)

        if self._output_transform is not None:
            y = self._output_transform(x, y)
        return {self.output_keys[0]: y}
