import graphtype
import paddle
import paddle.nn as nn


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
    def __init__(self, config, src_type="mesh", dst_type="mesh"):
        super().__init__()

        self.src = src_type
        self.dst = dst_type
        self.config = config

        self.edge_in_dim = config.grid_node_emb_dim + config.mesh_node_emb_dim
        if src_type == "mesh" and dst_type == "mesh":
            self.edge_in_dim += config.mesh_edge_emb_dim
            self.edge_out_dim = config.mesh_edge_emb_dim
            self.node_in_dim = config.mesh_node_emb_dim + config.mesh_edge_emb_dim
            self.node_out_dim = config.mesh_node_emb_dim
        elif src_type == "grid" and dst_type == "mesh":
            self.edge_in_dim += config.grid2mesh_edge_emb_dim
            self.edge_out_dim = config.grid2mesh_edge_emb_dim
            self.node_in_dim = config.mesh_node_emb_dim + config.grid2mesh_edge_emb_dim
            self.node_out_dim = config.mesh_node_emb_dim
        elif src_type == "mesh" and dst_type == "grid":
            self.edge_in_dim += config.mesh2grid_edge_emb_dim
            self.edge_out_dim = config.mesh2grid_edge_emb_dim
            self.node_in_dim = config.grid_node_emb_dim + config.mesh2grid_edge_emb_dim
            self.node_out_dim = config.grid_node_emb_dim
        else:
            raise ValueError

        self.edge_layer = GraphCastMLP(self.edge_in_dim, self.edge_out_dim)
        self.node_layer = GraphCastMLP(self.node_in_dim, self.node_out_dim)

    def forward(self, graph: graphtype.GraphGridMesh):
        if self.src == "mesh" and self.dst == "mesh":
            edge_feats = graph.mesh_edge_feat
            src_node_feats = graph.mesh_node_feat
            dst_node_feats = graph.mesh_node_feat
            src_idx = graph.mesh2mesh_src_index
            dst_idx = graph.mesh2mesh_dst_index
            dst_node_num = self.config.mesh_node_num
        elif self.src == "grid" and self.dst == "mesh":
            edge_feats = graph.grid2mesh_edge_feat
            src_node_feats = graph.grid_node_feat
            dst_node_feats = graph.mesh_node_feat
            src_idx = graph.grid2mesh_src_index
            dst_idx = graph.grid2mesh_dst_index
            dst_node_num = self.config.mesh_node_num
        elif self.src == "mesh" and self.dst == "grid":
            edge_feats = graph.mesh2grid_edge_feat
            src_node_feats = graph.mesh_node_feat
            dst_node_feats = graph.grid_node_feat
            src_idx = graph.mesh2grid_src_index
            dst_idx = graph.mesh2grid_dst_index
            dst_node_num = self.config.grid_node_num

        # 更新edge特征
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
        # 更新node特征
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
    def __init__(self, config):
        super().__init__()

        self.grid_node_embedding = GraphCastMLP(
            config.grid_node_dim, config.grid_node_emb_dim
        )
        self.mesh_node_embedding = GraphCastMLP(
            config.mesh_node_dim, config.mesh_node_emb_dim
        )
        self.mesh_edge_embedding = GraphCastMLP(
            config.mesh_edge_dim, config.mesh_edge_emb_dim
        )
        self.grid2mesh_edge_embedding = GraphCastMLP(
            config.grid2mesh_edge_dim, config.grid2mesh_edge_emb_dim
        )
        self.mesh2grid_edge_embedding = GraphCastMLP(
            config.mesh2grid_edge_dim, config.mesh2grid_edge_emb_dim
        )

    def forward(self, graph: graphtype.GraphGridMesh):
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


class GraphCastGrid2Mesh(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.grid2mesh_gnn = GraphCastGNN(config, src_type="grid", dst_type="mesh")
        self.grid_node_layer = ResidualConnection(
            GraphCastMLP(config.grid_node_emb_dim, config.grid_node_emb_dim)
        )

    def forward(self, graph: graphtype.GraphGridMesh):
        graph = self.grid2mesh_gnn(graph)
        graph.grid_node_feat = self.grid_node_layer(graph.grid_node_feat)
        return graph


class GraphCastMesh2Grid(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.mesh2grid_gnn = GraphCastGNN(config, src_type="mesh", dst_type="grid")
        self.mesh_node_layer = ResidualConnection(
            GraphCastMLP(config.mesh_node_emb_dim, config.mesh_node_emb_dim)
        )

    def forward(self, graph: graphtype.GraphGridMesh):
        graph = self.mesh2grid_gnn(graph)
        graph.mesh_node_feat = self.mesh_node_layer(graph.mesh_node_feat)
        return graph


class GraphCastEncoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.embedding = GraphCastEmbedding(config)
        self.grid2mesh_gnn = GraphCastGrid2Mesh(config)

    def forward(self, graph: graphtype.GraphGridMesh):
        graph = self.embedding(graph)
        graph = self.grid2mesh_gnn(graph)
        return graph


class GraphCastDecoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.mesh2grid_gnn = GraphCastMesh2Grid(config)
        self.grid_node_layer = GraphCastMLP(
            config.grid_node_emb_dim,
            config.node_output_dim,
            latent_features=config.grid_node_emb_dim,
            layer_norm=False,
        )

    def forward(self, graph: graphtype.GraphGridMesh):
        graph = self.mesh2grid_gnn(graph)
        graph.grid_node_feat = self.grid_node_layer(graph.grid_node_feat)
        return graph


class GraphCastProcessor(nn.Layer):
    def __init__(self, config):
        super().__init__()

        self.processor = nn.Sequential()
        for idx in range(config.gnn_msg_steps):
            self.processor.add_sublayer(
                f"{idx}",
                GraphCastGNN(config, src_type="mesh", dst_type="mesh"),
            )

    def forward(self, graph: graphtype.GraphGridMesh):
        graph = self.processor(graph)
        return graph


class GraphCastNet(nn.Layer):
    def __init__(self, config):
        super().__init__()

        self.graphcast = nn.Sequential(
            ("encoder", GraphCastEncoder(config)),
            ("processor", GraphCastProcessor(config)),
            ("decoder", GraphCastDecoder(config)),
        )

    def forward(self, graph: graphtype.GraphGridMesh):
        graph = self.graphcast(graph)
        return graph
