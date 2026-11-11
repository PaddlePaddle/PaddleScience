"""gencast denoiser backbone: GraphCastNet + sparse MeshTransformer (jointContribution/gencast/conf/gencast.yaml)."""
import functools
import sys
import types
from pathlib import Path

import numpy as np
import paddle
from cases import _common as C
from omegaconf import OmegaConf

GENCAST_DIR = Path(__file__).resolve().parents[3] / "jointContribution" / "gencast"
# gencast does `from graphcast import ...` (jointContribution/graphcast) and that
# module does `import sparse_transformer` (jointContribution/gencast)
for _path in (str(GENCAST_DIR.parent), str(GENCAST_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import sparse_transformer  # noqa: E402
from graphcast import graphcast as graphcast_arch  # noqa: E402
from graphcast import graphtype  # noqa: E402

EXAMPLE = "jointContribution/gencast"
NOTE = (
    "ported from jointContribution/gencast/benchmark_cinn.py; denoiser_architecture_config "
    "(mesh_size=4, 1.0 deg grid, 16-layer triblockdiag MeshTransformer); graph structure "
    "rebuilt by graphtype.GraphGridMesh(config), random physical grid channels, zero-padded "
    "mesh channels, random 16-dim global_norm_conditioning (stands in for the diffusion "
    "noise-level encoding, so it comes from build_batch) and random target, MSE loss; "
    "workarounds carried over for both modes: sparse_transformer.layernorm replaced by a "
    "param-free F.layer_norm (it built nn.LayerNorm in forward), and the constant scipy "
    "adjacency / k_hop mask / block size are precomputed once and monkeypatched in "
    "(scipy + .numpy() cannot be traced); the graph is passed as flat tensors and rebuilt "
    "in forward, the grid-node tensor is returned instead of the graph object"
)
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
INDEX_NAMES = (
    "mesh2mesh_src_index",
    "mesh2mesh_dst_index",
    "grid2mesh_src_index",
    "grid2mesh_dst_index",
    "mesh2grid_src_index",
    "mesh2grid_dst_index",
)
FEAT_NAMES = (
    "grid_node_feat",
    "mesh_node_feat",
    "mesh_edge_feat",
    "grid2mesh_edge_feat",
    "mesh2grid_edge_feat",
)


@functools.lru_cache(maxsize=None)
def _config():
    return OmegaConf.load(
        GENCAST_DIR / "conf" / "gencast.yaml"
    ).denoiser_architecture_config


@functools.lru_cache(maxsize=None)
def _graph_structure():
    return graphtype.GraphGridMesh(_config())  # numpy only


def _layernorm(x, create_scale: bool, create_offset: bool):
    assert not create_scale and not create_offset
    return paddle.nn.functional.layer_norm(
        x, normalized_shape=x.shape[-1:], weight=None, bias=None, epsilon=1e-5
    )


class _ConstantAdjacency:
    """Stands in for the constant scipy mesh adjacency; Block does `adj ** k_hop`."""

    def __init__(self, num_nodes):
        self.shape = (num_nodes, num_nodes)

    def __pow__(self, k_hop):
        return self


def _install_patches():
    """Precompute the constant mesh attention mask (eager tensor) and patch helpers."""
    from scipy import sparse

    config = _config()
    graph = _graph_structure()
    num_nodes = int(graph.mesh_node_feat.shape[0])
    adj_mat = sparse.csr_matrix((num_nodes, num_nodes), dtype=np.bool_)
    adj_mat[
        np.asarray(graph.mesh2mesh_src_index), np.asarray(graph.mesh2mesh_dst_index)
    ] = True
    adj_mat[np.arange(num_nodes), np.arange(num_nodes)] = True  # add_self_edges
    mask = adj_mat**config.sparse_transformer_config.attention_k_hop
    block_size = int(sparse_transformer.get_mask_block_size(mask))
    num_padding_nodes = int(np.ceil(num_nodes / block_size) * block_size - num_nodes)
    mask_blocks = sparse_transformer.mask_block_diags(
        mask, num_padding_nodes, block_size
    )
    constant_adjacency = _ConstantAdjacency(num_nodes)
    sparse_transformer.layernorm = _layernorm
    sparse_transformer._get_adj_matrix_for_edge_set = (
        lambda graph, add_self_edges: constant_adjacency
    )
    sparse_transformer.get_mask_block_size = lambda mask: block_size
    sparse_transformer.mask_block_diags = (
        lambda mask, num_padding_nodes, block_size: mask_blocks
    )


class GenCastDenoiserBackbone(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.graphcast_net = graphcast_arch.GraphCastNet(config=config)

    def forward(self, graph):
        return self.graphcast_net(graph=graph).grid_node_feat


def build_model():
    model = GenCastDenoiserBackbone(_config())
    _install_patches()  # after init so no RNG draw shifts the weights
    return model


def build_batch():
    config = _config()
    graph = _graph_structure()
    feat_dim = config.grid_node_dim - graph.grid_node_feat.shape[-1]
    inputs = {
        name: paddle.to_tensor(getattr(graph, name), dtype="int64")
        for name in INDEX_NAMES
    }
    for name in FEAT_NAMES:
        inputs[name] = paddle.to_tensor(getattr(graph, name), dtype="float32")
    inputs["grid_node_feat"] = paddle.concat(
        [
            C.randn([graph.grid_num_nodes, BATCH_SIZE, feat_dim]),
            inputs["grid_node_feat"],
        ],
        axis=-1,
    )
    inputs["mesh_node_feat"] = paddle.concat(
        [
            paddle.zeros([graph.mesh_num_nodes, BATCH_SIZE, feat_dim]),
            inputs["mesh_node_feat"],
        ],
        axis=-1,
    )
    inputs["global_norm_conditioning"] = C.randn(
        [BATCH_SIZE, config.norm_conditioning_feat]
    )
    label = C.randn([graph.grid_num_nodes, BATCH_SIZE, config.node_output_dim])
    return inputs, {"grid_node_feat": label}


def forward(model, inputs):
    graph = types.SimpleNamespace(**inputs)
    return {"grid_node_feat": model(graph)}


def compute_loss(model, inputs, labels):
    out = forward(model, inputs)["grid_node_feat"]
    return C.mse(out, labels["grid_node_feat"])
