"""graphcast GraphCastNet on a 1.0 deg grid (examples/graphcast/conf/graphcast_small.yaml)."""
import functools
import types

import paddle
from cases import _common as C

import ppsci

EXAMPLE = "graphcast"
NOTE = (
    "ported from examples/graphcast/benchmark_cinn.py; graph structure (edges + "
    "structural lat/lon features) rebuilt from mesh_size=5/resolution=1.0, 183 random "
    "grid channels + 3 structural (=186), mesh nodes zero-padded the same way; random "
    "label, MSE on grid-node output; the graph is passed as flat tensors and rebuilt "
    "into a plain attribute container inside forward, and the grid-node tensor is "
    "returned instead of the GraphGridMesh object (to_static cannot return it); the "
    "embedding layer rebinds every feature before the in-place += updates, so the "
    "batch tensors are never mutated"
)
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
MESH_SIZE = 5
RESOLUTION = 1.0
RADIUS_QUERY_FRACTION_EDGE_LENGTH = 0.6
MESH2GRID_EDGE_NORMALIZATION_FACTOR = 0.6180338738074472
GRID_NODE_NUM = 65160
MESH_NODE_NUM = 10242
GRID_NODE_DIM = 186
NODE_OUTPUT_DIM = 83
STRUCTURAL_NODE_DIM = 3
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


def build_model():
    return ppsci.arch.GraphCastNet(
        input_keys=("input",),
        output_keys=("pred",),
        grid_node_dim=GRID_NODE_DIM,
        grid_node_num=GRID_NODE_NUM,
        grid_node_emb_dim=512,
        mesh_node_dim=GRID_NODE_DIM,
        mesh_node_num=MESH_NODE_NUM,
        mesh_edge_dim=4,
        mesh_node_emb_dim=512,
        mesh_edge_emb_dim=512,
        mesh2grid_edge_dim=4,
        mesh2grid_edge_emb_dim=512,
        grid2mesh_edge_dim=4,
        grid2mesh_edge_emb_dim=512,
        gnn_msg_steps=16,
        node_output_dim=NODE_OUTPUT_DIM,
    )


@functools.lru_cache(maxsize=None)
def _graph_structure():
    from ppsci.data.dataset import atmospheric_dataset

    return atmospheric_dataset.GraphGridMesh(
        mesh_size=MESH_SIZE,
        radius_query_fraction_edge_length=RADIUS_QUERY_FRACTION_EDGE_LENGTH,
        mesh2grid_edge_normalization_factor=MESH2GRID_EDGE_NORMALIZATION_FACTOR,
        resolution=RESOLUTION,
    )


def build_batch():
    graph = _graph_structure()
    feat_dim = GRID_NODE_DIM - STRUCTURAL_NODE_DIM  # 183 synthetic channels
    inputs = {
        name: paddle.to_tensor(getattr(graph, name), dtype="int64")
        for name in INDEX_NAMES
    }
    for name in FEAT_NAMES:
        inputs[name] = paddle.to_tensor(getattr(graph, name), dtype="float32")
    inputs["grid_node_feat"] = paddle.concat(
        [C.randn([GRID_NODE_NUM, BATCH_SIZE, feat_dim]), inputs["grid_node_feat"]],
        axis=-1,
    )
    inputs["mesh_node_feat"] = paddle.concat(
        [
            paddle.zeros([MESH_NODE_NUM, BATCH_SIZE, feat_dim]),
            inputs["mesh_node_feat"],
        ],
        axis=-1,
    )
    label = C.randn([GRID_NODE_NUM, BATCH_SIZE, NODE_OUTPUT_DIM])
    return inputs, {"pred": label}


def forward(model, inputs):
    graph = types.SimpleNamespace(**inputs)
    return {"pred": model({"input": graph})["pred"].grid_node_feat}


def compute_loss(model, inputs, labels):
    return C.mse(forward(model, inputs)["pred"], labels["pred"])
