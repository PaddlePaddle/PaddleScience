import enum

import numpy as np
from dataset.Load_mesh import CustomGraphData
from utils.paddle_aux import scatter_paddle


class MeshType(enum.IntEnum):
    Triangle = 1
    Tetrahedron = 2
    Quad = 3
    Line = 4
    Flat = 5


def calc_cell_centered_with_node_attr(
    node_attr, cells_node, cells_index, reduce="mean", map=True
):
    if tuple(cells_node.shape) != tuple(cells_index.shape):
        raise ValueError("wrong cells_node/cells_index dim")
    if len(tuple(cells_node.shape)) > 1:
        cells_node = cells_node.view(-1)
    if len(tuple(cells_index.shape)) > 1:
        cells_index = cells_index.view(-1)
    if map:
        mapped_node_attr = node_attr[cells_node]
    else:
        mapped_node_attr = node_attr
    cell_attr = scatter_paddle(
        src=mapped_node_attr, index=cells_index, dim=0, reduce=reduce
    )
    return cell_attr


def calc_node_centered_with_cell_attr(
    cell_attr, cells_node, cells_index, reduce="mean", map=True
):
    if tuple(cells_node.shape) != tuple(cells_index.shape):
        raise ValueError("wrong cells_node/cells_index dim ")
    if len(tuple(cells_node.shape)) > 1:
        cells_node = cells_node.view(-1)
    if len(tuple(cells_index.shape)) > 1:
        cells_index = cells_index.view(-1)
    if map:
        maped_cell_attr = cell_attr[cells_index]
    else:
        maped_cell_attr = cell_attr
    cell_attr = scatter_paddle(
        src=maped_cell_attr, index=cells_node, dim=0, reduce=reduce
    )
    return cell_attr


def decompose_and_trans_node_attr_to_cell_attr_graph(
    graph, has_changed_node_attr_to_cell_attr
):
    x, edge_index, edge_attr, face, global_attr, ball_edge_index = (
        None,
        None,
        None,
        None,
        None,
        None,
    )
    for key in graph.keys:
        if key == "x":
            x = graph.x
        elif key == "edge_index":
            edge_index = graph.edge_index
        elif key == "edge_attr":
            edge_attr = graph.edge_attr
        elif key == "global_attr":
            global_attr = graph.global_attr
        elif key == "face":
            face = graph.face
        elif key == "ball_edge_index":
            ball_edge_index = graph.ball_edge_index
        else:
            pass
    return x, edge_index, edge_attr, face, global_attr, ball_edge_index


def copy_geometric_data(graph, has_changed_node_attr_to_cell_attr):
    """return a copy of gl.graph.Graph
    This function should be carefully used based on
    which keys in a given graph.
    """
    (
        node_attr,
        edge_index,
        edge_attr,
        face,
        global_attr,
        ball_edge_index,
    ) = decompose_and_trans_node_attr_to_cell_attr_graph(
        graph, has_changed_node_attr_to_cell_attr
    )
    ret = CustomGraphData(
        x=node_attr,
        edge_index=edge_index,
        edge_attr=edge_attr,
        face=face,
        ball_edge_index=ball_edge_index,
    )
    ret.keys = ["x", "num_graphs", "edge_index", "batch", "edge_attr"]
    return ret


def shuffle_np(array):
    array_t = array.copy()
    np.random.shuffle(array_t)
    return array_t
