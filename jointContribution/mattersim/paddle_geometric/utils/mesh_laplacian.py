from typing import Optional, Tuple

import paddle
from paddle import Tensor

from paddle_geometric.utils import add_self_loops, scatter, to_undirected


def get_mesh_laplacian(
    pos: Tensor,
    face: Tensor,
    normalization: Optional[str] = None,
) -> Tuple[Tensor, Tensor]:
    """Computes the mesh Laplacian of a mesh given by pos and face."""

    assert pos.shape[1] == 3 and face.shape[0] == 3
    num_nodes = pos.shape[0]

    def get_cots(left: Tensor, centre: Tensor, right: Tensor) -> Tensor:
        left_pos, central_pos, right_pos = pos[left], pos[centre], pos[right]
        left_vec = left_pos - central_pos
        right_vec = right_pos - central_pos
        dot = paddle.sum(left_vec * right_vec, axis=1)
        cross = paddle.norm(paddle.cross(left_vec, right_vec, axis=1), axis=1)
        cot = dot / cross
        return cot / 2.0

    cot_021 = get_cots(face[0], face[2], face[1])
    cot_102 = get_cots(face[1], face[0], face[2])
    cot_012 = get_cots(face[0], face[1], face[2])
    cot_weight = paddle.concat([cot_021, cot_102, cot_012])

    cot_index = paddle.concat([face[:2], face[1:], face[::2]], axis=1)
    cot_index, cot_weight = to_undirected(cot_index, cot_weight)

    cot_deg = scatter(cot_weight, cot_index[0], 0, num_nodes, reduce='sum')
    edge_index, _ = add_self_loops(cot_index, num_nodes=num_nodes)
    edge_weight = paddle.concat([cot_weight, -cot_deg], axis=0)

    if normalization is not None:

        def get_areas(left: Tensor, centre: Tensor, right: Tensor) -> Tensor:
            central_pos = pos[centre]
            left_vec = pos[left] - central_pos
            right_vec = pos[right] - central_pos
            cross = paddle.norm(paddle.cross(left_vec, right_vec, axis=1), axis=1)
            area = cross / 6.0
            return area / 2.0

        area_021 = get_areas(face[0], face[2], face[1])
        area_102 = get_areas(face[1], face[0], face[2])
        area_012 = get_areas(face[0], face[1], face[2])
        area_weight = paddle.concat([area_021, area_102, area_012])
        area_index = paddle.concat([face[:2], face[1:], face[::2]], axis=1)
        area_index, area_weight = to_undirected(area_index, area_weight)
        area_deg = scatter(area_weight, area_index[0], 0, num_nodes, 'sum')

        if normalization == 'sym':
            area_deg_inv_sqrt = area_deg.pow(-0.5)
            area_deg_inv_sqrt = paddle.where(area_deg_inv_sqrt == float('inf'), paddle.zeros_like(area_deg_inv_sqrt), area_deg_inv_sqrt)
            edge_weight = (area_deg_inv_sqrt[edge_index[0]] * edge_weight *
                           area_deg_inv_sqrt[edge_index[1]])
        elif normalization == 'rw':
            area_deg_inv = 1.0 / area_deg
            area_deg_inv = paddle.where(area_deg_inv == float('inf'), paddle.zeros_like(area_deg_inv), area_deg_inv)
            edge_weight = area_deg_inv[edge_index[0]] * edge_weight

    return edge_index, edge_weight
