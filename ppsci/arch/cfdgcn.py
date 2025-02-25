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

import os
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypeVar
from typing import Union

import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F

from ppsci.data.dataset import airfoil_dataset

try:
    import pgl
except ModuleNotFoundError:
    pass

GenTensor = TypeVar("GenTensor", paddle.Tensor, np.ndarray)

SU2_SHAPE_IDS = {
    "line": 3,
    "triangle": 5,
    "quad": 9,
}


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


def is_cw(
    points: paddle.Tensor, triangles: paddle.Tensor, ret_val=False
) -> Union[bool, paddle.Tensor]:
    tri_pts = points[triangles]
    a = tri_pts[:, 0] - tri_pts[:, 1]
    b = tri_pts[:, 1] - tri_pts[:, 2]
    cross = b[:, 0] * a[:, 1] - b[:, 1] * a[:, 0]

    if not ret_val:
        return cross > 0
    else:
        return cross


def left_orthogonal(v: paddle.Tensor) -> paddle.Tensor:
    return paddle.stack([-v[..., 1], v[..., 0]], axis=-1)


def signed_dist_graph(
    nodes: paddle.Tensor, marker_inds, with_sign=False
) -> paddle.Tensor:
    # assumes shape is convex
    # approximate signed distance by distance to closest point on surface
    signed_dists = paddle.zeros([nodes.shape[0]], dtype=paddle.float32)
    marker_nodes = nodes[marker_inds]
    if type(marker_inds) is paddle.Tensor:
        marker_inds = marker_inds.tolist()
    marker_inds = set(marker_inds)

    if with_sign:
        marker_surfaces = marker_nodes[:-1] - marker_nodes[1:]
        last_surface = marker_nodes[-1] - marker_nodes[0]
        marker_surfaces = paddle.concat([marker_surfaces, last_surface.unsqueeze(0)])
        normals = left_orthogonal(marker_surfaces) / marker_surfaces.norm(
            dim=1
        ).unsqueeze(1)
    for i, x in enumerate(nodes):
        if i not in marker_inds:
            vecs = marker_nodes - x
            dists = paddle.linalg.norm(vecs, axis=1)
            min_dist = dists.min()

            if with_sign:
                # if sign is requested, check if inside marker shape
                # dot product with normals to find if inside shape
                surface_dists = (vecs * normals).sum(dim=1)
                if (surface_dists < 0).unique().shape[0] == 1:
                    # if all point in same direction it is inside
                    min_dist *= -1

            signed_dists[i] = min_dist
    return signed_dists


def quad2tri(elems: np.array) -> Tuple[List[int], Union[List[int], paddle.Tensor]]:
    new_elems = []
    new_edges = []
    for e in elems:
        if len(e) <= 3:
            new_elems.append(e)
        else:
            new_elems.append([e[0], e[1], e[2]])
            new_elems.append([e[0], e[2], e[3]])
            new_edges.append(paddle.to_tensor([[e[0]], [e[2]]], dtype=paddle.int64))
    new_edges = (
        paddle.concat(new_edges, axis=1)
        if new_edges
        else paddle.to_tensor([], dtype=paddle.int64)
    )
    return new_elems, new_edges


def write_graph_mesh(
    output_filename: str,
    points: GenTensor,
    elems_list: Sequence[Sequence[Sequence[int]]],
    marker_dict: Dict[str, Sequence[Sequence[int]]],
    dims: int = 2,
) -> None:
    def seq2str(s: Sequence[int]) -> str:
        return " ".join(str(x) for x in s)

    with open(output_filename, "w") as f:
        f.write(f"NDIME={dims}\n")

        num_points = points.shape[0]
        f.write(f"NPOIN={num_points}\n")
        for i, p in enumerate(points):
            f.write(f"{seq2str(p.tolist())} {i}\n")
        f.write("\n")

        num_elems = sum([len(elems) for elems in elems_list])
        f.write(f"NELEM={num_elems}\n")
        for elems in elems_list:
            for e in elems:
                if len(e) != 3 and len(e) != 4:
                    raise ValueError(
                        f"Meshes only support triangles and quadrilaterals, "
                        f"passed element had {len(e)} vertices."
                    )
                elem_id = (
                    SU2_SHAPE_IDS["triangle"] if len(e) == 3 else SU2_SHAPE_IDS["quad"]
                )
                f.write(f"{elem_id} {seq2str(e)}\n")
        f.write("\n")

        num_markers = len(marker_dict)
        f.write(f"NMARK={num_markers}\n")
        for marker_tag in marker_dict:
            f.write(f"MARKER_TAG={marker_tag}\n")
            marker_elems = marker_dict[marker_tag]
            f.write(f"MARKER_ELEMS={len(marker_elems)}\n")
            for m in marker_elems:
                f.write(f'{SU2_SHAPE_IDS["line"]} {seq2str(m)}\n')
        f.write("\n")


class CFDGCN(nn.Layer):
    """Graph Neural Networks for Fluid Flow Prediction.

    [Filipe De Avila Belbute-Peres, Thomas Economon, Zico Kolter Proceedings of the 37th International Conference on Machine Learning, PMLR 119:2402-2411, 2020.](https://proceedings.mlr.press/v119/de-avila-belbute-peres20a.html)

    Code reference: https://github.com/locuslab/cfd-gcn

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("input", ).
        output_keys (Tuple[str, ...]): Name of output keys, such as ("pred", ).
        config_file (str): Name of configuration file for su2 module.
        coarse_mesh (str): Path of coarse mesh file.
        fine_marker_dict (Dict[str, List[List[int]]]): Dict of fine marker.
        process_sim (Callable, optional): Preprocess funtion. Defaults to `lambda x, y: x`.
        freeze_mesh (bool, optional): Whether set `stop_gradient=True` for nodes. Defaults to False.
        num_convs (int, optional): Number of conv layers. Defaults to 6.
        num_end_convs (int, optional): Number of end conv layers. Defaults to 3.
        hidden_channel (int, optional): Number of channels of hidden layer. Defaults to 512.
        out_channel (int, optional): Number of channels of output. Defaults to 3.
        su2_module (Optional[Callable]): SU2Module Object. Defaults to None.

    Examples:
        >>> import ppsci
        >>> import su2paddle # doctest: +SKIP
        >>> model = ppsci.arch.CFDGCN(
        ...     input_keys=("input"),
        ...     output_keys=("pred"),
        ...     config_file="/path/to/file.cfg",
        ...     coarse_mesh="/path/to/file.su2",
        ...     process_sim=None,
        ...     freeze_mesh=False,
        ...     num_convs=6,
        ...     num_end_convs=3,
        ...     hidden_channel=512,
        ...     out_channel=3,
        ...     su2_module=su2paddle.SU2Module,
        ... ) # doctest: +SKIP
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        config_file: str,
        coarse_mesh: str,
        fine_marker_dict: Dict[str, List[List[int]]],
        process_sim: Callable = lambda x, y: x,
        freeze_mesh: bool = False,
        num_convs: int = 6,
        num_end_convs: int = 3,
        hidden_channel: int = 512,
        out_channel: int = 3,
        su2_module: Optional[Callable] = None,
    ):

        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        meshes_temp_dir = "temp_meshes"
        os.makedirs(meshes_temp_dir, exist_ok=True)
        self.mesh_file = os.path.join(meshes_temp_dir, f"{str(os.getpid())}_mesh.su2")

        if not coarse_mesh:
            raise ValueError("Need to provide a coarse mesh for CFD-GCN.")
        nodes, edges, self.elems, self.marker_dict = airfoil_dataset._get_mesh_graph(
            coarse_mesh
        )
        if not freeze_mesh:
            self.nodes = paddle.to_tensor(nodes, stop_gradient=False)
        else:
            self.nodes = paddle.to_tensor(nodes, stop_gradient=True)

        self.elems, new_edges = quad2tri(sum(self.elems, []))
        self.elems = [self.elems]

        if is_cw(self.nodes, paddle.to_tensor(self.elems[0])).nonzero().shape[0] != 0:
            raise ("Mesh has flipped elems.")

        self.edges = paddle.to_tensor(edges)
        self.edges = paddle.concat([self.edges, new_edges], axis=1)
        self.marker_inds = paddle.to_tensor(sum(self.marker_dict.values(), [])).unique()
        self.fine_marker_dict = paddle.to_tensor(fine_marker_dict["airfoil"]).unique()
        self.process_sim = process_sim

        self.write_mesh_file(
            self.nodes, self.elems, self.marker_dict, filename=self.mesh_file
        )
        self.su2 = su2_module(config_file, mesh_file=self.mesh_file)
        self.sdf = None

        self.num_convs = num_end_convs
        self.convs = []
        if self.num_convs > 0:
            self.convs = nn.LayerList()
            in_channels = out_channel + hidden_channel
            for i in range(self.num_convs - 1):
                self.convs.append(pgl.nn.GCNConv(in_channels, hidden_channel))
                in_channels = hidden_channel
            self.convs.append(pgl.nn.GCNConv(in_channels, out_channel))

        self.num_pre_convs = num_convs - num_end_convs
        self.pre_convs = []
        if self.num_pre_convs > 0:
            in_channels = 5 + 1  # one extra channel for sdf
            self.pre_convs = nn.LayerList()
            for i in range(self.num_pre_convs - 1):
                self.pre_convs.append(pgl.nn.GCNConv(in_channels, hidden_channel))
                in_channels = hidden_channel
            self.pre_convs.append(pgl.nn.GCNConv(in_channels, hidden_channel))

    def forward(self, x: Dict[str, "pgl.Graph"]) -> Dict[str, paddle.Tensor]:
        graph = x[self.input_keys[0]]
        batch_size = graph.shape[0]
        x_list = paddle.split(graph.x, batch_size)
        fine_x_list = []

        for idx in range(batch_size):
            x = x_list[idx]
            if self.sdf is None:
                with paddle.no_grad():
                    self.sdf = signed_dist_graph(
                        x[:, :2], self.fine_marker_dict
                    ).unsqueeze(1)
            fine_x = paddle.concat([x, self.sdf], axis=1)
            for conv in self.pre_convs:
                fine_x = F.relu(conv(graph, fine_x))
            fine_x_list.append(fine_x)
        nodes_input = self.get_nodes().tile([batch_size, 1, 1])

        batch_y = self.su2(
            nodes_input[..., 0],
            nodes_input[..., 1],
            graph.aoa[..., None],
            graph.mach_or_reynolds[..., None],
        )
        batch_y = self.process_sim(batch_y, False)

        pred_fields = []
        for idx in range(batch_size):
            coarse_y = paddle.stack([y[idx].flatten() for y in batch_y], axis=1).astype(
                "float32"
            )
            nodes = self.get_nodes()
            x = x_list[idx]
            fine_y = _knn_interpolate(
                features=coarse_y, coarse_nodes=nodes[:, :2], fine_nodes=x[:, :2]
            )
            fine_y = paddle.concat([fine_y, fine_x_list[idx]], axis=1)

            for conv in self.convs[:-1]:
                fine_y = F.relu(conv(graph, fine_y))
            fine_y = self.convs[-1](graph, fine_y)
            pred_fields.append(fine_y)
        pred_fields = paddle.concat(pred_fields, axis=0)
        return {self.output_keys[0]: pred_fields}

    def get_nodes(self) -> paddle.Tensor:
        return self.nodes

    @staticmethod
    def write_mesh_file(
        x: paddle.Tensor,
        elems: paddle.Tensor,
        marker_dict: Dict[str, Sequence[Sequence[int]]],
        filename: str = "mesh.su2",
    ) -> None:
        write_graph_mesh(filename, x[:, :2], elems, marker_dict)
