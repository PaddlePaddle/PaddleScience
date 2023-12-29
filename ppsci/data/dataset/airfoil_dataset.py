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

import os
import pickle
from os import path as osp
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import paddle
from paddle import io

try:
    import pgl
except ModuleNotFoundError:
    pass

SU2_SHAPE_IDS = {
    "line": 3,
    "triangle": 5,
    "quad": 9,
}


#  HACK: Simplify code below
def _get_mesh_graph(
    mesh_filename: str, dtype: np.dtype = np.float32
) -> Tuple[np.ndarray, np.ndarray, List[List[List[int]]], Dict[str, List[List[int]]]]:
    def get_rhs(s: str) -> str:
        return s.split("=")[-1]

    marker_dict = {}
    with open(mesh_filename) as f:
        for line in f:
            if line.startswith("NPOIN"):
                num_points = int(get_rhs(line))
                mesh_points = [
                    [float(p) for p in f.readline().split()[:2]]
                    for _ in range(num_points)
                ]
                nodes = np.array(mesh_points, dtype=dtype)
            elif line.startswith("NMARK"):
                num_markers = int(get_rhs(line))
                for _ in range(num_markers):
                    line = f.readline()
                    assert line.startswith("MARKER_TAG")
                    marker_tag = get_rhs(line).strip()
                    num_elems = int(get_rhs(f.readline()))
                    marker_elems = [
                        [int(e) for e in f.readline().split()[-2:]]
                        for _ in range(num_elems)
                    ]
                    marker_dict[marker_tag] = marker_elems
            elif line.startswith("NELEM"):
                edges = []
                triangles = []
                quads = []
                num_edges = int(get_rhs(line))
                for _ in range(num_edges):
                    elem = [int(p) for p in f.readline().split()]
                    if elem[0] == SU2_SHAPE_IDS["triangle"]:
                        n = 3
                        triangles.append(elem[1 : 1 + n])
                    elif elem[0] == SU2_SHAPE_IDS["quad"]:
                        n = 4
                        quads.append(elem[1 : 1 + n])
                    else:
                        raise NotImplementedError
                    elem = elem[1 : 1 + n]
                    edges += [[elem[i], elem[(i + 1) % n]] for i in range(n)]
                edges = np.array(edges, dtype=np.compat.long).transpose()
                elems = [triangles, quads]
    return nodes, edges, elems, marker_dict


class MeshAirfoilDataset(io.Dataset):
    """Dataset for `MeshAirfoil`.

    Args:
        input_keys (Tuple[str, ...]): Name of input data.
        label_keys (Tuple[str, ...]): Name of label data.
        data_dir (str): Directory of MeshAirfoil data.
        mesh_graph_path (str): Path of mesh graph.
        transpose_edges (bool, optional): Whether transpose the edges array from (2, num_edges) to (num_edges, 2) for convenient of slicing.

    Examples:
        >>> import ppsci
        >>> dataset = ppsci.data.dataset.MeshAirfoilDataset(
        ...     "input_keys": ("input",),
        ...     "label_keys": ("output",),
        ...     "data_dir": "/path/to/MeshAirfoilDataset",
        ...     "mesh_graph_path": "/path/to/file.su2",
        ...     "transpose_edges": False,
        ... )  # doctest: +SKIP
    """

    use_pgl: bool = True

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        data_dir: str,
        mesh_graph_path: str,
        transpose_edges: bool = False,
    ):
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.data_dir = data_dir
        self.file_list = os.listdir(self.data_dir)
        self.len = len(self.file_list)
        self.mesh_graph = _get_mesh_graph(mesh_graph_path)

        with open(osp.join(osp.dirname(self.data_dir), "train_max_min.pkl"), "rb") as f:
            self.normalization_factors = pickle.load(f)

        self.nodes = self.mesh_graph[0]
        self.edges = self.mesh_graph[1]
        if transpose_edges:
            self.edges = self.edges.transpose([1, 0])
        self.elems_list = self.mesh_graph[2]
        self.marker_dict = self.mesh_graph[3]
        self.node_markers = np.full([self.nodes.shape[0], 1], fill_value=-1)
        for i, (marker_tag, marker_elems) in enumerate(self.marker_dict.items()):
            for elem in marker_elems:
                self.node_markers[elem[0]] = i
                self.node_markers[elem[1]] = i

        self.raw_graphs = [self.get(i) for i in range(len(self))]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (
            {
                self.input_keys[0]: self.raw_graphs[idx],
            },
            {
                self.label_keys[0]: self.raw_graphs[idx],
            },
            None,
        )

    def get(self, idx):
        with open(osp.join(self.data_dir, self.file_list[idx]), "rb") as f:
            fields = pickle.load(f)
        fields = self._preprocess(fields)
        aoa, reynolds, mach = self._get_params_from_name(self.file_list[idx])
        # aoa = aoa
        mach_or_reynolds = mach if reynolds is None else reynolds
        # mach_or_reynolds = mach_or_reynolds
        norm_aoa = aoa / 10
        norm_mach_or_reynolds = (
            mach_or_reynolds if reynolds is None else (mach_or_reynolds - 1.5e6) / 1.5e6
        )

        nodes = np.concatenate(
            [
                self.nodes,
                np.repeat(a=norm_aoa, repeats=self.nodes.shape[0])[:, np.newaxis],
                np.repeat(a=norm_mach_or_reynolds, repeats=self.nodes.shape[0])[
                    :, np.newaxis
                ],
                self.node_markers,
            ],
            axis=-1,
        ).astype(paddle.get_default_dtype())

        data = pgl.Graph(
            num_nodes=nodes.shape[0],
            edges=self.edges,
        )
        data.x = nodes
        data.y = fields
        data.pos = self.nodes
        data.edge_index = self.edges

        sender = data.x[data.edge_index[0]]
        receiver = data.x[data.edge_index[1]]
        relation_pos = sender[:, 0:2] - receiver[:, 0:2]
        post = np.linalg.norm(relation_pos, ord=2, axis=1, keepdims=True).astype(
            paddle.get_default_dtype()
        )
        data.edge_attr = post
        std_epsilon = [1e-8]
        a = np.mean(data.edge_attr, axis=0)
        b = data.edge_attr.std(axis=0)
        b = np.maximum(b, std_epsilon).astype(paddle.get_default_dtype())
        data.edge_attr = (data.edge_attr - a) / b
        data.aoa = aoa
        data.norm_aoa = norm_aoa
        data.mach_or_reynolds = mach_or_reynolds
        data.norm_mach_or_reynolds = norm_mach_or_reynolds
        return data

    def _preprocess(self, tensor_list, stack_output=True):
        data_max, data_min = self.normalization_factors
        normalized_tensors = []
        for i in range(len(tensor_list)):
            normalized = (tensor_list[i] - data_min[i]) / (
                data_max[i] - data_min[i]
            ) * 2 - 1
            normalized_tensors.append(normalized)
        if stack_output:
            normalized_tensors = np.stack(normalized_tensors, axis=1)
        return normalized_tensors

    def _get_params_from_name(self, filename):
        s = filename.rsplit(".", 1)[0].split("_")
        aoa = np.array(s[s.index("aoa") + 1])[np.newaxis].astype(
            paddle.get_default_dtype()
        )
        reynolds = s[s.index("re") + 1]
        reynolds = (
            np.array(reynolds)[np.newaxis].astype(paddle.get_default_dtype())
            if reynolds != "None"
            else None
        )
        mach = np.array(s[s.index("mach") + 1])[np.newaxis].astype(
            paddle.get_default_dtype()
        )
        return aoa, reynolds, mach
