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
from os import path as osp
from typing import Tuple

import numpy as np
import paddle
from paddle import io

from ppsci.data.dataset import airfoil_dataset

try:
    import pgl
except ModuleNotFoundError:
    pass

SU2_SHAPE_IDS = {
    "line": 3,
    "triangle": 5,
    "quad": 9,
}


class MeshCylinderDataset(io.Dataset):
    """Dataset for `MeshCylinder`.

    Args:
        input_keys (Tuple[str, ...]): Name of input data.
        label_keys (Tuple[str, ...]): Name of label data.
        data_dir (str): Directory of MeshCylinder data.
        mesh_graph_path (str): Path of mesh graph.

    Examples:
        >>> import ppsci
        >>> dataset = ppsci.data.dataset.MeshAirfoilDataset(
        ...     "input_keys": ("input",),
        ...     "label_keys": ("output",),
        ...     "data_dir": "/path/to/MeshAirfoilDataset",
        ...     "mesh_graph_path": "/path/to/file.su2",
        ... )  # doctest: +SKIP
    """

    use_pgl: bool = True

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        data_dir: str,
        mesh_graph_path: str,
    ):
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.data_dir = data_dir
        self.file_list = os.listdir(self.data_dir)
        self.len = len(self.file_list)
        self.mesh_graph = airfoil_dataset._get_mesh_graph(mesh_graph_path)

        self.normalization_factors = np.array(
            [[978.6001, 48.9258, 24.8404], [-692.3159, -6.9950, -24.8572]],
            dtype=paddle.get_default_dtype(),
        )

        self.nodes = self.mesh_graph[0]
        self.meshnodes = self.mesh_graph[0]
        self.edges = self.mesh_graph[1]
        self.elems_list = self.mesh_graph[2]
        self.marker_dict = self.mesh_graph[3]
        self.bounder = []
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
        with open(osp.join(self.data_dir, self.file_list[idx]), "r") as f:
            field = []
            pos = []
            for line in f.read().splitlines()[1:]:
                lines_pos = line.split(",")[1:3]
                lines_field = line.split(",")[3:]
                numbers_float = list(eval(i) for i in lines_pos)
                array = np.array(numbers_float, paddle.get_default_dtype())
                pos.append(array)
                numbers_float = list(eval(i) for i in lines_field)
                array = np.array(numbers_float, paddle.get_default_dtype())
                field.append(array)

        field = np.stack(field, axis=0)
        pos = np.stack(pos, axis=0)
        indexlist = []
        for i in range(self.meshnodes.shape[0]):
            b = self.meshnodes[i : (i + 1)]
            b = np.squeeze(b)
            index = np.nonzero(
                np.sum((pos == b), axis=1, dtype=paddle.get_default_dtype())
                == pos.shape[1]
            )
            indexlist.append(index)
        indexlist = np.stack(indexlist, axis=0)
        indexlist = np.squeeze(indexlist)
        fields = field[indexlist]
        velocity = self._get_params_from_name(self.file_list[idx])

        norm_aoa = velocity / 40
        # add physics parameters to graph
        nodes = np.concatenate(
            [
                self.nodes,
                np.repeat(a=norm_aoa, repeats=self.nodes.shape[0])[:, np.newaxis],
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
        data.velocity = velocity

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
        a = np.mean(data.y, axis=0)
        b = data.y.std(axis=0)
        b = np.maximum(b, std_epsilon).astype(paddle.get_default_dtype())
        data.y = (data.y - a) / b
        data.norm_max = a
        data.norm_min = b

        # find the face of the boundary,our cylinder dataset come from fluent solver
        with open(osp.join(osp.dirname(self.data_dir), "bounder"), "r") as f:
            field = []
            pos = []
            for line in f.read().splitlines()[1:]:
                lines_pos = line.split(",")[1:3]
                lines_field = line.split(",")[3:]
                numbers_float = list(eval(i) for i in lines_pos)
                array = np.array(numbers_float, paddle.get_default_dtype())
                pos.append(array)
                numbers_float = list(eval(i) for i in lines_field)
                array = np.array(numbers_float, paddle.get_default_dtype())
                field.append(array)

        field = np.stack(field, axis=0)
        pos = np.stack(pos, axis=0)

        indexlist = []
        for i in range(pos.shape[0]):
            b = pos[i : (i + 1)]
            b = np.squeeze(b)
            index = np.nonzero(
                np.sum((self.nodes == b), axis=1, dtype=paddle.get_default_dtype())
                == self.nodes.shape[1]
            )
            indexlist.append(index)

        indexlist = np.stack(indexlist, axis=0)
        indexlist = np.squeeze(indexlist)
        self.bounder = indexlist
        return data

    def _get_params_from_name(self, filename):
        s = filename.rsplit(".", 1)[0]
        reynolds = np.array(s[13:])[np.newaxis].astype(paddle.get_default_dtype())
        return reynolds
