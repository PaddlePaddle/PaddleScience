import os
import pickle
from pathlib import Path

import numpy as np
import paddle
import pgl
from pgl.utils.data.dataloader import Dataloader, Dataset

from mesh_utils import get_mesh_graph


class MeshAirfoilDataset(Dataset):
    def __init__(self, root, mode='train'):
        super().__init__()

        self.mode = mode
        self.data_dir = Path(root) / ('outputs_' + mode)
        self.file_list = os.listdir(self.data_dir)
        self.len = len(self.file_list)

        self.mesh_graph = get_mesh_graph(Path(root) / 'mesh_fine.su2')

        # either [maxes, mins] or [means, stds] from data for normalization
        # with open(self.data_dir / 'train_mean_std.pkl', 'rb') as f:
        with open(self.data_dir.parent / 'train_max_min.pkl', 'rb') as f:
            self.normalization_factors = pickle.load(f)

        self.nodes = self.mesh_graph[0]
        self.edges = paddle.to_tensor(self.mesh_graph[1]).transpose([1, 0])
        self.elems_list = self.mesh_graph[2]
        self.marker_dict = self.mesh_graph[3]
        self.node_markers = np.full([self.nodes.shape[0], 1], fill_value=-1)
        for i, (marker_tag, marker_elems) in enumerate(self.marker_dict.items()):
            for elem in marker_elems:
                self.node_markers[elem[0]] = i
                self.node_markers[elem[1]] = i

        self.graphs = []

        for idx in range(self.len):
            with open(self.data_dir / self.file_list[idx], 'rb') as f:
                fields = pickle.load(f)
            fields = paddle.to_tensor(self.preprocess(fields))

            aoa, reynolds, mach = self.get_params_from_name(self.file_list[idx])
            aoa = paddle.to_tensor(aoa)
            mach_or_reynolds = paddle.to_tensor(mach if reynolds is None else reynolds)

            norm_aoa = paddle.to_tensor(aoa / 10)
            norm_mach_or_reynolds = paddle.to_tensor(
                mach_or_reynolds if reynolds is None else (mach_or_reynolds - 1.5e6) / 1.5e6)

            # add physics parameters to graph
            nodes = np.concatenate([
                self.nodes,
                np.repeat(a=norm_aoa, repeats=self.nodes.shape[0])[:, np.newaxis],
                np.repeat(a=norm_mach_or_reynolds, repeats=self.nodes.shape[0])[:, np.newaxis],
                self.node_markers
            ], axis=-1).astype(np.float32)
            nodes = paddle.to_tensor(nodes)

            graph = pgl.Graph(num_nodes=nodes.shape[0],
                              edges=self.edges,
                              node_feat={"feature": nodes})

            graph.y = fields
            graph.aoa = paddle.to_tensor(aoa)
            graph.norm_aoa = paddle.to_tensor(norm_aoa)
            graph.mach_or_reynolds = paddle.to_tensor(mach_or_reynolds)
            graph.norm_mach_or_reynolds = paddle.to_tensor(norm_mach_or_reynolds)

            self.graphs.append(graph)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.graphs[idx]

    def preprocess(self, tensor_list, stack_output=True):
        # data_means, data_stds = self.normalization_factors
        data_max, data_min = self.normalization_factors
        normalized_tensors = []
        for i in range(len(tensor_list)):
            # tensor_list[i] = (tensor_list[i] - data_means[i]) / data_stds[i] / 10
            normalized = (tensor_list[i] - data_min[i]) / (data_max[i] - data_min[i]) * 2 - 1
            normalized_tensors.append(normalized)
        if stack_output:
            normalized_tensors = np.stack(normalized_tensors, axis=1)
        return normalized_tensors

    @staticmethod
    def get_params_from_name(filename):
        s = filename.rsplit('.', 1)[0].split('_')
        aoa = np.array(s[s.index('aoa') + 1])[np.newaxis].astype(np.float32)
        reynolds = s[s.index('re') + 1]
        reynolds = np.array(reynolds)[np.newaxis].astype(np.float32) if reynolds != 'None' else None
        mach = np.array(s[s.index('mach') + 1])[np.newaxis].astype(np.float32)
        return aoa, reynolds, mach


if __name__ == '__main__':

    train_data = MeshAirfoilDataset("data/NACA0012_interpolate", mode='train')
    val_data = MeshAirfoilDataset("data/NACA0012_interpolate", mode='test')

    val_loader = Dataloader(train_data, batch_size=2, shuffle=False, num_workers=1)

    for i, x in enumerate(val_loader):
        print(i)
        print(x)
