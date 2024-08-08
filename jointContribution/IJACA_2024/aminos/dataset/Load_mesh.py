import os
import re

import h5py
import numpy as np
import paddle
import pgl
from utils.knn import k_hop_subgraph


def trans_edge_index(edge_index):
    assert isinstance(edge_index, paddle.Tensor)
    edge_index_np = edge_index.numpy()
    edges = list(zip(edge_index_np[0], edge_index_np[1]))
    return edges


class DOF_Dataset(paddle.io.Dataset):
    def __init__(
        self, params=None, is_training=True, split="train", device=None, is_norm=False
    ):
        self.params = params
        self.is_training = is_training
        self.device = device
        self.epoch = 0
        self.pool = h5py.File(self.params.dataset_dir + f"/DOF_{split}.h5", "r")
        self.split = split
        self.is_norm = is_norm
        self.key_list = list(self.pool.keys())
        self.key_list = np.sort([int(key) for key in self.key_list])
        self.key_list = [str(key) for key in self.key_list]
        self.mean_std = paddle.to_tensor(
            data=np.loadtxt(self.params.dataset_dir + "dof_mean_std.txt")
        ).to("float32")

    def __getitem__(self, index):
        dof = paddle.to_tensor(data=self.pool[self.key_list[index]][:]).to("float32")
        if self.is_norm:
            dof = dof.transpose(perm=[1, 2, 0])
            dof -= self.mean_std[0]
            dof /= self.mean_std[1]
            dof = dof.transpose(perm=[1, 2, 0])
        return dof

    def len(self):
        return len(self.key_list)


class CarDataset(paddle.io.Dataset):
    def __init__(self, path, mesh_indices):
        super().__init__()
        self.mesh_indices = mesh_indices
        self.dataset_dir = path
        self.file_handle = h5py.File(self.dataset_dir, "r")

    def __getitem__(self, index):
        file_idx = str(int(self.mesh_indices[index]))
        handle = self.file_handle[file_idx]
        handle_dict = dict(handle)
        for k, v in handle_dict.items():
            handle_dict[k] = v[:]
        return handle_dict, file_idx

    def __len__(self):
        return len(self.mesh_indices)


class CarDataset4UNet(CarDataset):
    def __init__(self, path, mesh_indices, gt_exist=True):
        super().__init__(path, mesh_indices)
        self.current_idx = None
        self.gt_exist = gt_exist

    def __getitem__(self, index):
        data, file_idx = super().__getitem__(index)
        self.current_idx = file_idx
        rdata = {}
        rdata["node|pos"] = paddle.to_tensor(data=data["node|pos"]).to("float32")
        if self.gt_exist:
            rdata["node|pressure"] = paddle.to_tensor(data=data["node|pressure"]).to(
                "float32"
            )
        rdata["voxel|sdf"] = (
            paddle.to_tensor(data=data["voxel|sdf"])
            .reshape(1, *tuple(data["voxel|grid"].shape)[:-1])
            .to("float32")
        )
        rdata["node|unit_norm_v"] = paddle.to_tensor(data=data["node|unit_norm_v"]).to(
            "float32"
        )
        return rdata

    def get_cur_file_idx(self):
        return self.current_idx


class CarDatasetGraph(CarDataset4UNet):
    def __init__(self, path, mesh_indices, gt_exist=True):
        super().__init__(path, mesh_indices, gt_exist)

    def __getitem__(self, index):
        raw_data, _ = super().__getitem__(index)
        data = super().__getitem__(index)
        num_nodes = data["node|unit_norm_v"].shape[0]
        edges = trans_edge_index(raw_data["face|face_node"])
        pgl.graph.Graph(
            num_nodes=num_nodes,
            edges=edges,
            pos=data["node|pos"],
            pressure=data["node|pressure"],
        )


def GetCarDatasetInfoList(params, path, split: list):
    dataset_dir = os.path.join(path, "train.h5")
    pressure_min_std = (np.loadtxt(os.path.join(path, "train_pressure_min_std.txt")),)
    bounds = (np.loadtxt(os.path.join(path, "watertight_global_bounds.txt")),)
    all_mesh_indices = np.loadtxt(os.path.join(path, "watertight_meshes.txt")).reshape(
        -1
    )
    splited_mesh_indices = [
        all_mesh_indices[start:end] for start, end in zip(split[:-1], split[1:])
    ]
    return dataset_dir, *pressure_min_std, *bounds, splited_mesh_indices


class CFDdatasetmap(paddle.io.Dataset):
    def __init__(
        self, params, path, split="train", dataset_type="h5", is_training=False
    ):
        super().__init__()
        self.path = path
        self.split = split
        self.dataset_dir = path
        self.params = params
        self.is_training = is_training
        if dataset_type == "h5":
            self.file_handle = h5py.File(self.dataset_dir + f"/{split}.h5", "r")
        else:
            raise ValueError("invalid data format")

    def __getitem__(self, index):
        trajectory_handle = self.file_handle[str(index)]
        trajectory = {}
        for key in trajectory_handle.keys():
            trajectory[key] = paddle.to_tensor(data=trajectory_handle[key][:])
        return trajectory

    def __len__(self):
        return len(self.file_handle)


def sort_key_list(in_list: list):
    a_list = []
    b_list = []
    for k in in_list:
        if k.startswith("A"):
            a_list.append(k)
        elif k.startswith("B"):
            b_list.append(k)
    sorted_a_list = sorted(a_list, key=lambda s: int(re.search("_(\\d+)", s).group(1)))
    sorted_b_list = sorted(b_list, key=lambda s: int(re.search("_(\\d+)", s).group(1)))
    rt_list = sorted_a_list + sorted_b_list
    return rt_list


class Data_Pool:
    def __init__(self, params=None, is_training=True, split="train", device=None):
        self.params = params
        self.is_training = is_training
        self.device = device
        self.epoch = 0
        self.load_mesh_to_cpu(split=split, dataset_dir=params.dataset_dir)

    def load_mesh_to_cpu(self, split="train", dataset_dir=None):
        self.valid_pool = []
        if dataset_dir is not None:
            self.pool = h5py.File(dataset_dir + f"/{split}.h5", "r")
        else:
            self.pool = h5py.File(self.params.dataset_dir + f"/{split}.h5", "r")
        self.key_list = list(self.pool.keys())
        self.key_list = sort_key_list(self.key_list)
        return self.params.dataset_dir

    @staticmethod
    def datapreprocessing(graph_cell, is_training=False):
        def randbool(*size, device="cuda"):
            """Returns 50% channce of True of False"""
            return paddle.randint(low=2, high=size) == paddle.randint(low=2, high=size)

        graph_cell.ball_edge_index = None
        cell_attr = paddle.concat(x=(graph_cell.x, graph_cell.pos), axis=-1)
        senders, receivers = graph_cell.edge_index
        if is_training:
            random_mask = randbool(
                1, tuple(senders.shape)[0], device=senders.place
            ).repeat(2, 1)
            random_direction_edge = paddle.where(
                condition=random_mask,
                x=paddle.stack(x=(senders, receivers), axis=0),
                y=paddle.stack(x=(receivers, senders), axis=0),
            )
        else:
            random_direction_edge = paddle.stack(x=(senders, receivers), axis=0)
        releative_node_attr = (
            cell_attr[random_direction_edge[0]] - cell_attr[random_direction_edge[1]]
        )
        graph_cell.edge_index = random_direction_edge
        graph_cell.edge_attr = releative_node_attr
        return graph_cell


class CustomGraphData(pgl.graph.Graph):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.keys = [
            "norm_y",
            "pos",
            "voxel",
            "origin_id",
            "press_std",
            "edge_attr",
            "press_mean",
            "edge_index",
            "batch",
            "graph_index",
            "y",
            "x",
            "query",
            "ptr",
            "ao",
        ]
        edges = trans_edge_index(self.edge_index)
        super().__init__(num_nodes=self.x.shape[0], edges=edges, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        offset_rules = {
            "edge_index": self.num_nodes,
            "face": self.num_nodes,
            "cells_node": self.num_nodes,
            "face_node": self.num_nodes,
            "cells_face": self.num_nodes,
            "neighbour_cell": self.num_nodes,
            "face_node_x": self.num_nodes,
            "pos": 0,
            "A_node_to_node": 0,
            "A_node_to_node_x": 0,
            "B_node_to_node": 0,
            "B_node_to_node_x": 0,
            "cell_area": 0,
            "node_type": 0,
            "graph_index": 0,
            "pde_theta": 0,
            "neural_network_output_mask": 0,
            "uvp_dim": 0,
            "dt_graph": 0,
            "x": 0,
            "y": 0,
            "m_ids": 0,
            "m_gs": 0,
            "case_global_index": 0,
        }
        return offset_rules.get(key, super().__inc__(key, value, *args, **kwargs))

    def __cat_dim__(self, key, value, *args, **kwargs):
        cat_dim_rules = {
            "x": 0,
            "pos": 0,
            "y": 0,
            "norm_y": 0,
            "query": 0,
            "edge_index": 1,
            "voxel": 0,
            "graph_index": 0,
        }
        return cat_dim_rules.get(key, super().__cat_dim__(key, value, *args, **kwargs))


class GraphCellDataset(paddle.io.Dataset):
    def __init__(
        self,
        base_dataset,
        len_ds=None,
        indices=None,
        params=None,
        subsampling=False,
        sample_ratio=0.2,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self._len = len_ds
        self.idx_indices = None
        if indices is not None:
            self._len = len(indices)
            self.idx_indices = indices
        self.params = params
        self.subsampling = subsampling
        self.sample_ratio = sample_ratio
        self.k_hop = self.params.sample_khop

    @property
    def pool(self):
        return self.base_dataset.pool

    @property
    def key_list(self):
        return self.base_dataset.key_list

    def len(self):
        return self._len

    def __len__(self):
        return self._len

    def load_A_data(self, idx):
        """<KeysViewHDF5 ['cells_face', 'cells_index',
        'cells_node', 'cell|cells_area', 'cell|centroid',
        'face|face_center_pos', 'face|face_length',
        'face|face_node', 'face|neighbour_cell',
        'node|pos', 'node|pressure',
        'node|unit_norm_v', 'voxel|grid',
        'voxel|sdf']>"""
        minibatch_data = self.pool[self.key_list[idx]]
        mesh_pos = paddle.to_tensor(
            data=minibatch_data["node|pos"][:], dtype=paddle.get_default_dtype()
        )
        unit_norm_v = paddle.to_tensor(
            data=minibatch_data["node|unit_norm_v"][:], dtype=paddle.get_default_dtype()
        )
        face_node = paddle.to_tensor(
            data=minibatch_data["face|face_node"][:], dtype=paddle.int64
        )
        ao = paddle.to_tensor(
            data=minibatch_data["node|ao"][:], dtype=paddle.get_default_dtype()
        )
        voxel = paddle.to_tensor(
            data=minibatch_data["voxel|sdf"][:], dtype=paddle.get_default_dtype()
        ).reshape(1, 1, *tuple(minibatch_data["voxel|grid"][:].shape)[:-1])
        voxel = (voxel - minibatch_data["voxel_mean_std"][0]) / minibatch_data[
            "voxel_mean_std"
        ][1]
        bounds = minibatch_data["bounds"]
        mid = (bounds[0] + bounds[1]) / 2
        scale = (bounds[1] - bounds[0]) / 2
        canonical_query = (mesh_pos - mid) / scale
        canonical_query = canonical_query.astype("float32")
        y = paddle.to_tensor(
            data=minibatch_data["node|pressure"][:], dtype=paddle.get_default_dtype()
        )
        norm_y = (y - minibatch_data["pressure_mean_std"][0]) / minibatch_data[
            "pressure_mean_std"
        ][1]
        graph_node = CustomGraphData(
            x=unit_norm_v,
            edge_index=face_node,
            pos=mesh_pos,
            y=y,
            norm_y=norm_y,
            ao=ao,
            voxel=voxel,
            query=canonical_query,
            graph_index=paddle.to_tensor(data=[idx], dtype="int64"),
            origin_id=paddle.to_tensor(
                data=[ord(char) for char in self.key_list[idx]], dtype="int64"
            ),
            press_mean=paddle.to_tensor(data=minibatch_data["pressure_mean_std"][0]),
            press_std=paddle.to_tensor(data=minibatch_data["pressure_mean_std"][1]),
        )
        return graph_node

    def load_B_data(self, idx):
        """<KeysViewHDF5 ['cells_face', 'cells_index',
        'cells_node', 'cell|cells_area', 'cell|centroid',
        'face|face_center_pos', 'face|face_length',
        'face|face_node', 'face|neighbour_cell',
        'node|pos', 'node|pressure',
        'node|unit_norm_v', 'voxel|grid',
        'voxel|sdf']>"""
        minibatch_data = self.pool[self.key_list[idx]]
        mesh_pos = paddle.to_tensor(
            data=minibatch_data["cell|centroid"][:], dtype=paddle.get_default_dtype()
        )
        normals = paddle.to_tensor(
            data=minibatch_data["cell|unit_norm_v"][:], dtype=paddle.get_default_dtype()
        )
        edge_index = paddle.to_tensor(
            data=minibatch_data["face|neighbour_cell"][:], dtype=paddle.int64
        )
        y = paddle.to_tensor(
            data=minibatch_data["cell|pressure"][:], dtype=paddle.get_default_dtype()
        )
        norm_y = (y - minibatch_data["pressure_mean_std"][0]) / minibatch_data[
            "pressure_mean_std"
        ][1]
        voxel = paddle.to_tensor(
            data=minibatch_data["voxel|sdf"][:], dtype=paddle.get_default_dtype()
        ).reshape(1, 1, *tuple(minibatch_data["voxel|grid"][:].shape)[:-1])
        voxel = (voxel - minibatch_data["voxel_mean_std"][0]) / minibatch_data[
            "voxel_mean_std"
        ][1]
        bounds = minibatch_data["bounds"]
        mid = (bounds[0] + bounds[1]) / 2
        scale = (bounds[1] - bounds[0]) / 2
        canonical_query = (mesh_pos - mid) / scale
        canonical_query = canonical_query.astype("float32")
        ao = paddle.zeros_like(x=y)
        if self.subsampling:
            sampled_nodes = paddle.randint(
                low=0, high=tuple(normals.shape)[0], shape=[self.params.num_samples]
            )
            subgraph_nodes, subgraph_edge_index, _, _ = k_hop_subgraph(
                sampled_nodes, self.k_hop, edge_index, relabel_nodes=True
            )
            normals = normals[subgraph_nodes]
            mesh_pos = mesh_pos[subgraph_nodes]
            y = y[subgraph_nodes]
            norm_y = norm_y[subgraph_nodes]
            ao = ao[subgraph_nodes]
            canonical_query = canonical_query[subgraph_nodes]
            edge_index = subgraph_edge_index
        graph_cell = CustomGraphData(
            x=normals,
            edge_index=edge_index,
            pos=mesh_pos,
            y=y,
            norm_y=norm_y,
            ao=ao,
            query=canonical_query,
            voxel=voxel,
            graph_index=paddle.to_tensor(data=[idx], dtype="int64"),
            origin_id=paddle.to_tensor(
                data=[ord(char) for char in self.key_list[idx]], dtype="int64"
            ),
            press_mean=paddle.to_tensor(data=minibatch_data["pressure_mean_std"][0]),
            press_std=paddle.to_tensor(data=minibatch_data["pressure_mean_std"][1]),
        )
        return graph_cell

    def get(self, idx):
        """<KeysViewHDF5 ['cells_face', 'cells_index',
        'cells_node', 'cell|cells_area', 'cell|centroid',
        'face|face_center_pos', 'face|face_length',
        'face|face_node', 'face|neighbour_cell',
        'node|pos', 'node|pressure',
        'node|unit_norm_v', 'voxel|grid',
        'voxel|sdf']>"""
        if self.idx_indices is not None:
            idx = self.idx_indices[idx]
        if self.key_list[idx].startswith("A"):
            graph_cell = self.load_A_data(idx)
        elif self.key_list[idx].startswith("B"):
            graph_cell = self.load_B_data(idx)
        else:
            minibatch_data = self.key_list[self.key_list[idx]]
            if tuple(minibatch_data["cell|centroid"].shape)[0] < 10000:
                graph_cell = self.load_A_data(idx)
            else:
                graph_cell = self.load_B_data(idx)
        return graph_cell

    def __getitem__(self, idx):
        return self.get(idx)


class CustomGraphDataLoader(paddle.io.DataLoader):
    def __init__(
        self, dataset, batch_size=1, shuffle=False, collate_fn=None, num_workers=0
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        self.dataset = dataset
        self.batch_size = batch_size
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration

        batch_data = [
            self.dataset[i]
            for i in range(
                self.index, min(self.index + self.batch_size, len(self.dataset))
            )
        ]
        self.index += self.batch_size

        return batch_data


class DatasetFactory:
    def __init__(self, params=None, device=None, split="test"):
        self.params = params
        self.train_dataset = Data_Pool(
            params=params, is_training=True, split=split, device=device
        )
        self.test_dataset = Data_Pool(
            params=params, is_training=False, split=split, device=device
        )

    def create_trainset(
        self,
        batch_size=100,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        indices=None,
        subsampling=True,
        ratio=0.2,
    ):
        """training set"""
        graph_cell_dataset = GraphCellDataset(
            base_dataset=self.train_dataset,
            len_ds=len(self.train_dataset.pool),
            indices=indices,
            params=self.params,
            subsampling=subsampling,
            sample_ratio=ratio,
        )
        loader = paddle.io.DataLoader(
            dataset=graph_cell_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )
        """ training set """
        return self.train_dataset, loader

    def create_testset(
        self,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        valid_num=10,
        subsampling=True,
        indices=None,
    ):
        """test set"""
        if indices is not None:
            valid_num = len(indices)
        graph_cell_dataset = GraphCellDataset(
            base_dataset=self.test_dataset,
            len_ds=valid_num,
            params=self.params,
            subsampling=subsampling,
            indices=indices,
        )
        loader = CustomGraphDataLoader(
            dataset=graph_cell_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )
        """ test set """
        return self.test_dataset, loader
