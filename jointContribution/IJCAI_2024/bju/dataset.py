import os
import random

import numpy as np
import paddle
from utils.utils import k_hop_subgraph
from utils.utils import radius_graph


def read_data(args, norm=True):
    with open(os.path.join(args.info_dir, "global_bounds.txt"), "r") as fp:
        min_bounds = fp.readline().split(" ")
        max_bounds = fp.readline().split(" ")
        min_in = np.array([float(a) for a in min_bounds])
        max_in = np.array([float(a) for a in max_bounds])
    with open(os.path.join(args.info_dir, "train_pressure_mean_std.txt"), "r") as fp:
        min_bounds = fp.readline().split(" ")
        max_bounds = fp.readline().split(" ")
        mean_out = np.array([float(a) for a in min_bounds])
        std_out = np.array([float(a) for a in max_bounds])
    coef_norm = min_in, max_in, mean_out, std_out
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir
    extra_data_dir = args.extra_data_dir
    train_samples = []
    test_samples = []
    train_files = os.listdir(train_data_dir)
    test_files = os.listdir(test_data_dir)
    for file in train_files:
        if file.startswith("press_"):
            path = os.path.join(train_data_dir, file)
            train_samples.append(path)
    if extra_data_dir is not None:
        extra_files = os.listdir(extra_data_dir)
        for file in extra_files:
            if file.startswith("press_"):
                path = os.path.join(extra_data_dir, file)
                train_samples.append(path)
    for file in test_files:
        if file.startswith("centroid_"):
            path = os.path.join(test_data_dir, file)
            test_samples.append(path)
    val_samples = train_samples[-50:]
    train_samples = train_samples[:-50]
    train_dataset = []
    val_dataset = []
    test_dataset = []
    for k, s in enumerate(train_samples):
        file_name_press = s
        file_name_point = s.replace("press", "centroid")
        if not (os.path.exists(file_name_press) or os.path.exists(file_name_point)):
            continue
        press = np.load(file_name_press)
        points_press = np.load(file_name_point)
        x = paddle.to_tensor(data=points_press)
        y = paddle.to_tensor(data=press)
        data = CustomData(x=x, y=y)
        if norm is True:
            data.x = ((data.x - min_in) / (max_in - min_in + 1e-08)).astype(
                dtype="float32"
            )
            data.y = ((data.y - mean_out) / (std_out + 1e-08)).astype(dtype="float32")
        train_dataset.append(data)
    for k, s in enumerate(val_samples):
        file_name_press = s
        file_name_point = s.replace("press", "centroid")
        if not (os.path.exists(file_name_press) or os.path.exists(file_name_point)):
            continue
        press = np.load(file_name_press)
        points_press = np.load(file_name_point)
        x = paddle.to_tensor(data=points_press)
        y = paddle.to_tensor(data=press)
        data = CustomData(x=x, y=y)
        if norm is True:
            data.x = ((data.x - min_in) / (max_in - min_in + 1e-08)).astype(
                dtype="float32"
            )
            data.y = ((data.y - mean_out) / (std_out + 1e-08)).astype(dtype="float32")
        val_dataset.append(data)
    for k, s in enumerate(test_samples):
        file_name_point = s
        points_press = np.load(file_name_point)
        x = paddle.to_tensor(data=points_press)
        data = CustomData(x=x)
        if norm is True:
            data.x = ((data.x - min_in) / (max_in - min_in + 1e-08)).astype(
                dtype="float32"
            )
        test_dataset.append(data)
    test_index = [
        int(os.path.basename(i).lstrip("centroid_").rstrip(".npy"))
        for i in test_samples
    ]
    return train_dataset, val_dataset, test_dataset, coef_norm, test_index


def get_induced_graph(data, idx, num_hops):
    subset, sub_edge_index, _, _ = k_hop_subgraph(
        node_idx=idx, num_hops=num_hops, edge_index=data.edge_index, relabel_nodes=True
    )
    return CustomData(x=data.x[subset], y=data.y[idx], edge_index=sub_edge_index)


def pc_normalize(pc):
    centroid = paddle.mean(pc, axis=0)
    pc = pc - centroid
    m = paddle.max(x=paddle.sqrt(x=paddle.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def get_shape(data, max_n_point=3682, normalize=True, use_height=False):
    if len(data.x) > max_n_point:
        surf_indices = np.array(random.sample(range(len(data.x)), max_n_point))
    shape_pc = data.x[surf_indices].clone()
    if normalize:
        shape_pc = pc_normalize(shape_pc)
    return shape_pc.astype(dtype="float32")


def create_edge_index_radius(data, r, max_neighbors=32):
    data.edge_index = radius_graph(
        x=data.pos, r=r, loop=True, max_num_neighbors=max_neighbors
    )
    return data


class CustomData:
    def __init__(self, **kwargs):
        self.edge_index = None
        for key, value in kwargs.items():
            setattr(self, key, value)


class GraphDataset(paddle.io.Dataset):
    def __init__(self, datalist, use_height=False, use_cfd_mesh=True, r=None):
        super().__init__()
        self.datalist = datalist
        self.use_height = use_height
        if not use_cfd_mesh:
            assert r is not None
            for i in range(len(self.datalist)):
                self.datalist[i] = create_edge_index_radius(self.datalist[i], r)

    def __len__(self):
        return len(self.datalist)

    def get(self, idx):
        data = self.datalist[idx]
        shape = get_shape(data, use_height=self.use_height)
        return self.datalist[idx], shape

    def __getitem__(self, idx):
        return self.get(idx)

    def collate_fn(self, batch):
        batch_data = [data for (data, _) in batch]
        batch_shape = paddle.stack([shape for (_, shape) in batch], axis=0)
        return batch_data, batch_shape


if __name__ == "__main__":
    root = "./data/mlcfd_data/training_data"
