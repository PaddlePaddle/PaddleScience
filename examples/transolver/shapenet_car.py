"""
Reference: https://github.com/thuml/Transolver
"""
from __future__ import annotations

import itertools
import os
import os.path as osp
import random
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import paddle
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

try:
    import vtk
except ModuleNotFoundError:
    pass


try:
    from paddle_geometric import nn as nng
except ModuleNotFoundError:
    pass
try:
    from paddle_geometric.data import Data
except ModuleNotFoundError:
    pass
try:
    from paddle_geometric.utils import k_hop_subgraph
except ModuleNotFoundError:
    pass

try:
    from vtk.util.numpy_support import vtk_to_numpy
except ModuleNotFoundError:
    pass


def load_unstructured_grid_data(file_name):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_name)
    reader.Update()
    output = reader.GetOutput()
    return output


def unstructured_grid_data_to_poly_data(unstructured_grid_data):
    filter = vtk.vtkDataSetSurfaceFilter()
    filter.SetInputData(unstructured_grid_data)
    filter.Update()
    poly_data = filter.GetOutput()
    return poly_data, filter


def get_sdf(target, boundary):
    nbrs = NearestNeighbors(n_neighbors=1).fit(boundary)
    dists, indices = nbrs.kneighbors(target)
    neis = np.array([boundary[i[0]] for i in indices])
    dirs = (target - neis) / (dists + 1e-08)
    return dists.reshape(-1), dirs


def get_normal(unstructured_grid_data):
    poly_data, surface_filter = unstructured_grid_data_to_poly_data(
        unstructured_grid_data
    )
    normal_filter = vtk.vtkPolyDataNormals()
    normal_filter.SetInputData(poly_data)
    normal_filter.SetAutoOrientNormals(1)
    normal_filter.SetConsistency(1)
    normal_filter.SetComputeCellNormals(1)
    normal_filter.SetComputePointNormals(0)
    normal_filter.Update()
    """
    normal_filter.SetComputeCellNormals(0)
    normal_filter.SetComputePointNormals(1)
    normal_filter.Update()
    #visualize_poly_data(poly_data, surface_filter, normal_filter)
    poly_data.GetPointData().SetNormals(normal_filter.GetOutput().GetPointData().GetNormals())
    p2c = vtk.vtkPointDataToCellData()
    p2c.ProcessAllArraysOn()
    p2c.SetInputData(poly_data)
    p2c.Update()
    unstructured_grid_data.GetCellData().SetNormals(p2c.GetOutput().GetCellData().GetNormals())
    #visualize_poly_data(poly_data, surface_filter, p2c)
    """
    unstructured_grid_data.GetCellData().SetNormals(
        normal_filter.GetOutput().GetCellData().GetNormals()
    )
    c2p = vtk.vtkCellDataToPointData()
    c2p.SetInputData(unstructured_grid_data)
    c2p.Update()
    unstructured_grid_data = c2p.GetOutput()
    normal = vtk_to_numpy(c2p.GetOutput().GetPointData().GetNormals()).astype(np.double)
    normal /= np.max(np.abs(normal), axis=1, keepdims=True) + 1e-08
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-08
    if np.isnan(normal).sum() > 0:
        print(np.isnan(normal).sum())
        print("recalculate")
        return get_normal(unstructured_grid_data)
    return normal


def visualize_poly_data(poly_data, surface_filter, normal_filter=None):
    if normal_filter is not None:
        mask = vtk.vtkMaskPoints()
        mask.SetInputData(normal_filter.GetOutput())
        mask.Update()
        arrow = vtk.vtkArrowSource()
        arrow.Update()
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(mask.GetOutput())
        glyph.SetSourceData(arrow.GetOutput())
        glyph.SetVectorModeToUseNormal()
        glyph.SetScaleFactor(0.1)
        glyph.Update()
        norm_mapper = vtk.vtkPolyDataMapper()
        norm_mapper.SetInputData(normal_filter.GetOutput())
        glyph_mapper = vtk.vtkPolyDataMapper()
        glyph_mapper.SetInputData(glyph.GetOutput())
        norm_actor = vtk.vtkActor()
        norm_actor.SetMapper(norm_mapper)
        glyph_actor = vtk.vtkActor()
        glyph_actor.SetMapper(glyph_mapper)
        glyph_actor.GetProperty().SetColor(1, 0, 0)
        norm_render = vtk.vtkRenderer()
        norm_render.AddActor(norm_actor)
        norm_render.SetBackground(0, 1, 0)
        glyph_render = vtk.vtkRenderer()
        glyph_render.AddActor(glyph_actor)
        glyph_render.AddActor(norm_actor)
        glyph_render.SetBackground(0, 0, 1)
    scalar_range = poly_data.GetScalarRange()
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(surface_filter.GetOutputPort())
    mapper.SetScalarRange(scalar_range)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)
    renderer_window = vtk.vtkRenderWindow()
    renderer_window.AddRenderer(renderer)
    if normal_filter is not None:
        renderer_window.AddRenderer(norm_render)
        renderer_window.AddRenderer(glyph_render)
    renderer_window.Render()
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderer_window)
    interactor.Initialize()
    interactor.Start()


def get_datalist(
    root, samples, norm=False, coef_norm=None, savedir=None, preprocessed=False
):
    dataset = []
    mean_in, mean_out = 0, 0
    std_in, std_out = 0, 0
    for k, s in enumerate(tqdm(samples)):
        save_path = osp.join(savedir, s)
        if preprocessed and savedir is not None:
            if not osp.exists(save_path):
                continue
            init = np.load(osp.join(save_path, "x.npy"))
            target = np.load(osp.join(save_path, "y.npy"))
            pos = np.load(osp.join(save_path, "pos.npy"))
            surf = np.load(osp.join(save_path, "surf.npy"))
            edge_index = np.load(osp.join(save_path, "edge_index.npy"))
        else:
            if savedir is not None and all(
                [
                    osp.exists(path)
                    for path in [
                        osp.join(save_path, "x.npy"),
                        osp.join(save_path, "y.npy"),
                        osp.join(save_path, "pos.npy"),
                        osp.join(save_path, "surf.npy"),
                        osp.join(save_path, "edge_index.npy"),
                    ]
                ]
            ):
                init = np.load(osp.join(save_path, "x.npy"))
                target = np.load(osp.join(save_path, "y.npy"))
                pos = np.load(osp.join(save_path, "pos.npy"))
                surf = np.load(osp.join(save_path, "surf.npy"))
                edge_index = np.load(osp.join(save_path, "edge_index.npy"))
            else:
                file_name_press = osp.join(root, osp.join(s, "quadpress_smpl.vtk"))
                file_name_velo = osp.join(root, osp.join(s, "hexvelo_smpl.vtk"))
                if not osp.exists(file_name_press) or not osp.exists(file_name_velo):
                    continue
                unstructured_grid_data_press = load_unstructured_grid_data(
                    file_name_press
                )
                unstructured_grid_data_velo = load_unstructured_grid_data(
                    file_name_velo
                )
                velo = vtk_to_numpy(
                    unstructured_grid_data_velo.GetPointData().GetVectors()
                )
                press = vtk_to_numpy(
                    unstructured_grid_data_press.GetPointData().GetScalars()
                )
                points_velo = vtk_to_numpy(
                    unstructured_grid_data_velo.GetPoints().GetData()
                )
                points_press = vtk_to_numpy(
                    unstructured_grid_data_press.GetPoints().GetData()
                )
                edges_press = get_edges(
                    unstructured_grid_data_press, points_press, cell_size=4
                )
                edges_velo = get_edges(
                    unstructured_grid_data_velo, points_velo, cell_size=8
                )
                sdf_velo, normal_velo = get_sdf(points_velo, points_press)
                sdf_press = np.zeros(points_press.shape[0])
                normal_press = get_normal(unstructured_grid_data_press)
                surface = {tuple(p) for p in points_press}
                exterior_indices = [
                    i for i, p in enumerate(points_velo) if tuple(p) not in surface
                ]
                velo_dict = {tuple(p): velo[i] for i, p in enumerate(points_velo)}
                pos_ext = points_velo[exterior_indices]
                pos_surf = points_press
                sdf_ext = sdf_velo[exterior_indices]
                sdf_surf = sdf_press
                normal_ext = normal_velo[exterior_indices]
                normal_surf = normal_press
                velo_ext = velo[exterior_indices]
                velo_surf = np.array(
                    [
                        (velo_dict[tuple(p)] if tuple(p) in velo_dict else np.zeros(3))
                        for p in pos_surf
                    ]
                )
                press_ext = np.zeros([len(exterior_indices), 1])
                press_surf = press
                init_ext = np.c_[pos_ext, sdf_ext, normal_ext]
                init_surf = np.c_[pos_surf, sdf_surf, normal_surf]
                target_ext = np.c_[velo_ext, press_ext]
                target_surf = np.c_[velo_surf, press_surf]
                surf = np.concatenate([np.zeros(len(pos_ext)), np.ones(len(pos_surf))])
                pos = np.concatenate([pos_ext, pos_surf])
                init = np.concatenate([init_ext, init_surf])
                target = np.concatenate([target_ext, target_surf])
                edge_index = get_edge_index(pos, edges_press, edges_velo)
                if savedir is not None:
                    save_path = osp.join(savedir, s)
                    if not osp.exists(save_path):
                        os.makedirs(save_path)
                    np.save(osp.join(save_path, "x.npy"), init)
                    np.save(osp.join(save_path, "y.npy"), target)
                    np.save(osp.join(save_path, "pos.npy"), pos)
                    np.save(osp.join(save_path, "surf.npy"), surf)
                    np.save(osp.join(save_path, "edge_index.npy"), edge_index)
        surf = paddle.tensor(surf)
        pos = paddle.tensor(pos)
        x = paddle.tensor(init)
        y = paddle.tensor(target)
        edge_index = paddle.tensor(edge_index)
        if norm and coef_norm is None:
            if k == 0:
                old_length = init.shape[0]
                mean_in = init.mean(axis=0)
                mean_out = target.mean(axis=0)
            else:
                new_length = old_length + init.shape[0]
                mean_in += (init.sum(axis=0) - init.shape[0] * mean_in) / new_length
                mean_out += (target.sum(axis=0) - init.shape[0] * mean_out) / new_length
                old_length = new_length
        data = Data(pos=pos, x=x, y=y, surf=surf.bool(), edge_index=edge_index)
        dataset.append(data)
    if norm and coef_norm is None:
        for k, data in enumerate(dataset):
            if k == 0:
                old_length = data.x.numpy().shape[0]
                std_in = ((data.x.numpy() - mean_in) ** 2).sum(axis=0) / old_length
                std_out = ((data.y.numpy() - mean_out) ** 2).sum(axis=0) / old_length
            else:
                new_length = old_length + data.x.numpy().shape[0]
                std_in += (
                    ((data.x.numpy() - mean_in) ** 2).sum(axis=0)
                    - data.x.numpy().shape[0] * std_in
                ) / new_length
                std_out += (
                    ((data.y.numpy() - mean_out) ** 2).sum(axis=0)
                    - data.x.numpy().shape[0] * std_out
                ) / new_length
                old_length = new_length
        std_in = np.sqrt(std_in)
        std_out = np.sqrt(std_out)
        for data in dataset:
            data.x = ((data.x - mean_in) / (std_in + 1e-08)).float()
            data.y = ((data.y - mean_out) / (std_out + 1e-08)).float()
        coef_norm = mean_in, std_in, mean_out, std_out
        dataset = dataset, coef_norm
    elif coef_norm is not None:
        for data in dataset:
            data.x = ((data.x - coef_norm[0]) / (coef_norm[1] + 1e-08)).float()
            data.y = ((data.y - coef_norm[2]) / (coef_norm[3] + 1e-08)).float()

    return dataset


def get_edges(unstructured_grid_data, points, cell_size=4):
    edge_indeces = set()
    cells = vtk_to_numpy(unstructured_grid_data.GetCells().GetData()).reshape(
        -1, cell_size + 1
    )
    for i in range(len(cells)):
        for j, k in itertools.product(range(1, cell_size + 1), repeat=2):
            edge_indeces.add((cells[i][j], cells[i][k]))
            edge_indeces.add((cells[i][k], cells[i][j]))
    edges = [[], []]
    for u, v in edge_indeces:
        edges[0].append(tuple(points[u]))
        edges[1].append(tuple(points[v]))
    return edges


def get_edge_index(pos, edges_press, edges_velo):
    indices = {tuple(pos[i]): i for i in range(len(pos))}
    edges = set()
    for i in range(len(edges_press[0])):
        edges.add((indices[edges_press[0][i]], indices[edges_press[1][i]]))
    for i in range(len(edges_velo[0])):
        edges.add((indices[edges_velo[0][i]], indices[edges_velo[1][i]]))
    edge_index = np.array(list(edges)).T
    return edge_index


def get_induced_graph(data, idx, num_hops):
    subset, sub_edge_index, _, _ = k_hop_subgraph(
        node_idx=idx, num_hops=num_hops, edge_index=data.edge_index, relabel_nodes=True
    )
    return Data(x=data.x[subset], y=data.y[idx], edge_index=sub_edge_index)


def pc_normalize(pc):
    centroid = paddle.mean(pc, axis=0)
    pc = pc - centroid
    m = paddle.compat.max(paddle.sqrt(paddle.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def get_shape(data, max_n_point=8192, normalize=True, use_height=False):
    surf_indices = paddle.where(data.surf)[0].tolist()
    if len(surf_indices) > max_n_point:
        surf_indices = np.array(random.sample(range(len(surf_indices)), max_n_point))
    shape_pc = data.pos[surf_indices].clone()
    if normalize:
        shape_pc = pc_normalize(shape_pc)
    if use_height:
        gravity_dim = 1
        height_array = (
            shape_pc[:, gravity_dim : gravity_dim + 1]
            - shape_pc[:, gravity_dim : gravity_dim + 1]._min()
        )
        shape_pc = paddle.cat((shape_pc, height_array), axis=1)
    return shape_pc


def create_edge_index_radius(
    data, r, max_neighbors=32
) -> "paddle.Tensor":  # noqa: F821
    data.edge_index = nng.radius_graph(
        x=data.pos, r=r, loop=True, max_num_neighbors=max_neighbors
    )
    return data


class ShapeNetCarDataset(paddle.io.Dataset):
    """
    Dataset class for ShapeNet car data.

    Args:
        input_keys (Tuple[str, ...]): Tuple of input key names.
        label_keys (Tuple[str, ...]): Tuple of label key names.
        datalist (List[Data]): List of Data objects containing the dataset.
        use_height (bool): Whether to use height information. Defaults to False.
        use_cfd_mesh (bool): Whether to use CFD mesh. Defaults to True.
        r (Optional[float]): Radius for radius graph if not using CFD mesh.
        training (bool): Whether in training mode. Defaults to True.
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        datalist: List[Data],
        use_height: bool = False,
        use_cfd_mesh: bool = True,
        r: Optional[float] = None,
        training: bool = True,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.datalist = datalist
        self.use_height = use_height
        self.training = training
        if not use_cfd_mesh:
            assert r is not None
            for i in range(len(self.datalist)):
                self.datalist[i] = create_edge_index_radius(self.datalist[i], r)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        sample = self.datalist[idx]
        shape = get_shape(sample, use_height=self.use_height)
        velo_press = sample.y

        inp = {
            "x": (sample.x),
            # "pos": (sample.pos),
        }
        lab = {
            "velo_vec": velo_press[..., 0:3],  # x,y,z
            "press": velo_press[..., 3:],  # p
            "surf": (sample.surf),
        }
        wei = {}
        # lab["sdf"] = inp["x"][:, 3:4]
        # lab["normal_vec"] = inp["x"][:, 4:]

        if not self.training:
            lab["shape"] = shape

        return inp, lab, wei


def get_samples(root: str) -> List[List[str]]:
    folds = [f"param{i}" for i in range(9)]
    samples: List[List[str]] = []
    for fold in folds:
        fold_samples: List[str] = []
        files = os.listdir(osp.join(root, fold))

        for file in files:
            path = osp.join(root, osp.join(fold, file))
            if osp.isdir(path):
                fold_samples.append(osp.join(fold, file))

        samples.append(fold_samples)

    return samples


def load_train_val_fold(
    data_dir: str, val_fold_id: int, save_dir: Optional[str], preprocessed: bool
) -> Tuple[paddle.io.Dataset, paddle.io.Dataset, np.ndarray]:
    samples = get_samples(data_dir)
    trainlst = []
    for i in range(len(samples)):
        if i == val_fold_id:
            continue
        trainlst += samples[i]

    vallist = samples[val_fold_id] if 0 <= val_fold_id < len(samples) else None

    if preprocessed:
        print("use preprocessed data")
    print("loading data")
    # trainlst = trainlst[:10]
    # vallist = vallist[:10]
    train_dataset, coef_norm = get_datalist(
        data_dir,
        trainlst,
        norm=True,
        savedir=save_dir,
        preprocessed=preprocessed,
    )
    val_dataset = get_datalist(
        data_dir,
        vallist,
        coef_norm=coef_norm,
        savedir=save_dir,
        preprocessed=preprocessed,
    )
    print("load data finish")
    return train_dataset, val_dataset, coef_norm


def load_train_val_fold_file(
    data_dir: str, val_fold_id: int, save_dir: Optional[str], preprocessed: bool
):
    samples = get_samples(data_dir)
    trainlst = []
    for i in range(len(samples)):
        if i == val_fold_id:
            continue
        trainlst += samples[i]

    vallist = samples[val_fold_id] if 0 <= val_fold_id < len(samples) else None

    if preprocessed:
        print("use preprocessed data")

    print("loading data")
    train_dataset, coef_norm = get_datalist(
        data_dir,
        trainlst,
        norm=True,
        savedir=save_dir,
        preprocessed=preprocessed,
    )
    val_dataset = get_datalist(
        data_dir,
        vallist,
        coef_norm=coef_norm,
        savedir=save_dir,
        preprocessed=preprocessed,
    )
    print("load data finish")

    return train_dataset, val_dataset, coef_norm, vallist
