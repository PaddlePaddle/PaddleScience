import itertools
import os
import random
from typing import List
from typing import Union

import numpy as np
import paddle
import vtk
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from vtk.util.numpy_support import vtk_to_numpy


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


def get_scalar_data(unstructured_grid, scalar_name):
    point_data = unstructured_grid.GetPointData()
    if point_data:
        scalar_array = point_data.GetArray(scalar_name)
        if scalar_array:
            return vtk_to_numpy(scalar_array)
    return None


def bget_datalist(
    root, samples, norm=False, coef_norm=None, savedir=None, preprocessed=False
):
    dataset = []
    mean_in, mean_out = 0, 0
    std_in, std_out = 0, 0
    for k, s in enumerate(tqdm(samples, desc="Processing samples")):
        if preprocessed and savedir is not None:
            save_path = os.path.join(savedir, s)
            if not os.path.exists(save_path):
                continue
            init = np.load(os.path.join(save_path, "x.npy"))
            target = np.load(os.path.join(save_path, "y.npy"))
            pos = np.load(os.path.join(save_path, "pos.npy"))
            surf = np.load(os.path.join(save_path, "surf.npy"))
            area = np.load(os.path.join(save_path, "area.npy"))
        else:
            file_name_press = os.path.join(root, s)
            if not os.path.exists(file_name_press):
                continue
            unstructured_grid_data_press = load_unstructured_grid_data(file_name_press)
            scalar_names = ["Pressure", "point_scalars"]
            for scalar_name in scalar_names:
                press = get_scalar_data(unstructured_grid_data_press, scalar_name)
                if press is not None:
                    break
            points_press = vtk_to_numpy(
                unstructured_grid_data_press.GetPoints().GetData()
            )
            sdf_press = np.zeros(tuple(points_press.shape)[0])
            pos_surf = points_press
            sdf_surf = sdf_press
            press_surf = press
            mesh_number = s[-8:-4]
            area_file_name = os.path.join("data_track_B", f"area_{mesh_number}.npy")
            if os.path.exists(area_file_name):
                area = np.load(area_file_name)
            else:
                area = np.zeros(len(pos_surf))
            info = np.full((len(pos_surf), 1), 30.0)
            init_surf = np.c_[pos_surf, sdf_surf, area, info]
            target_surf = np.c_[np.zeros((len(pos_surf), 3)), press_surf]
            surf = np.ones(len(pos_surf))
            pos = pos_surf
            init = init_surf
            target = target_surf
            if savedir is not None:
                save_path = os.path.join(savedir, s)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                np.save(os.path.join(save_path, "x.npy"), init)
                np.save(os.path.join(save_path, "y.npy"), target)
                np.save(os.path.join(save_path, "pos.npy"), pos)
                np.save(os.path.join(save_path, "surf.npy"), surf)
                np.save(os.path.join(save_path, "area.npy"), area)
        surf = paddle.to_tensor(data=surf)
        pos = paddle.to_tensor(data=pos)
        x = paddle.to_tensor(data=init)
        y = paddle.to_tensor(data=target)
        if norm and coef_norm is None:
            if k == 0:
                old_length = tuple(init.shape)[0]
                mean_in = init.mean(axis=0)
                mean_out = target.mean(axis=0)
            else:
                new_length = old_length + tuple(init.shape)[0]
                mean_in += (
                    init.sum(axis=0) - tuple(init.shape)[0] * mean_in
                ) / new_length
                mean_out += (
                    target.sum(axis=0) - tuple(init.shape)[0] * mean_out
                ) / new_length
                old_length = new_length
        data = CustomData(pos=pos, x=x, y=y, surf=surf.astype(dtype="bool"))
        dataset.append(data)
    if norm and coef_norm is None:
        for k, data in enumerate(dataset):
            if k == 0:
                old_length = tuple(data.x.numpy().shape)[0]
                std_in = ((data.x.numpy() - mean_in) ** 2).sum(axis=0) / old_length
                std_out = ((data.y.numpy() - mean_out) ** 2).sum(axis=0) / old_length
            else:
                new_length = old_length + tuple(data.x.numpy().shape)[0]
                std_in += (
                    ((data.x.numpy() - mean_in) ** 2).sum(axis=0)
                    - tuple(data.x.numpy().shape)[0] * std_in
                ) / new_length
                std_out += (
                    ((data.y.numpy() - mean_out) ** 2).sum(axis=0)
                    - tuple(data.x.numpy().shape)[0] * std_out
                ) / new_length
                old_length = new_length
        std_in = np.sqrt(std_in)
        std_out = np.sqrt(std_out)
        for data in dataset:
            data.x = ((data.x - mean_in) / (std_in + 1e-08)).astype(dtype="float32")
            data.y = ((data.y - mean_out) / (std_out + 1e-08)).astype(dtype="float32")
        coef_norm = mean_in, std_in, mean_out, std_out
        dataset = dataset, coef_norm
    elif coef_norm is not None:
        for data in dataset:
            data.x = ((data.x - coef_norm[0]) / (coef_norm[1] + 1e-08)).astype(
                dtype="float32"
            )
            data.y = ((data.y - coef_norm[2]) / (coef_norm[3] + 1e-08)).astype(
                dtype="float32"
            )
    return dataset


def bget_datalist_for_prediction(
    root, samples, norm=False, coef_norm=None, savedir=None, preprocessed=False
):
    dataset = []
    mean_in, std_in = 0, 0
    for k, s in enumerate(tqdm(samples, desc="Processing samples")):
        if preprocessed and savedir is not None:
            save_path = os.path.join(savedir, s)
            if not os.path.exists(save_path):
                continue
            init = np.load(os.path.join(save_path, "x.npy"))
            pos = np.load(os.path.join(save_path, "pos.npy"))
            surf = np.load(os.path.join(save_path, "surf.npy"))
            area = np.load(os.path.join(save_path, "area.npy"))
        else:
            file_name = os.path.join(root, s)
            if not os.path.exists(file_name):
                continue
            unstructured_grid_data = load_unstructured_grid_data(file_name)
            points = vtk_to_numpy(unstructured_grid_data.GetPoints().GetData())
            sdf = np.zeros(tuple(points.shape)[0])
            pos_surf = points
            sdf_surf = sdf
            mesh_number = int(s.split("_")[-1].split(".")[0])
            area_file_name = os.path.join(
                "../data/IJCAI_Car/track_B", f"area_{mesh_number}.npy"
            )
            if os.path.exists(area_file_name):
                area = np.load(area_file_name)
            else:
                area = np.zeros(len(pos_surf))
            info = np.full((len(pos_surf), 1), 30.0)
            init_surf = np.c_[pos_surf, sdf_surf, area, info]
            surf = np.ones(len(pos_surf))
            pos = pos_surf
            init = init_surf
            if savedir is not None:
                save_path = os.path.join(savedir, s)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                np.save(os.path.join(save_path, "x.npy"), init)
                np.save(os.path.join(save_path, "pos.npy"), pos)
                np.save(os.path.join(save_path, "surf.npy"), surf)
                np.save(os.path.join(save_path, "area.npy"), area)
        surf = paddle.to_tensor(data=surf)
        pos = paddle.to_tensor(data=pos)
        x = paddle.to_tensor(data=init)
        y = paddle.zeros(shape=(tuple(x.shape)[0], 4))
        if norm and coef_norm is None:
            if k == 0:
                old_length = tuple(init.shape)[0]
                mean_in = init.mean(axis=0)
            else:
                new_length = old_length + tuple(init.shape)[0]
                mean_in += (
                    init.sum(axis=0) - tuple(init.shape)[0] * mean_in
                ) / new_length
                old_length = new_length
        data = CustomData(pos=pos, x=x, y=y, surf=surf.astype(dtype="bool"))
        dataset.append(data)
    if norm and coef_norm is None:
        for k, data in enumerate(dataset):
            if k == 0:
                old_length = tuple(data.x.numpy().shape)[0]
                std_in = ((data.x.numpy() - mean_in) ** 2).sum(axis=0) / old_length
            else:
                new_length = old_length + tuple(data.x.numpy().shape)[0]
                std_in += (
                    ((data.x.numpy() - mean_in) ** 2).sum(axis=0)
                    - tuple(data.x.numpy().shape)[0] * std_in
                ) / new_length
                old_length = new_length
        std_in = np.sqrt(std_in)
        for data in dataset:
            data.x = ((data.x - mean_in) / (std_in + 1e-08)).astype(dtype="float32")
        coef_norm = mean_in, std_in
    elif coef_norm is not None:
        for data in dataset:
            data.x = ((data.x - coef_norm[0]) / (coef_norm[1] + 1e-08)).astype(
                dtype="float32"
            )
    return dataset


def bget_data_for_prediction(file_name, norm=False, coef_norm=None):
    if not os.path.exists(file_name):
        return
    unstructured_grid_data = load_unstructured_grid_data(file_name)
    points = vtk_to_numpy(unstructured_grid_data.GetPoints().GetData())
    sdf = np.zeros(tuple(points.shape)[0])
    pos_surf = points
    sdf_surf = sdf
    mesh_number = int(file_name.split("_")[-1].split(".")[0])
    area_file_name = os.path.join(
        "../data/IJCAI_Car/track_B", f"area_{mesh_number}.npy"
    )
    if os.path.exists(area_file_name):
        area = np.load(area_file_name)
    else:
        area = np.zeros(len(pos_surf))
    info = np.full((len(pos_surf), 1), 30.0)
    init_surf = np.c_[pos_surf, sdf_surf, area, info]
    surf = np.ones(len(pos_surf))
    pos = pos_surf
    init = init_surf

    surf = paddle.to_tensor(data=surf)
    pos = paddle.to_tensor(data=pos)
    x = paddle.to_tensor(data=init)
    y = paddle.zeros(shape=(tuple(x.shape)[0], 4))

    data = CustomData(pos=pos, x=x, y=y, surf=surf.astype(dtype="bool"))

    if coef_norm is not None:
        data.x = ((data.x - coef_norm[0]) / (coef_norm[1] + 1e-08)).astype(
            dtype="float32"
        )
    return data


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
    return CustomData(x=data.x[subset], y=data.y[idx], edge_index=sub_edge_index)


def pc_normalize(pc):
    centroid = paddle.mean(pc, axis=0)
    pc = pc - centroid
    m = paddle.max(x=paddle.sqrt(x=paddle.sum(pc**2, axis=1)))
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
            - shape_pc[:, gravity_dim : gravity_dim + 1].min()
        )
        shape_pc = paddle.cat((shape_pc, height_array), axis=1)
    return shape_pc


def create_edge_index_radius(data, r, max_neighbors=32):
    if isinstance(data, list):
        print("Error: 'data' is a list, expected 'CustomData' object.")
        print("CustomData content:", data)
        return None
    data.edge_index = radius_graph(
        x=data.pos, r=r, loop=True, max_num_neighbors=max_neighbors
    )
    return data


class GraphDataset(paddle.io.Dataset):
    def __init__(
        self,
        datalist,
        use_height=False,
        use_cfd_mesh=True,
        r=None,
        root=None,
        norm=False,
        coef_norm=None,
    ):
        super().__init__()
        self.datalist = datalist
        self.use_height = use_height
        self.use_cfd_mesh = use_cfd_mesh
        self.r = r
        self.root = root
        self.norm = norm
        self.coef_norm = coef_norm

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        file_name = os.path.join(self.root, self.datalist[idx])
        data = bget_data_for_prediction(file_name, self.norm, self.coef_norm)
        if not self.use_cfd_mesh:
            data = create_edge_index_radius(data, self.r)
        shape = get_shape(data, use_height=self.use_height)
        return data, shape

    def collate_fn(self, batch):
        batch_data = [data for (data, _) in batch]
        batch_shape = paddle.stack([shape for (_, shape) in batch], axis=0)
        if len(batch_data) == 1:
            return batch_data[0], batch_shape
        return batch_data, batch_shape


def get_samples(root):
    samples = []
    files = os.listdir(root)
    for file in files:
        if file.endswith(".vtk"):
            samples.append(file)
    return samples


def B_load_train_val_fold(args, preprocessed):
    samples = get_samples(args.data_dir)
    np.random.shuffle(samples)
    trainlst = samples[: args.train_split]
    vallst = samples[args.train_split : args.val_split]
    if preprocessed:
        print("use preprocessed data")
    print("loading data")
    train_dataset, coef_norm = bget_datalist(
        args.data_dir,
        trainlst,
        norm=True,
        savedir=args.save_dir,
        preprocessed=preprocessed,
    )
    val_dataset = bget_datalist(
        args.data_dir,
        vallst,
        coef_norm=coef_norm,
        savedir=args.save_dir,
        preprocessed=preprocessed,
    )
    print("load data finish")
    return train_dataset, val_dataset, coef_norm


def Bload_train_val_fold_file(args, preprocessed, coef_norm):
    samples = get_samples(args.test_data_dir)
    np.random.shuffle(samples)
    vallst = samples[:50]
    if preprocessed:
        print("use preprocessed data")
    print("loading data")
    val_dataset = bget_datalist_for_prediction(
        args.test_data_dir,
        vallst,
        norm=True,
        savedir=args.save_dir,
        preprocessed=preprocessed,
        coef_norm=coef_norm,
    )
    print("load data finish")
    return val_dataset, vallst


def radius_graph(x, r, batch=None, loop=False, max_num_neighbors=32):
    num_nodes = x.shape[0]
    if batch is None:
        batch = paddle.zeros(shape=[num_nodes], dtype=paddle.int64)

    dist_matrix = paddle.norm(x.unsqueeze(1) - x.unsqueeze(0), axis=-1, p=2)

    adj_matrix = dist_matrix < r

    if not loop:
        adj_matrix = adj_matrix * (1 - paddle.eye(num_nodes, dtype=paddle.bool))

    mask = batch.unsqueeze(1) == batch.unsqueeze(0)
    adj_matrix = adj_matrix * mask

    degree = adj_matrix.sum(axis=-1)
    if max_num_neighbors < degree.max():
        idx = degree.argsort(descending=True)
        idx = idx[:max_num_neighbors]
        adj_matrix = adj_matrix[:, idx]

    return adj_matrix


def k_hop_subgraph(
    edge_index: paddle.Tensor,
    num_hops: int,
    node_idx: Union[int, List[int], paddle.Tensor],
    relabel_nodes: bool = False,
) -> paddle.Tensor:
    if not isinstance(node_idx, paddle.Tensor):
        node_idx = paddle.to_tensor(node_idx, dtype="int64")

    visited = paddle.zeros([edge_index.max() + 1], dtype="bool")
    queue = node_idx.tolist() if isinstance(node_idx, paddle.Tensor) else node_idx
    visited[queue] = True
    sub_edge_index = []

    current_hop = 0

    while queue and current_hop < num_hops:
        current_hop += 1
        next_queue = []

        for node in queue:
            neighbors = edge_index[1] == node
            neighbors = edge_index[0][neighbors]
            neighbors = neighbors[~visited[neighbors]]

            next_queue.extend(neighbors.tolist())
            visited[neighbors] = True

            for neighbor in neighbors:
                if relabel_nodes:
                    original_idx = (
                        paddle.nonzero(node_idx == node)[0].item()
                        if isinstance(node_idx, paddle.Tensor)
                        else node_idx.index(node)
                    )
                    sub_edge_index.append([original_idx, len(sub_edge_index) // 2 + 1])
                else:
                    sub_edge_index.append([node, neighbor])

        queue = next_queue

    sub_edge_index = paddle.to_tensor(sub_edge_index, dtype="int64")
    if relabel_nodes:
        return sub_edge_index.reshape([-1, 2])[:, 1]
    else:
        return sub_edge_index.reshape([-1, 2])


class CustomData:
    def __init__(self, **kwargs):
        self.edge_index = None
        for key, value in kwargs.items():
            setattr(self, key, value)
