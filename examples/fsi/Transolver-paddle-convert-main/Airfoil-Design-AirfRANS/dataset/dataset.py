import sys

# sys.path.append('../../utils')
from utils import paddle_aux
import os
import paddle
import numpy as np
import pyvista as pv
from utils.reorganize import reorganize
from tqdm import tqdm
from paddle.io import Dataset

class Data:
    def __init__(self, pos=None, x=None, y=None, surf=None, edge_index=None):
        self.pos = pos  # 节点的坐标
        self.x = x  # 节点特征
        self.y = y  # 标签或目标值
        self.edge_index = edge_index  # 边的索引
        self.surf = surf  # 其他自定义属性，如 surf

    def to(self, device):
        # 将数据移动到指定设备（如 GPU 或 CPU）
        if self.pos is not None:
            self.pos = self.pos.to(device)
        if self.x is not None:
            self.x = self.x.to(device)
        if self.y is not None:
            self.y = self.y.to(device)
        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(device)
        if self.surf is not None:
            self.surf = self.surf.to(device)
        return self

    def clone(self):
        # 创建当前 Data 对象的深拷贝
        pos_clone = self.pos.clone() if self.pos is not None else None
        x_clone = self.x.clone() if self.x is not None else None
        y_clone = self.y.clone() if self.y is not None else None
        edge_index_clone = self.edge_index.clone() if self.edge_index is not None else None
        surf_clone = self.surf.clone() if self.surf is not None else None
        return Data(pos=pos_clone, x=x_clone, y=y_clone, surf=surf_clone, edge_index=edge_index_clone)

    def size(self, dim=None):
        """返回 x 的大小或指定维度的大小."""
        if self.x is not None:
            # 如果 dim 是整数，则返回对应维度的大小；否则返回完整形状
            if dim is not None and isinstance(dim, int):
                return self.x.shape[dim]
            return self.x.shape
        else:
            raise AttributeError("Attribute 'x' is not set.")

    def __repr__(self):
        return (f"Data(x={self._format_attr(self.x)}, "
                f"edge_index={self._format_attr(self.edge_index)}, "
                f"y={self._format_attr(self.y)}, "
                f"pos={self._format_attr(self.pos)}, "
                f"surf={self._format_attr(self.surf)})")

    def _format_attr(self, attr):
        if attr is None:
            return "None"
        elif hasattr(attr, 'shape'):
            return f"[{', '.join(map(str, attr.shape))}]"
        else:
            return str(attr)


def cell_sampling_2d(cell_points, cell_attr=None):
    """
    Sample points in a two dimensional cell via parallelogram sampling and triangle interpolation via barycentric coordinates. The vertices have to be ordered in a certain way.

    Args:
        cell_points (array): Vertices of the 2 dimensional cells. Shape (N, 4) for N cells with 4 vertices.
        cell_attr (array, optional): Features of the vertices of the 2 dimensional cells. Shape (N, 4, k) for N cells with 4 edges and k features. 
            If given shape (N, 4) it will resize it automatically in a (N, 4, 1) array. Default: ``None``
    """
    v0, v1 = cell_points[:, 1] - cell_points[:, 0], cell_points[:, 3
                                                    ] - cell_points[:, 0]
    v2, v3 = cell_points[:, 3] - cell_points[:, 2], cell_points[:, 1
                                                    ] - cell_points[:, 2]
    a0, a1 = np.abs(np.linalg.det(np.hstack([v0[:, :2], v1[:, :2]]).reshape
                                  (-1, 2, 2))), np.abs(np.linalg.det(np.hstack([v2[:, :2], v3[:, :2]]
                                                                               ).reshape(-1, 2, 2)))
    p = a0 / (a0 + a1)
    index_triangle = np.random.binomial(1, p)[:, None]
    u = np.random.uniform(size=(len(p), 2))
    sampled_point = index_triangle * (u[:, 0:1] * v0 + u[:, 1:2] * v1) + (1 -
                                                                          index_triangle) * (
                            u[:, 0:1] * v2 + u[:, 1:2] * v3)
    sampled_point_mirror = index_triangle * ((1 - u[:, 0:1]) * v0 + (1 - u[
                                                                         :, 1:2]) * v1) + (1 - index_triangle) * (
                                   (1 - u[:, 0:1]) * v2 + (1 -
                                                           u[:, 1:2]) * v3)
    reflex = u.sum(axis=1) > 1
    sampled_point[reflex] = sampled_point_mirror[reflex]
    if cell_attr is not None:
        t0, t1, t2 = np.zeros_like(v0), index_triangle * v0 + (1 -
                                                               index_triangle) * v2, index_triangle * v1 + (
                             1 - index_triangle
                     ) * v3
        w = (t1[:, 1] - t2[:, 1]) * (t0[:, 0] - t2[:, 0]) + (t2[:, 0] - t1[
                                                                        :, 0]) * (t0[:, 1] - t2[:, 1])
        w0 = (t1[:, 1] - t2[:, 1]) * (sampled_point[:, 0] - t2[:, 0]) + (t2
                                                                         [:, 0] - t1[:, 0]) * (
                     sampled_point[:, 1] - t2[:, 1])
        w1 = (t2[:, 1] - t0[:, 1]) * (sampled_point[:, 0] - t2[:, 0]) + (t0
                                                                         [:, 0] - t2[:, 0]) * (
                     sampled_point[:, 1] - t2[:, 1])
        w0, w1 = w0 / w, w1 / w
        w2 = 1 - w0 - w1
        if len(tuple(cell_attr.shape)) == 2:
            cell_attr = cell_attr[:, :, None]
        attr0 = index_triangle * cell_attr[:, 0] + (1 - index_triangle
                                                    ) * cell_attr[:, 2]
        attr1 = index_triangle * cell_attr[:, 1] + (1 - index_triangle
                                                    ) * cell_attr[:, 1]
        attr2 = index_triangle * cell_attr[:, 3] + (1 - index_triangle
                                                    ) * cell_attr[:, 3]
        sampled_attr = w0[:, None] * attr0 + w1[:, None] * attr1 + w2[:, None
                                                                   ] * attr2
    sampled_point += index_triangle * cell_points[:, 0] + (1 - index_triangle
                                                           ) * cell_points[:, 2]
    return np.hstack([sampled_point[:, :2], sampled_attr]
                     ) if cell_attr is not None else sampled_point[:, :2]


def cell_sampling_1d(line_points, line_attr=None):
    """
    Sample points in a one dimensional cell via linear sampling and interpolation.

    Args:
        line_points (array): Edges of the 1 dimensional cells. Shape (N, 2) for N cells with 2 edges.
        line_attr (array, optional): Features of the edges of the 1 dimensional cells. Shape (N, 2, k) for N cells with 2 edges and k features.
            If given shape (N, 2) it will resize it automatically in a (N, 2, 1) array. Default: ``None``
    """
    u = np.random.uniform(size=(len(line_points), 1))
    sampled_point = u * line_points[:, 0] + (1 - u) * line_points[:, 1]
    if line_attr is not None:
        if len(tuple(line_attr.shape)) == 2:
            line_attr = line_attr[:, :, None]
        sampled_attr = u * line_attr[:, 0] + (1 - u) * line_attr[:, 1]
    return np.hstack([sampled_point[:, :2], sampled_attr]
                     ) if line_attr is not None else sampled_point[:, :2]


def Dataset(set, norm=False, coef_norm=None, crop=None, sample=None, n_boot=int(500000.0), surf_ratio=0.1, my_path='/data/path'):
    """
    Create a list of simulation to input in a PyTorch Geometric DataLoader. Simulation are transformed by keeping vertices of the CFD mesh or
    by sampling (uniformly or via the mesh density) points in the simulation cells.

    Args:
        set (list): List of geometry names to include in the dataset.
        norm (bool, optional): If norm is set to ``True``, the mean and the standard deviation of the dataset will be computed and returned.
            Moreover, the dataset will be normalized by these quantities. Ignored when ``coef_norm`` is not None. Default: ``False``
        coef_norm (tuple, optional): This has to be a tuple of the form (mean input, std input, mean output, std ouput) if not None.
            The dataset generated will be normalized by those quantites. Default: ``None``
        crop (list, optional): List of the vertices of the rectangular [xmin, xmax, ymin, ymax] box to crop simulations. Default: ``None``
        sample (string, optional): Type of sampling. If ``None``, no sampling strategy is applied and the nodes of the CFD mesh are returned.
            If ``uniform`` or ``mesh`` is chosen, uniform or mesh density sampling is applied on the domain. Default: ``None``
        n_boot (int, optional): Used only if sample is not None, gives the size of the sampling for each simulation. Defaul: ``int(5e5)``
        surf_ratio (float, optional): Used only if sample is not None, gives the ratio of point over the airfoil to sample with respect to point
            in the volume. Default: ``0.1``
    """
    if norm and coef_norm is not None:
        raise ValueError(
            'If coef_norm is not None and norm is True, the normalization will be done via coef_norm'
        )
    dataset = []
    for k, s in enumerate(tqdm(set)):
        internal = pv.read(os.path.join(my_path, s, s + '_internal.vtu'))
        aerofoil = pv.read(os.path.join(my_path, s, s + '_aerofoil.vtp'))
        internal = internal.compute_cell_sizes(length=False, volume=False)
        if crop is not None:
            bounds = crop[0], crop[1], crop[2], crop[3], 0, 1
            internal = internal.clip_box(bounds=bounds, invert=False,
                                         crinkle=True)
        if sample is not None:
            if sample == 'uniform':
                p = internal.cell_data['Area'] / internal.cell_data['Area'
                ].sum()
                sampled_cell_indices = np.random.choice(internal.n_cells,
                                                        size=n_boot, p=p)
                surf_p = aerofoil.cell_data['Length'] / aerofoil.cell_data[
                    'Length'].sum()
                sampled_line_indices = np.random.choice(aerofoil.n_cells,
                                                        size=int(n_boot * surf_ratio), p=surf_p)
            elif sample == 'mesh':
                sampled_cell_indices = np.random.choice(internal.n_cells,
                                                        size=n_boot)
                sampled_line_indices = np.random.choice(aerofoil.n_cells,
                                                        size=int(n_boot * surf_ratio))
            cell_dict = internal.cells.reshape(-1, 5)[sampled_cell_indices, 1:]
            cell_points = internal.points[cell_dict]
            line_dict = aerofoil.lines.reshape(-1, 3)[sampled_line_indices, 1:]
            line_points = aerofoil.points[line_dict]
            geom = -internal.point_data['implicit_distance'][cell_dict, None]
            Uinf, alpha = float(s.split('_')[2]), float(s.split('_')[3]
                                                        ) * np.pi / 180
            u = (np.array([np.cos(alpha), np.sin(alpha)]) * Uinf).reshape(1, 2
                                                                          ) * np.ones_like(
                internal.point_data['U'][cell_dict, :1])
            normal = np.zeros_like(u)
            surf_geom = np.zeros_like(aerofoil.point_data['U'][line_dict, :1])
            surf_u = (np.array([np.cos(alpha), np.sin(alpha)]) * Uinf).reshape(
                1, 2) * np.ones_like(aerofoil.point_data['U'][line_dict, :1])
            surf_normal = -aerofoil.point_data['Normals'][line_dict, :2]
            attr = np.concatenate([u, geom, normal, internal.point_data['U'
                                                    ][cell_dict, :2], internal.point_data['p'][cell_dict, None],
                                   internal.point_data['nut'][cell_dict, None]], axis=-1)
            surf_attr = np.concatenate([surf_u, surf_geom, surf_normal,
                                        aerofoil.point_data['U'][line_dict, :2], aerofoil.
                                       point_data['p'][line_dict, None], aerofoil.point_data['nut'
                                        ][line_dict, None]], axis=-1)
            sampled_points = cell_sampling_2d(cell_points, attr)
            surf_sampled_points = cell_sampling_1d(line_points, surf_attr)
            pos = sampled_points[:, :2]
            init = sampled_points[:, :7]
            target = sampled_points[:, 7:]
            surf_pos = surf_sampled_points[:, :2]
            surf_init = surf_sampled_points[:, :7]
            surf_target = surf_sampled_points[:, 7:]
            surf = paddle.concat(x=[paddle.zeros(shape=len(pos)), paddle.
                                 ones(shape=len(surf_pos))], axis=0)
            pos = paddle.concat(x=[paddle.to_tensor(data=pos, dtype=
            'float32'), paddle.to_tensor(data=surf_pos, dtype='float32'
                                         )], axis=0)
            x = paddle.concat(x=[paddle.to_tensor(data=init, dtype=
            'float32'), paddle.to_tensor(data=surf_init, dtype=
            'float32')], axis=0)
            y = paddle.concat(x=[paddle.to_tensor(data=target, dtype=
            'float32'), paddle.to_tensor(data=surf_target, dtype=
            'float32')], axis=0)
        else:
            surf_bool = internal.point_data['U'][:, 0] == 0
            geom = -internal.point_data['implicit_distance'][:, None]
            Uinf, alpha = float(s.split('_')[2]), float(s.split('_')[3]
                                                        ) * np.pi / 180
            u = (np.array([np.cos(alpha), np.sin(alpha)]) * Uinf).reshape(1, 2
                                                                          ) * np.ones_like(
                internal.point_data['U'][:, :1])
            normal = np.zeros_like(u)
            normal[surf_bool] = reorganize(aerofoil.points[:, :2], internal
                                           .points[surf_bool, :2], -aerofoil.point_data['Normals'][:, :2])
            attr = np.concatenate([u, geom, normal, internal.point_data['U'
                                                    ][:, :2], internal.point_data['p'][:, None], internal.
                                  point_data['nut'][:, None]], axis=-1)
            pos = internal.points[:, :2]
            init = np.concatenate([pos, attr[:, :5]], axis=1)
            target = attr[:, 5:]
            surf = paddle.to_tensor(data=surf_bool)
            pos = paddle.to_tensor(data=pos, dtype='float32')
            x = paddle.to_tensor(data=init, dtype='float32')
            y = paddle.to_tensor(data=target, dtype='float32')
        if norm and coef_norm is None:
            if k == 0:
                old_length = tuple(init.shape)[0]
                mean_in = init.mean(axis=0, dtype=np.double)
                mean_out = target.mean(axis=0, dtype=np.double)
            else:
                new_length = old_length + tuple(init.shape)[0]
                mean_in += (init.sum(axis=0, dtype=np.double) - tuple(init.
                                                                      shape)[0] * mean_in) / new_length
                mean_out += (target.sum(axis=0, dtype=np.double) - tuple(
                    init.shape)[0] * mean_out) / new_length
                old_length = new_length
        data = Data(pos=pos, x=x, y=y, surf=surf.astype(dtype='bool'))
        dataset.append(data)
    if norm and coef_norm is None:
        mean_in = mean_in.astype(np.single)
        mean_out = mean_out.astype(np.single)
        for k, data in enumerate(dataset):
            if k == 0:
                old_length = tuple(data.x.numpy().shape)[0]
                std_in = ((data.x.numpy() - mean_in) ** 2).sum(axis=0,
                                                               dtype=np.double) / old_length
                std_out = ((data.y.numpy() - mean_out) ** 2).sum(axis=0,
                                                                 dtype=np.double) / old_length
            else:
                new_length = old_length + tuple(data.x.numpy().shape)[0]
                std_in += (((data.x.numpy() - mean_in) ** 2).sum(axis=0,
                                                                 dtype=np.double) - tuple(data.x.numpy().shape)[
                               0] * std_in
                           ) / new_length
                std_out += (((data.y.numpy() - mean_out) ** 2).sum(axis=0,
                                                                   dtype=np.double) - tuple(data.x.numpy().shape)[
                                0] * std_out
                            ) / new_length
                old_length = new_length
        std_in = np.sqrt(std_in).astype(np.single)
        std_out = np.sqrt(std_out).astype(np.single)
        for data in dataset:
            data.x = (data.x - mean_in) / (std_in + 1e-08)
            data.y = (data.y - mean_out) / (std_out + 1e-08)
        coef_norm = mean_in, std_in, mean_out, std_out
        dataset = dataset, coef_norm
    elif coef_norm is not None:
        for data in dataset:
            data.x = (data.x - coef_norm[0]) / (coef_norm[1] + 1e-08)
            data.y = (data.y - coef_norm[2]) / (coef_norm[3] + 1e-08)
    return dataset

# class CFDataset:
#     def __init__(self, set, norm=False, coef_norm=None, crop=None, sample=None, n_boot=int(500000.0), surf_ratio=0.1, my_path='/data/path'):
#         """
#         Create a list of simulation to input in a Paddle DataLoader. Simulation are transformed by keeping vertices of the CFD mesh or
#         by sampling (uniformly or via the mesh density) points in the simulation cells.
#
#         Args:
#             set (list): List of geometry names to include in the dataset.
#             norm (bool, optional): If norm is set to ``True``, the mean and the standard deviation of the dataset will be computed and returned.
#                 Moreover, the dataset will be normalized by these quantities. Ignored when ``coef_norm`` is not None. Default: ``False``
#             coef_norm (tuple, optional): This has to be a tuple of the form (mean input, std input, mean output, std ouput) if not None.
#                 The dataset generated will be normalized by those quantites. Default: ``None``
#             crop (list, optional): List of the vertices of the rectangular [xmin, xmax, ymin, ymax] box to crop simulations. Default: ``None``
#             sample (string, optional): Type of sampling. If ``None``, no sampling strategy is applied and the nodes of the CFD mesh are returned.
#                 If ``uniform`` or ``mesh`` is chosen, uniform or mesh density sampling is applied on the domain. Default: ``None``
#             n_boot (int, optional): Used only if sample is not None, gives the size of the sampling for each simulation. Default: ``int(5e5)``
#             surf_ratio (float, optional): Used only if sample is not None, gives the ratio of point over the airfoil to sample with respect to point
#                 in the volume. Default: ``0.1``
#         """
#         self.set = set
#         self.norm = norm
#         self.coef_norm = coef_norm
#         self.crop = crop
#         self.sample = sample
#         self.n_boot = n_boot
#         self.surf_ratio = surf_ratio
#         self.my_path = my_path
#         self.dataset = []
#         self.mean_in, self.std_in, self.mean_out, self.std_out = None, None, None, None
#
#         # Load the dataset
#         self._load_dataset()
#
#         # Compute normalization if required
#         if self.norm and self.coef_norm is None:
#             self._compute_normalization()
#             self._apply_normalization()
#         elif self.coef_norm is not None:
#             self.mean_in, self.std_in, self.mean_out, self.std_out = self.coef_norm
#             self._apply_normalization()
#
#
#     def _load_dataset(self):
#         """
#         Load all samples into the dataset.
#         """
#         for k, s in enumerate(tqdm(self.set, desc="Loading dataset")):
#             data = self._load_single_sample(s, k)
#             self.dataset.append(data)
#
#     def _load_single_sample(self, s, k):
#         """
#         Load a single sample and return a Data object.
#         """
#         internal = pv.read(os.path.join(self.my_path, s, s + '_internal.vtu'))
#         aerofoil = pv.read(os.path.join(self.my_path, s, s + '_aerofoil.vtp'))
#         internal = internal.compute_cell_sizes(length=False, volume=False)
#
#         # Apply cropping if specified
#         if self.crop is not None:
#             bounds = self.crop[0], self.crop[1], self.crop[2], self.crop[3], 0, 1
#             internal = internal.clip_box(bounds=bounds, invert=False, crinkle=True)
#
#         # Sampling logic
#         if self.sample is not None:
#             pos, x, y, surf = self._sample_data(internal, aerofoil, s)
#         else:
#             surf_bool = internal.point_data['U'][:, 0] == 0
#             geom = -internal.point_data['implicit_distance'][:, None]
#             Uinf, alpha = float(s.split('_')[2]), float(s.split('_')[3]) * np.pi / 180
#             u = (np.array([np.cos(alpha), np.sin(alpha)]) * Uinf).reshape(1, 2) * np.ones_like(
#                 internal.point_data['U'][:, :1]
#             )
#             normal = np.zeros_like(u)
#             normal[surf_bool] = reorganize(
#                 aerofoil.points[:, :2], internal.points[surf_bool, :2], -aerofoil.point_data['Normals'][:, :2]
#             )
#             attr = np.concatenate(
#                 [u, geom, normal, internal.point_data['U'][:, :2], internal.point_data['p'][:, None], internal.point_data['nut'][:, None]],
#                 axis=-1
#             )
#             pos = paddle.to_tensor(data=internal.points[:, :2], dtype='float32')
#             x = paddle.to_tensor(data=attr[:, :5], dtype='float32')
#             y = paddle.to_tensor(data=attr[:, 5:], dtype='float32')
#             surf = paddle.to_tensor(data=surf_bool, dtype='bool')
#
#         # 检查 x 是否为空
#         if x is None or x.size == 0:
#             raise ValueError(
#                 f"Failed to load x for sample {s} at index {k}. Check input files or preprocessing logic.")
#
#         # print(f"Loaded sample {s}: x.shape={x.shape}, y.shape={y.shape}")
#         return Data(pos=pos, x=x, y=y, surf=surf)
#
#     def _sample_data(self, internal, aerofoil, s):
#         """
#         Perform sampling on the data and return sampled points and attributes.
#         """
#         if self.sample == 'uniform':
#             p = internal.cell_data['Area'] / internal.cell_data['Area'].sum()
#             sampled_cell_indices = np.random.choice(internal.n_cells, size=self.n_boot, p=p)
#             surf_p = aerofoil.cell_data['Length'] / aerofoil.cell_data['Length'].sum()
#             sampled_line_indices = np.random.choice(aerofoil.n_cells, size=int(self.n_boot * self.surf_ratio), p=surf_p)
#         elif self.sample == 'mesh':
#             sampled_cell_indices = np.random.choice(internal.n_cells, size=self.n_boot)
#             sampled_line_indices = np.random.choice(aerofoil.n_cells, size=int(self.n_boot * self.surf_ratio))
#
#         cell_dict = internal.cells.reshape(-1, 5)[sampled_cell_indices, 1:]
#         cell_points = internal.points[cell_dict]
#         line_dict = aerofoil.lines.reshape(-1, 3)[sampled_line_indices, 1:]
#         line_points = aerofoil.points[line_dict]
#
#         geom = -internal.point_data['implicit_distance'][cell_dict, None]
#         Uinf, alpha = float(s.split('_')[2]), float(s.split('_')[3]) * np.pi / 180
#         u = (np.array([np.cos(alpha), np.sin(alpha)]) * Uinf).reshape(1, 2) * np.ones_like(internal.point_data['U'][cell_dict, :1])
#         normal = np.zeros_like(u)
#         surf_geom = np.zeros_like(aerofoil.point_data['U'][line_dict, :1])
#         surf_u = (np.array([np.cos(alpha), np.sin(alpha)]) * Uinf).reshape(1, 2) * np.ones_like(aerofoil.point_data['U'][line_dict, :1])
#         surf_normal = -aerofoil.point_data['Normals'][line_dict, :2]
#
#         attr = np.concatenate(
#             [u, geom, normal, internal.point_data['U'][cell_dict, :2], internal.point_data['p'][cell_dict, None], internal.point_data['nut'][cell_dict, None]],
#             axis=-1
#         )
#         surf_attr = np.concatenate(
#             [surf_u, surf_geom, surf_normal, aerofoil.point_data['U'][line_dict, :2], aerofoil.point_data['p'][line_dict, None], aerofoil.point_data['nut'][line_dict, None]],
#             axis=-1
#         )
#
#         sampled_points = cell_sampling_2d(cell_points, attr)
#         surf_sampled_points = cell_sampling_1d(line_points, surf_attr)
#
#         pos = paddle.concat(
#             [paddle.to_tensor(sampled_points[:, :2], dtype='float32'), paddle.to_tensor(surf_sampled_points[:, :2], dtype='float32')], axis=0
#         )
#         x = paddle.concat(
#             [paddle.to_tensor(sampled_points[:, :7], dtype='float32'), paddle.to_tensor(surf_sampled_points[:, :7], dtype='float32')], axis=0
#         )
#         y = paddle.concat(
#             [paddle.to_tensor(sampled_points[:, 7:], dtype='float32'), paddle.to_tensor(surf_sampled_points[:, 7:], dtype='float32')], axis=0
#         )
#         surf = paddle.concat(
#             [paddle.zeros(shape=[len(sampled_points)]), paddle.ones(shape=[len(surf_sampled_points)])], axis=0
#         )
#
#         return pos, x, y, surf
#
#     def _compute_normalization(self):
#         """
#         Compute mean and std for normalization.
#         """
#         print("Computing normalization...")
#         for k, data in enumerate(self.dataset):
#             if data.x is None:
#                 raise ValueError(f"Data.x is None for sample at index {k}")
#             x, y = data.x.numpy(), data.y.numpy()
#             if k == 0:
#                 self.mean_in = x.mean(axis=0, dtype=np.double)
#                 self.std_in = x.std(axis=0, dtype=np.double)
#                 self.mean_out = y.mean(axis=0, dtype=np.double)
#                 self.std_out = y.std(axis=0, dtype=np.double)
#             else:
#                 self.mean_in += x.mean(axis=0, dtype=np.double)
#                 self.std_in += x.std(axis=0, dtype=np.double)
#                 self.mean_out += y.mean(axis=0, dtype=np.double)
#                 self.std_out += y.std(axis=0, dtype=np.double)
#
#         # # 检查 mean 和 std 是否正确
#         # if self.mean_in is None or self.std_in is None:
#         #     raise ValueError("Failed to compute mean or std for input features.")
#         # print(f"Mean input: {self.mean_in}, Std input: {self.std_in}")
#         # print(f"Mean output: {self.mean_out}, Std output: {self.std_out}")
#
#     def _apply_normalization(self):
#         """
#         Apply normalization to the dataset.
#         """
#         for data in self.dataset:
#             if data.x is None:
#                 raise ValueError(f"Data.x is None for sample {data}. Check data loading process.")
#             # print(f"Normalizing data.x: {data.x.shape}, mean: {self.mean_in.shape}, std: {self.std_in.shape}")
#             data.x = (data.x - self.mean_in) / (self.std_in + 1e-8)
#             data.y = (data.y - self.mean_out) / (self.std_out + 1e-8)
#
#     def __len__(self):
#         """
#         Return the length of the dataset.
#         """
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         """
#         Get a single item from the dataset.
#         """
#         data = self.dataset[idx]
#         print(f"Returning sample {idx}: {data}")
#         return data
#
#     def __call__(self):
#         """
#         Make the class callable to maintain compatibility with the original function style.
#         """
#         if self.norm:
#             return self.dataset, (self.mean_in, self.std_in, self.mean_out, self.std_out)
#         else:
#             return self.dataset
#
#     def get(self, idx):
#         data = self.datalist[idx]
#         return data