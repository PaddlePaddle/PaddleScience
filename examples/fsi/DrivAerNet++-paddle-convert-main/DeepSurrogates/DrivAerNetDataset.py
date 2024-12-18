import sys
sys.path.append('/ssd1/ken/DrivAerNet-paddle-convert-main/utils')
import paddle_aux
import os
import paddle
"""
@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu mohamed.elrefaie@tum.de

This module is part of the research presented in the paper:
"DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks".

The module defines two PyTorch Datasets for loading and transforming 3D car models from the DrivAerNet++ dataset:
1. DrivAerNetDataset: Handles point cloud data, allowing loading, transforming, and augmenting 3D car models from STL files or existing point clouds.
2. DrivAerNetGNNDataset: Processes the dataset into graph format suitable for Graph Neural Networks (GNNs).
"""
import logging
import numpy as np
import pandas as pd
import trimesh
import pyvista as pv
import seaborn as sns
from typing import Callable, Optional, Tuple, List
from torch_geometric.data import Data
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')


class DataAugmentation:
    """
    Class encapsulating various data augmentation techniques for point clouds.
    """

    @staticmethod
    def translate_pointcloud(pointcloud: paddle.Tensor, translation_range:
        Tuple[float, float]=(2.0 / 3.0, 3.0 / 2.0)) ->paddle.Tensor:
        """
        Translates the pointcloud by a random factor within a given range.

        Args:
            pointcloud: The input point cloud as a torch.Tensor.
            translation_range: A tuple specifying the range for translation factors.

        Returns:
            Translated point cloud as a torch.Tensor.
        """
        xyz1 = np.random.uniform(low=translation_range[0], high=
            translation_range[1], size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2
            ).astype('float32')
        return paddle.to_tensor(data=translated_pointcloud, dtype='float32')

    @staticmethod
    def jitter_pointcloud(pointcloud: paddle.Tensor, sigma: float=0.01,
        clip: float=0.02) ->paddle.Tensor:
        """
        Adds Gaussian noise to the pointcloud.

        Args:
            pointcloud: The input point cloud as a torch.Tensor.
            sigma: Standard deviation of the Gaussian noise.
            clip: Maximum absolute value for noise.

        Returns:
            Jittered point cloud as a torch.Tensor.
        """
        N, C = tuple(pointcloud.shape)
        jittered_pointcloud = pointcloud + paddle.clip(x=sigma * paddle.
            randn(shape=[N, C]), min=-clip, max=clip)
        return jittered_pointcloud

    @staticmethod
    def drop_points(pointcloud: paddle.Tensor, drop_rate: float=0.1
        ) ->paddle.Tensor:
        """
        Randomly removes points from the point cloud based on the drop rate.

        Args:
            pointcloud: The input point cloud as a torch.Tensor.
            drop_rate: The percentage of points to be randomly dropped.

        Returns:
            The point cloud with points dropped as a torch.Tensor.
        """
        num_drop = int(drop_rate * pointcloud.shape[0])
        drop_indices = np.random.choice(pointcloud.shape[0], num_drop,
            replace=False)
        keep_indices = np.setdiff1d(np.arange(pointcloud.shape[0]),
            drop_indices)
        dropped_pointcloud = pointcloud[keep_indices, :]
        return dropped_pointcloud


class DrivAerNetDataset(paddle.io.Dataset):
    """
    PyTorch Dataset class for the DrivAerNet dataset, handling loading, transforming, and augmenting 3D car models.
    """

    def __init__(self, root_dir: str, csv_file: str, num_points: int, transform: Optional[Callable] = None,
                 pointcloud_exist: bool = False):
        """
        Initializes the DrivAerNetDataset instance.

        Args:
            root_dir: Directory containing the STL files for 3D car models.
            csv_file: Path to the CSV file with metadata for the models.
            num_points: Fixed number of points to sample from each 3D model.
            transform: Optional transform function to apply to each sample.
            pointcloud_exist (bool): Whether the point clouds already exist as .pt files.
        """
        super().__init__()
        self.root_dir = root_dir
        try:
            self.data_frame = pd.read_csv(csv_file)
        except Exception as e:
            logging.error(f'Failed to load CSV file: {csv_file}. Error: {e}')
            raise
        self.transform = transform
        self.num_points = num_points
        self.augmentation = DataAugmentation()
        self.pointcloud_exist = pointcloud_exist
        self.cache = {}

    def __len__(self) ->int:
        """Returns the total number of samples in the dataset."""
        return len(self.data_frame)

    def min_max_normalize(self, data: paddle.Tensor) ->paddle.Tensor:
        """
        Normalizes the data to the range [0, 1] based on min and max values.
        """
        min_vals, _ = data.min(axis=0, keepdim=True)
        max_vals, _ = data.max(axis=0, keepdim=True)
        normalized_data = (data - min_vals) / (max_vals - min_vals)
        return normalized_data

    def z_score_normalize(self, data: paddle.Tensor) ->paddle.Tensor:
        """
        Normalizes the data using z-score normalization (standard score).
        """
        mean_vals = data.mean(axis=0, keepdim=True)
        std_vals = data.std(axis=0, keepdim=True)
        normalized_data = (data - mean_vals) / std_vals
        return normalized_data

    def mean_normalize(self, data: paddle.Tensor) ->paddle.Tensor:
        """
        Normalizes the data to the range [-1, 1] based on mean and range.
        """
        mean_vals = data.mean(axis=0, keepdim=True)
        min_vals, _ = data.min(axis=0, keepdim=True)
        max_vals, _ = data.max(axis=0, keepdim=True)
        normalized_data = (data - mean_vals) / (max_vals - min_vals)
        return normalized_data

    def _sample_or_pad_vertices(self, vertices: paddle.Tensor, num_points: int
        ) ->paddle.Tensor:
        """
        Subsamples or pads the vertices of the model to a fixed number of points.

        Args:
            vertices: The vertices of the 3D model as a torch.Tensor.
            num_points: The desired number of points for the model.

        Returns:
            The vertices standardized to the specified number of points.
        """
        num_vertices = vertices.shape[0]
        if num_vertices > num_points:
            indices = np.random.choice(num_vertices, num_points, replace=False)
            vertices = vertices[indices]
        elif num_vertices < num_points:
            padding = paddle.zeros(shape=(num_points - num_vertices, 3),
                dtype='float32')
            vertices = paddle.concat(x=(vertices, padding), axis=0)
        return vertices

    def _load_point_cloud(self, design_id: str) ->Optional[paddle.Tensor]:
        load_path = os.path.join(self.root_dir, f'{design_id}.pdparams')
        if os.path.exists(load_path) and os.path.getsize(load_path) > 0:
            try:
                return paddle.load(path=str(load_path))
            except (EOFError, RuntimeError) as e:
                return None
        else:
            return None

    def __getitem__(self, idx: int, apply_augmentations: bool = True) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Retrieves a sample and its corresponding label from the dataset, with an option to apply augmentations.

        Args:
            idx (int): Index of the sample to retrieve.
            apply_augmentations (bool, optional): Whether to apply data augmentations. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The sample (point cloud) and its label (Cd value).
        """
        if paddle.is_tensor(idx):
            idx = idx.tolist()

        if idx in self.cache:
            return self.cache[idx]
        while True:
            row = self.data_frame.iloc[idx]
            design_id = row['Design']
            cd_value = row['Average Cd']

            if self.pointcloud_exist:
                vertices = self._load_point_cloud(design_id)

                if vertices is None:
                    # logging.warning(f"Skipping design {design_id} because point cloud is not found or corrupted.")
                    idx = (idx + 1) % len(self.data_frame)
                    continue
            else:
                geometry_path = os.path.join(self.root_dir, f"{design_id}.stl")
                try:
                    mesh = trimesh.load(geometry_path, force='mesh')
                    vertices = paddle.to_tensor(mesh.vertices, dtype=paddle.float32)
                    vertices = self._sample_or_pad_vertices(vertices, self.num_points)
                except Exception as e:
                    logging.error(f"Failed to load STL file: {geometry_path}. Error: {e}")
                    raise

            if apply_augmentations:
                vertices = self.augmentation.translate_pointcloud(vertices.numpy())
                vertices = self.augmentation.jitter_pointcloud(vertices)

            if self.transform:
                vertices = self.transform(vertices)

            point_cloud_normalized = self.min_max_normalize(vertices)
            cd_value = paddle.to_tensor(float(cd_value), dtype=paddle.float32).reshape([-1])

            self.cache[idx] = (point_cloud_normalized, cd_value)
            return point_cloud_normalized, cd_value

    def split_data(self, train_ratio: float=0.7, val_ratio: float=0.15,
        test_ratio: float=0.15) ->Tuple[List[int], List[int], List[int]]:
        """
        Splits the dataset into training, validation, and test sets.

        Args:
            train_ratio: The proportion of the data to be used for training.
            val_ratio: The proportion of the data to be used for validation.
            test_ratio: The proportion of the data to be used for testing.

        Returns:
            Indices for the training, validation, and test sets.
        """
        assert train_ratio + val_ratio + test_ratio == 1, 'Ratios must sum to 1'
        num_samples = len(self)
        indices = list(range(num_samples))
        train_size = int(train_ratio * num_samples)
        val_size = int(val_ratio * num_samples)
        test_size = num_samples - train_size - val_size
        train_indices, val_indices, test_indices = paddle.io.random_split(
            dataset=indices, lengths=[train_size, val_size, test_size])
        return train_indices, val_indices, test_indices

    def visualize_mesh(self, idx):
        """
        Visualize the STL mesh for a specific design from the dataset.

        Args:
            idx (int): Index of the design to visualize in the dataset.

        This function loads the mesh from the STL file corresponding to the design ID at the given index,
        wraps it using PyVista for visualization, and then sets up a PyVista plotter to display the mesh.
        """
        row = self.data_frame.iloc[idx]
        design_id = row['Design']
        geometry_path = os.path.join(self.root_dir, f'{design_id}.stl')
        try:
            mesh = trimesh.load(geometry_path, force='mesh')
        except Exception as e:
            logging.error(
                f'Failed to load STL file: {geometry_path}. Error: {e}')
            raise
        pv_mesh = pv.wrap(mesh)
        plotter = pv.Plotter()
        plotter.add_mesh(pv_mesh, color='lightgrey', show_edges=True)
        plotter.add_axes()
        camera_position = [(-11.073024242161921, -5.621499358347753, 
            5.862225824910342), (1.458462064391673, 0.002314306982062475, 
            0.6792134746589196), (0.34000174095454166, 0.10379556639001211,
            0.9346792479485448)]
        plotter.camera_position = camera_position
        plotter.show()

    def visualize_mesh_with_node(self, idx):
        """
        Visualizes the mesh for a specific design from the dataset with nodes highlighted.

        Args:
            idx (int): Index of the design to visualize in the dataset.

        This function loads the mesh from the STL file and highlights the nodes (vertices) of the mesh using spheres.
        It uses seaborn to obtain visually distinct colors for the mesh and nodes.
        """
        row = self.data_frame.iloc[idx]
        design_id = row['Design']
        geometry_path = os.path.join(self.root_dir, f'{design_id}.stl')
        try:
            mesh = trimesh.load(geometry_path, force='mesh')
            pv_mesh = pv.wrap(mesh)
        except Exception as e:
            logging.error(
                f'Failed to load STL file: {geometry_path}. Error: {e}')
            raise
        plotter = pv.Plotter()
        sns_blue = sns.color_palette('colorblind')[0]
        plotter.add_mesh(pv_mesh, color='lightgrey', show_edges=True,
            edge_color='black')
        nodes = pv_mesh.points
        plotter.add_points(nodes, color=sns_blue, point_size=5,
            render_points_as_spheres=True)
        plotter.add_axes()
        plotter.show()

    def visualize_point_cloud(self, idx):
        """
        Visualizes the point cloud for a specific design from the dataset.

        Args:
            idx (int): Index of the design to visualize in the dataset.

        This function retrieves the vertices for the specified design, converts them into a point cloud,
        and uses the z-coordinate for color mapping. PyVista's Eye-Dome Lighting is enabled for improved depth perception.
        """
        vertices, _ = self.__getitem__(idx)
        vertices = vertices.numpy()
        point_cloud = pv.PolyData(vertices)
        colors = vertices[:, 2]
        point_cloud['colors'] = colors
        plotter = pv.Plotter()
        plotter.add_points(point_cloud, scalars='colors', cmap='Blues',
            point_size=3, render_points_as_spheres=True)
        plotter.enable_eye_dome_lighting()
        plotter.add_axes()
        camera_position = [(-11.073024242161921, -5.621499358347753, 
            5.862225824910342), (1.458462064391673, 0.002314306982062475, 
            0.6792134746589196), (0.34000174095454166, 0.10379556639001211,
            0.9346792479485448)]
        plotter.camera_position = camera_position
        plotter.show()

    def visualize_augmentations(self, idx):
        """
        Visualizes various augmentations applied to the point cloud of a specific design in the dataset.

        Args:
            idx (int): Index of the sample in the dataset to be visualized.

        This function retrieves the original point cloud for the specified design and then applies a series of augmentations,
        including translation, jittering, and point dropping. Each version of the point cloud (original and augmented) is then
        visualized in a 2x2 grid using PyVista to illustrate the effects of these augmentations.
        """
        vertices, _ = self.__getitem__(idx, apply_augmentations=False)
        original_pc = pv.PolyData(vertices.numpy())
        translated_pc = self.augmentation.translate_pointcloud(vertices.numpy()
            )
        jittered_pc = self.augmentation.jitter_pointcloud(translated_pc)
        dropped_pc = self.augmentation.drop_points(jittered_pc)
        plotter = pv.Plotter(shape=(2, 2))
        plotter.subplot(0, 0)
        plotter.add_text('Original Point Cloud', font_size=10)
        plotter.add_mesh(original_pc, color='black', point_size=3)
        plotter.subplot(0, 1)
        plotter.add_text('Translated Point Cloud', font_size=10)
        plotter.add_mesh(pv.PolyData(translated_pc.numpy()), color=
            'lightblue', point_size=3)
        plotter.subplot(1, 0)
        plotter.add_text('Jittered Point Cloud', font_size=10)
        plotter.add_mesh(pv.PolyData(jittered_pc.numpy()), color=
            'lightgreen', point_size=3)
        plotter.subplot(1, 1)
        plotter.add_text('Dropped Point Cloud', font_size=10)
        plotter.add_mesh(pv.PolyData(dropped_pc.numpy()), color='salmon',
            point_size=3)
        plotter.show()


class DrivAerNetGNNDataset(paddle.io.Dataset):
    """
    PyTorch Dataset for loading and processing the DrivAerNet dataset into graph format suitable for GNNs.
    """

    def __init__(self, root_dir: str, csv_file: str, normalize: bool=True):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Path to the directory containing the STL files.
            csv_file (str): Path to the CSV file containing metadata such as aerodynamic coefficients.
            normalize (bool): Whether to normalize the node features.
        """
        self.root_dir = root_dir
        self.data_frame = pd.read_csv(csv_file)
        self.normalize = normalize
        self.cache = {}

    def __len__(self) ->int:
        """
        Return the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data_frame)

    def min_max_normalize(self, data: paddle.Tensor) ->paddle.Tensor:
        """
        Normalizes the data to the range [0, 1] based on min and max values.

        Args:
            data (torch.Tensor): The input data tensor to be normalized.

        Returns:
            torch.Tensor: The normalized data tensor.
        """
        min_vals, _ = data.min(axis=0, keepdim=True)
        max_vals, _ = data.max(axis=0, keepdim=True)
        normalized_data = (data - min_vals) / (max_vals - min_vals)
        return normalized_data

    def __getitem__(self, idx: int) ->Data:
        """
        Get a graph data item for GNN processing.

        Args:
            idx (int): Index of the item.

        Returns:
            Data: A PyTorch Geometric Data object containing edge_index, x (node features), and y (target variable).
        """
        if paddle.is_tensor(x=idx):
            idx = idx.tolist()
        if idx in self.cache:
            return self.cache[idx]
        row = self.data_frame.iloc[idx]
        stl_path = os.path.join(self.root_dir, f"{row['Design']}.stl")
        cd_value = row['Average Cd']
        try:
            mesh = trimesh.load(stl_path, force='mesh')
        except Exception as e:
            logging.error(f'Failed to load STL file: {stl_path}. Error: {e}')
            raise
        edge_index = paddle.to_tensor(data=np.array(mesh.edges).T, dtype=
            'int64')
        x = paddle.to_tensor(data=mesh.vertices, dtype='float32')
        if self.normalize:
            x = self.min_max_normalize(x)
        y = paddle.to_tensor(data=[cd_value], dtype='float32')
        data = Data(x=x, edge_index=edge_index, y=y)
        self.cache[idx] = data
        return data

    def visualize_mesh_with_node(self, idx: int) ->None:
        """
        Visualizes the mesh of a given sample index with triangles in light grey and nodes highlighted as spheres.

        Args:
            idx (int): Index of the sample to visualize.
        """
        row = self.data_frame.iloc[idx]
        design_id = row['Design']
        geometry_path = os.path.join(self.root_dir, f'{design_id}.stl')
        try:
            mesh = trimesh.load(geometry_path, force='mesh')
            pv_mesh = pv.wrap(mesh)
        except Exception as e:
            logging.error(
                f'Failed to load STL file: {geometry_path}. Error: {e}')
            raise
        plotter = pv.Plotter()
        sns_blue = sns.color_palette('colorblind')[0]
        plotter.add_mesh(pv_mesh, color='lightgrey', show_edges=True,
            edge_color='black')
        nodes = pv_mesh.points
        plotter.add_points(nodes, color=sns_blue, point_size=5,
            render_points_as_spheres=True)
        plotter.add_axes()
        camera_position = [(-11.073024242161921, -5.621499358347753, 
            5.862225824910342), (1.458462064391673, 0.002314306982062475, 
            0.6792134746589196), (0.34000174095454166, 0.10379556639001211,
            0.9346792479485448)]
        plotter.camera_position = camera_position
        plotter.show()

    def visualize_graph(self, idx: int) ->None:
        """
        Visualizes the graph representation of the 3D mesh using PyVista.

        Args:
            idx (int): Index of the sample to visualize.
        """
        data = self[idx]
        mesh = pv.PolyData(data.x.numpy())
        edges = data.edge_index.t().numpy()
        lines = np.full((tuple(edges.shape)[0], 3), 2, dtype=np.int_)
        lines[:, 1:] = edges
        mesh.lines = lines
        mesh['scalars'] = np.random.rand(mesh.n_points)
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, show_edges=True, line_width=1, color='white',
            point_size=8, render_points_as_spheres=True)
        plotter.add_scalar_bar('Scalar Values', 'scalars')
        edge_points = mesh.points[edges.flatten()]
        lines = pv.lines_from_points(edge_points)
        plotter.add_mesh(lines, color='blue', line_width=2)
        plotter.show()
