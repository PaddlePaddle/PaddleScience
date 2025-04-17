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


# Code is heavily based on paper "Geometry-Informed Neural Operator for Large-Scale 3D PDEs", we use paddle to reproduce the results of the paper


import warnings
from pathlib import Path
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import open3d as o3d
import paddle
from src.data.base_datamodule import BaseDataModule


class UnitGaussianNormalizer:
    def __init__(self, x, eps=1e-05, reduce_dim=0, verbose=True):
        super().__init__()
        n_samples, *shape = x.shape
        self.sample_shape = shape
        self.verbose = verbose
        self.reduce_dim = reduce_dim
        y = x.numpy()
        self.mean = paddle.to_tensor(
            np.mean(y, axis=reduce_dim, keepdims=True).squeeze(axis=0)
        )
        self.std = paddle.to_tensor(
            np.std(y, axis=reduce_dim, keepdims=True).squeeze(axis=0)
        )
        self.eps = eps
        if verbose:
            print(
                f"UnitGaussianNormalizer init on {n_samples}, reducing over {reduce_dim}, samples of shape {shape}."
            )
            print(f"   Mean and std of shape {self.mean.shape}, eps={eps}")

    def encode(self, x):
        x -= self.mean
        x /= self.std + self.eps
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps
                mean = self.mean[:, sample_idx]
        x *= std
        x += mean
        return x

    def cuda(self):
        self.mean = self.mean
        self.std = self.std
        return self

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        return self

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


class DictDataset(paddle.io.Dataset):
    def __init__(self, data_dict: dict):
        self.data_dict = data_dict
        for k, v in data_dict.items():
            assert len(v) == len(
                data_dict[list(data_dict.keys())[0]]
            ), "All data must have the same length"

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.data_dict.items()}

    def __len__(self):
        return len(self.data_dict[list(self.data_dict.keys())[0]])


class DictDatasetWithConstant(DictDataset):
    def __init__(self, data_dict: dict, constant_dict: dict):
        super().__init__(data_dict)
        self.constant_dict = constant_dict
        self.lazy_load_data = True if "data_module" in constant_dict else False

    def __getitem__(self, index):
        return_dict = {k: v[index] for k, v in self.data_dict.items()}
        return_dict.update(self.constant_dict)

        if self.lazy_load_data is True:
            data_dir = self.constant_dict["data_dir"]
            data_module = self.constant_dict["data_module"]
            mesh_index = self.constant_dict["mesh_index"][index]

            if "test" in str(data_dir):
                pass
            else:
                # only for train and validation, we load pressure
                # load [p]
                p = data_module.load_pressure(data_dir, "", mesh_index)
                p = paddle.to_tensor(data=p, dtype="float32")

                # encode [p]
                encode = data_module.pressure_normalization.encode
                p = encode(p)
                return_dict["pressure"] = p

            # load [centroid]
            centroid = data_module.load_centroid(data_dir, "", mesh_index)
            centroid = paddle.to_tensor(data=centroid)

            # normalized [centroid]
            centroid = data_module.location_normalization(
                centroid, data_module.min_bounds, data_module.max_bounds
            )
            return_dict[
                "vertices"
            ] = centroid  # It should be centroid, not vertices, TODO: fix it
        return return_dict

    def lazy_load(self, index, return_dict):
        """
        Lazy loading function for pressure and centroid data.

        Args:
            index (int): The index of the mesh to be loaded.
            return_dict (dict): A dictionary to store the loaded data.

        Returns:
            dict: A dictionary containing the loaded pressure and centroid data.

        """

        return return_dict


class CFDDataModule(BaseDataModule):
    def __init__(self, data_dir, n_train: int = 500, n_test: int = 111):
        super().__init__()
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        data_dir = data_dir.expanduser()
        assert data_dir.exists(), f"Path does not exist: {data_dir}"
        assert data_dir.is_dir(), f"Path is not a directory: {data_dir}"
        self.data_dir = data_dir
        valid_mesh_inds = self.load_valid_mesh_indices(data_dir)
        assert n_train + n_test <= len(valid_mesh_inds), "Not enough data"
        if n_train + n_test < len(valid_mesh_inds):
            warnings.warn(
                f"{len(valid_mesh_inds)} meshes are available, but {n_train + n_test} are requested."
            )
        train_indices = valid_mesh_inds[:n_train]
        test_indices = valid_mesh_inds[-n_test:]
        train_mesh_paths = [self.get_mesh_path(data_dir, i) for i in train_indices]
        test_mesh_paths = [self.get_mesh_path(data_dir, i) for i in test_indices]
        train_vertices = [
            self.vertices_from_mesh(mesh_path) for mesh_path in train_mesh_paths
        ]
        test_vertices = [
            self.vertices_from_mesh(mesh_path) for mesh_path in test_mesh_paths
        ]
        train_pressure = paddle.stack(
            x=[
                paddle.to_tensor(
                    data=self.load_pressure(data_dir, mesh_index), dtype="float32"
                )
                for mesh_index in train_indices
            ]
        )
        test_pressure = paddle.stack(
            x=[
                paddle.to_tensor(
                    data=self.load_pressure(data_dir, mesh_index), dtype="float32"
                )
                for mesh_index in test_indices
            ]
        )
        pressure_normalization = UnitGaussianNormalizer(
            train_pressure, eps=1e-06, reduce_dim=[0, 1], verbose=False
        )
        train_pressure = pressure_normalization.encode(train_pressure)
        test_pressure = pressure_normalization.encode(test_pressure)
        self._train_data = DictDataset(
            {"vertices": train_vertices, "pressure": train_pressure}
        )
        self._test_data = DictDataset(
            {"vertices": test_vertices, "pressure": test_pressure}
        )
        self.output_normalization = pressure_normalization

    def encode(self, pressure: paddle.Tensor) -> paddle.Tensor:
        return self.output_normalization.encode(pressure)

    def decode(self, ouput: paddle.Tensor) -> paddle.Tensor:
        # output.shape = [batch_size, points_number, 1]
        pressure = ouput[0, :, 0].reshape((-1, 1))
        pressure_decode = self.output_normalization[0].decode(pressure)
        if len(self.output_normalization) == 2:
            wss = ouput[:, 1].reshape((-1, 1))
            wss_decode = self.output_normalization[1].decode(wss)
            return paddle.concat([pressure_decode, wss_decode], axis=1)
        return pressure_decode

    @property
    def train_data(self):
        return self._train_data

    @property
    def test_data(self):
        return self._test_data

    def vertices_from_mesh(self, mesh_path: Path) -> paddle.Tensor:
        mesh = self.load_mesh(mesh_path)
        vertices = mesh.vertex.positions.numpy()
        return vertices

    def get_mesh_path(self, data_dir: Path, mesh_ind: int) -> Path:
        return data_dir / ("mesh_" + str(mesh_ind).zfill(3) + ".ply")

    def get_pressure_data_path(self, data_dir: Path, mesh_ind: int) -> Path:
        return data_dir / ("press_" + str(mesh_ind).zfill(3) + ".npy")

    def load_pressure(self, data_dir: Path, mesh_index: int) -> np.ndarray:
        press_path = self.get_pressure_data_path(data_dir, mesh_index)
        assert press_path.exists(), f"Pressure data does not exist: {press_path}"
        press = np.load(press_path).reshape((-1,)).astype(np.float32)
        press = np.concatenate((press[0:16], press[112:]), axis=0)
        return press

    def load_valid_mesh_indices(
        self, data_dir, filename="watertight_meshes.txt"
    ) -> List[int]:
        with open(data_dir / filename, "r") as fp:
            mesh_ind = fp.read().split("\n")
            mesh_ind = [int(a) for a in mesh_ind]
        return mesh_ind

    def load_mesh(self, mesh_path: Path) -> o3d.t.geometry.TriangleMesh:
        assert mesh_path.exists(), f"Mesh path does not exist: {mesh_path}"
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        return mesh

    def load_mesh_from_index(
        self, data_dir, mesh_index: int
    ) -> o3d.t.geometry.TriangleMesh:
        mesh_path = self.get_mesh_path(data_dir, mesh_index)
        return self.load_mesh(mesh_path)


class CFDSDFDataModule(CFDDataModule):
    def __init__(
        self,
        data_dir,
        n_train: int = 10,
        n_test: int = 5,
        spatial_resolution: Tuple[int, int, int] = None,
        query_points=None,
        eps=0.01,
        closest_points_to_query=True,
        test_data_dir=None,
    ):
        BaseDataModule.__init__(self)

        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
            test_data_dir = Path(test_data_dir)

        data_dir = data_dir.expanduser()
        test_data_dir = test_data_dir.expanduser()
        assert data_dir.exists(), "Path does not exist"
        assert data_dir.is_dir(), "Path is not a directory"
        self.data_dir = data_dir
        self.test_data_dir = test_data_dir

        min_bounds, max_bounds = self.load_bound(data_dir, eps=eps)
        valid_mesh_inds_train = self.load_valid_mesh_indices(data_dir)
        valid_mesh_inds_test = self.load_valid_mesh_indices(test_data_dir)
        valid_mesh_inds = valid_mesh_inds_train + valid_mesh_inds_test
        assert n_train + n_test <= len(valid_mesh_inds), "Not enough data"
        if n_train + n_test < len(valid_mesh_inds):
            warnings.warn(
                f"{len(valid_mesh_inds_train)} traning meshes are available, but n_train= {n_train} are requested."
            )
            warnings.warn(
                f"{len(valid_mesh_inds_test)} testing meshes are available, but n_train= {n_test} are requested."
            )

        train_indices = valid_mesh_inds_train[:n_train]
        test_indices = valid_mesh_inds_test[:n_test]
        train_mesh_paths = [self.get_mesh_path(data_dir, i) for i in train_indices]
        test_mesh_paths = [
            self.get_mesh_path(test_data_dir / "Feature_File", i) for i in test_indices
        ]
        self.test_mesh_paths = test_mesh_paths
        self.test_indices = test_indices
        if query_points is None:
            assert spatial_resolution is not None, "spatial_resolution must be given"
            tx = np.linspace(min_bounds[0], max_bounds[0], spatial_resolution[0])
            ty = np.linspace(min_bounds[1], max_bounds[1], spatial_resolution[1])
            tz = np.linspace(min_bounds[2], max_bounds[2], spatial_resolution[2])
            query_points = np.stack(
                np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1
            ).astype(np.float32)

        train_sdf_mesh_vertices = [
            self.sdf_vertices_closest_from_mesh(
                mesh_path, query_points, closest_points_to_query
            )
            for mesh_path in train_mesh_paths
        ]
        train_sdf = paddle.stack(
            x=[paddle.to_tensor(data=sdf) for sdf, _, _ in train_sdf_mesh_vertices]
        )
        train_vertices = paddle.stack(
            x=[
                paddle.to_tensor(data=vertices)
                for _, vertices, _ in train_sdf_mesh_vertices
            ]
        )
        if closest_points_to_query:
            train_closest_points = paddle.stack(
                x=[
                    paddle.to_tensor(data=closest)
                    for _, _, closest in train_sdf_mesh_vertices
                ]
            )
        else:
            train_closest_points = None
        del train_sdf_mesh_vertices
        train_pressure = paddle.stack(
            x=[
                paddle.to_tensor(
                    data=self.load_pressure(data_dir, mesh_index), dtype="float32"
                )
                for mesh_index in train_indices
            ]
        )
        min_bounds = paddle.to_tensor(data=min_bounds)
        max_bounds = paddle.to_tensor(data=max_bounds)
        train_vertices = self.location_normalization(
            train_vertices, min_bounds, max_bounds
        )
        if closest_points_to_query:
            train_closest_points = self.location_normalization(
                train_closest_points, min_bounds, max_bounds
            ).transpose(perm=[0, 4, 1, 2, 3])
        test_sdf_mesh_vertices = [
            self.sdf_vertices_closest_from_mesh(
                mesh_path, query_points, closest_points_to_query
            )
            for mesh_path in test_mesh_paths
        ]
        test_sdf = paddle.stack(
            x=[paddle.to_tensor(data=sdf) for sdf, _, _ in test_sdf_mesh_vertices]
        )
        test_vertices = paddle.stack(
            x=[
                paddle.to_tensor(data=vertices)
                for _, vertices, _ in test_sdf_mesh_vertices
            ]
        )

        if closest_points_to_query:
            test_closest_points = paddle.stack(
                x=[
                    paddle.to_tensor(data=closest)
                    for _, _, closest in test_sdf_mesh_vertices
                ]
            )
        else:
            test_closest_points = None
        del test_sdf_mesh_vertices

        # fake test pressure
        # print(f"fake pressure for {test_data_dir.name}")
        # for v, mesh_index in zip(test_vertices, test_indices):
        #     p_length = v.shape[0] + 96 # 96 = 112 - 16 for pressure truncation
        #     p_test_fake = np.ones((p_length,), dtype=np.float32)
        #     np.save(self.get_pressure_data_path(test_data_dir / "Label_file", mesh_index), p_test_fake)
        # import pdb;pdb.set_trace()
        test_pressure = paddle.stack(
            x=[
                paddle.to_tensor(
                    data=self.load_pressure(test_data_dir / "Label_file", mesh_index),
                    dtype="float32",
                )
                for mesh_index in test_indices
            ]
        )
        test_vertices = self.location_normalization(
            test_vertices, min_bounds, max_bounds
        )
        if closest_points_to_query:
            test_closest_points = self.location_normalization(
                test_closest_points, min_bounds, max_bounds
            ).transpose(perm=[0, 4, 1, 2, 3])
        pressure_normalization = UnitGaussianNormalizer(
            train_pressure, eps=1e-06, reduce_dim=(0, 1), verbose=False
        )

        mean, std = self.load_bound(
            data_dir, filename="train_pressure_min_std.txt", eps=0.0
        )
        pressure_normalization.mean, pressure_normalization.std = paddle.to_tensor(
            [mean[0]]
        ), paddle.to_tensor([std[0]])

        train_pressure = pressure_normalization.encode(train_pressure)
        test_pressure = pressure_normalization.encode(test_pressure)
        normalized_query_points = self.location_normalization(
            paddle.to_tensor(data=query_points), min_bounds, max_bounds
        ).transpose(perm=[3, 0, 1, 2])
        self._train_data = DictDatasetWithConstant(
            {"sdf": train_sdf, "vertices": train_vertices, "pressure": train_pressure},
            {"sdf_query_points": normalized_query_points},
        )

        self._test_data = DictDatasetWithConstant(
            {"sdf": test_sdf, "vertices": test_vertices, "pressure": test_pressure},
            {"sdf_query_points": normalized_query_points},
        )
        if closest_points_to_query:
            self._train_data.data_dict["closest_points"] = train_closest_points
            self._test_data.data_dict["closest_points"] = test_closest_points
        self.output_normalization = [
            pressure_normalization,
        ]

    def load_bound(
        self, data_dir, filename="watertight_global_bounds.txt", eps=1e-06
    ) -> Tuple[List[float], List[float]]:
        with open(data_dir / filename, "r") as fp:
            min_bounds = fp.readline().split(" ")
            max_bounds = fp.readline().split(" ")
            min_bounds = [(float(a) - eps) for a in min_bounds]
            max_bounds = [(float(a) + eps) for a in max_bounds]
        return min_bounds, max_bounds

    def location_normalization(
        self,
        locations: paddle.Tensor,
        min_bounds: paddle.Tensor,
        max_bounds: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Normalize locations to [-1, 1].
        """
        locations = (locations - min_bounds) / (max_bounds - min_bounds)
        locations = 2 * locations - 1
        return locations

    def compute_sdf(
        self, mesh: Union[Path, o3d.t.geometry.TriangleMesh], query_points
    ) -> np.ndarray:
        if isinstance(mesh, Path):
            mesh = self.load_mesh(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        signed_distance = scene.compute_signed_distance(query_points).numpy()
        return signed_distance

    def closest_points_to_query_from_mesh(
        self, mesh: o3d.t.geometry.TriangleMesh, query_points
    ) -> np.ndarray:
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        closest_points = scene.compute_closest_points(query_points)["points"].numpy()
        return closest_points

    def sdf_vertices_closest_from_mesh(
        self, mesh_path: Path, query_points: np.ndarray, closest_points: bool
    ) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
        mesh = self.load_mesh(mesh_path)
        sdf = self.compute_sdf(mesh, query_points)
        vertices = mesh.vertex.positions.numpy()
        if closest_points:
            closest_points = self.closest_points_to_query_from_mesh(mesh, query_points)
        else:
            closest_points = None
        return sdf, vertices, closest_points


class CarDataModule(CFDSDFDataModule):
    def __init__(
        self,
        data_dir,
        test_data_dir,
        n_train: int = 1,
        n_test: int = 1,
        spatial_resolution: Tuple[int, int, int] = None,
        query_points=None,
        eps=0.01,
        closest_points_to_query=True,
    ):
        BaseDataModule.__init__(self)
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
            test_data_dir = Path(test_data_dir)
        data_dir = data_dir.expanduser()
        test_data_dir = test_data_dir.expanduser()
        assert data_dir.exists(), "Path does not exist"
        assert data_dir.is_dir(), "Path is not a directory"
        self.data_dir = data_dir
        self.test_data_dir = test_data_dir
        min_bounds, max_bounds = self.load_bound(
            data_dir, filename="global_bounds.txt", eps=eps
        )
        min_info_bounds, max_info_bounds = self.load_bound(
            test_data_dir, filename="info_bounds.txt", eps=0.0
        )
        min_area_bound, max_area_bound = self.load_bound(
            test_data_dir, filename="area_bounds.txt", eps=0.0
        )
        assert n_train <= 500, "Not enough training data"
        assert n_test <= 51, "Not enough testing data"
        if n_train + n_test < 551:
            warnings.warn(
                f"551 meshes are available, but {n_train + n_test} are requested."
            )
        train_indices = np.loadtxt(data_dir / "train_index.txt", dtype=int)
        train_indices = train_indices[:n_train]
        test_indices = [(j + 1) for j in range(n_test)]
        self.test_indices = test_indices
        train_mesh_paths = [self.get_mesh_path(data_dir, "", i) for i in train_indices]
        test_mesh_paths = [
            self.get_mesh_path(test_data_dir, "", i) for i in test_indices
        ]
        self.test_mesh_paths = test_mesh_paths

        if query_points is None:
            assert spatial_resolution is not None, "spatial_resolution must be given"
            tx = np.linspace(min_bounds[0], max_bounds[0], spatial_resolution[0])
            ty = np.linspace(min_bounds[1], max_bounds[1], spatial_resolution[1])
            tz = np.linspace(min_bounds[2], max_bounds[2], spatial_resolution[2])
            query_points = np.stack(
                np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1
            ).astype(np.float32)
        train_df_closest = [
            self.df_from_mesh(mesh_path, query_points, closest_points_to_query)
            for mesh_path in train_mesh_paths
        ]
        train_sdf = paddle.stack(
            x=[paddle.to_tensor(data=df) for df, _ in train_df_closest]
        )

        if closest_points_to_query:
            train_closest_points = paddle.stack(
                x=[paddle.to_tensor(data=closest) for _, closest in train_df_closest]
            )
        else:
            train_closest_points = None

        del train_df_closest
        test_df_closest = [
            self.df_from_mesh(mesh_path, query_points, closest_points_to_query)
            for mesh_path in test_mesh_paths
        ]
        test_sdf = paddle.stack(
            x=[paddle.to_tensor(data=df) for df, _ in test_df_closest]
        )

        if closest_points_to_query:
            test_closest_points = paddle.stack(
                x=[paddle.to_tensor(data=closest) for _, closest in test_df_closest]
            )
        else:
            test_closest_points = None
        del test_df_closest
        # fake pressure for normalization
        train_pressure = paddle.stack(
            x=[
                paddle.to_tensor(
                    data=self.load_pressure(data_dir, "", train_indices[0]),
                    dtype="float32",
                )
            ]
        )

        self.pressure_normalization = UnitGaussianNormalizer(
            train_pressure, eps=1e-06, reduce_dim=(0), verbose=False
        )

        if n_train != 500:
            mean, std = self.load_bound(
                data_dir, filename="train_pressure_mean_std.txt", eps=0.0
            )
            self.pressure_normalization.mean, self.pressure_normalization.std = (
                mean[0],
                std[0],
            )

        self.min_bounds = paddle.to_tensor(data=min_bounds)
        self.max_bounds = paddle.to_tensor(data=max_bounds)
        normalized_query_points = self.location_normalization(
            paddle.to_tensor(data=query_points), self.min_bounds, self.max_bounds
        ).transpose(perm=[3, 0, 1, 2])
        if closest_points_to_query:
            train_closest_points = self.location_normalization(
                train_closest_points, self.min_bounds, self.max_bounds
            ).transpose(perm=[0, 4, 1, 2, 3])
            test_closest_points = self.location_normalization(
                test_closest_points, self.min_bounds, self.max_bounds
            ).transpose(perm=[0, 4, 1, 2, 3])

        self._train_data = DictDatasetWithConstant(
            {"sdf": train_sdf},
            {
                "sdf_query_points": normalized_query_points,
                "data_dir": data_dir,
                "data_module": self,
                "mesh_index": train_indices,
            },
        )

        self._test_data = DictDatasetWithConstant(
            {"sdf": test_sdf},
            {
                "sdf_query_points": normalized_query_points,
                "data_dir": test_data_dir,
                "data_module": self,
                "mesh_index": test_indices,
            },
        )

        if closest_points_to_query:
            # self._train_data.data_dict["closest_points"] = train_closest_points
            self._test_data.data_dict["closest_points"] = test_closest_points
        self._aggregatable = ["sdf", "closest_points", "sdf_query_points"]
        self.output_normalization = [self.pressure_normalization]

    def get_mesh_path(self, data_dir: Path, subfolder: str, mesh_ind: int) -> Path:
        return data_dir / subfolder / ("mesh_" + str(mesh_ind).zfill(4) + ".ply")

    def get_pressure_data_path(
        self, data_dir: Path, subfolder: str, mesh_ind: int
    ) -> Path:
        return data_dir / subfolder / ("press_" + str(mesh_ind).zfill(4) + ".npy")

    def get_wss_data_path(self, data_dir: Path, subfolder: str, mesh_ind: int) -> Path:
        return (
            data_dir
            / subfolder
            / ("wallshearstress_" + str(mesh_ind).zfill(4) + ".npy")
        )

    def load_wss(self, data_dir: Path, subfolder: str, mesh_index: int) -> np.ndarray:
        wss_path = self.get_wss_data_path(data_dir, subfolder, mesh_index)
        assert wss_path.exists(), "wallshearstress data does not exist"
        wss = np.load(wss_path).astype(np.float32)
        return wss

    def load_pressure(
        self, data_dir: Path, subfolder: str, mesh_index: int
    ) -> np.ndarray:
        press_path = self.get_pressure_data_path(data_dir, subfolder, mesh_index)
        assert press_path.exists(), "Pressure data does not exist"
        press = np.load(press_path).reshape((-1,)).astype(np.float32)
        return press

    def load_centroid(
        self, data_dir: Path, subfolder: str, mesh_index: int
    ) -> np.ndarray:
        centroid_path = (
            data_dir / subfolder / ("centroid_" + str(mesh_index).zfill(4) + ".npy")
        )
        assert centroid_path.exists(), "Centroid data does not exist"
        centroid = np.load(centroid_path).reshape((1, -1, 3)).astype(np.float32)
        return centroid

    def compute_df(
        self, mesh: Union[Path, o3d.t.geometry.TriangleMesh], query_points
    ) -> np.ndarray:
        if isinstance(mesh, Path):
            mesh = self.load_mesh(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        distance = scene.compute_distance(query_points).numpy()
        return distance

    def df_from_mesh(
        self, mesh_path: Path, query_points: np.ndarray, closest_points: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        mesh = self.load_mesh(mesh_path)
        df = self.compute_df(mesh, query_points)

        if closest_points:
            closest_points = self.closest_points_to_query_from_mesh(mesh, query_points)
        else:
            closest_points = None
        return df, closest_points

    def info_normalization(
        self, info: dict, min_bounds: List[float], max_bounds: List[float]
    ) -> dict:
        """
        Normalize info to [0, 1].
        """
        for i, (k, v) in enumerate(info.items()):
            info[k] = (v - min_bounds[i]) / (max_bounds[i] - min_bounds[i])
        return info

    def area_normalization(
        self, area: paddle.Tensor, min_bounds: float, max_bounds: float
    ) -> paddle.Tensor:
        """
        Normalize info to [0, 1].
        """
        return (area - min_bounds) / (max_bounds - min_bounds)

    def wss_normalization(
        self,
        area: paddle.Tensor,
        min_bounds,
        max_bounds,
    ) -> paddle.Tensor:
        """
        Normalize info to [0, 1].
        """
        return (area - min_bounds) / (max_bounds - min_bounds)

    def collate_fn(self, batch):
        aggr_dict = {}
        for key in self._aggregatable:
            aggr_dict.update(
                {key: paddle.stack(x=[data_dict[key] for data_dict in batch])}
            )
        remaining = list(set(batch[0].keys()) - set(self._aggregatable))
        for key in remaining:
            new_mini_batch_list = [data_dict[key] for data_dict in batch]
            if len(new_mini_batch_list) == 1:
                aggr_dict.update({key: new_mini_batch_list[0]})
            else:
                aggr_dict.update({key: new_mini_batch_list})

                # TODO for competitor : because centroid is not the same length, so a padding strategy may be needed.
                raise NotImplementedError(
                    "Not implemented for more than one element in the batch."
                )
        return aggr_dict
