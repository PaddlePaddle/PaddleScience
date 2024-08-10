import argparse
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
import threading

import h5py
import numpy as np
import paddle
import trimesh
import utils.DS_utils as DS_utils
from paddle_aux import scatter_paddle
from utils.knn import knn_graph
from utils.knn import knn_scipy_batched
from utils.utilities import calc_cell_centered_with_node_attr
from utils.utilities import calc_node_centered_with_cell_attr

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)


sys.stdout.flush()
lock = threading.Lock()


class Basemanager:
    def polygon_area(self, vertices):
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def triangles_to_faces(self, faces):
        """Computes mesh edges from triangles."""
        cells_face_node = paddle.concat(
            x=(
                faces[:, 0:2],
                faces[:, 1:3],
                paddle.stack(x=(faces[:, 2], faces[:, 0]), axis=1),
            ),
            axis=0,
        )
        return cells_face_node.numpy()

    def position_relative_to_line_paddle(A, B, angle_c):
        A = paddle.to_tensor(data=A, dtype="float64")
        B = paddle.to_tensor(data=B, dtype="float64")
        angle_c = paddle.to_tensor(data=angle_c, dtype="float64")
        direction_vector = paddle.to_tensor(
            data=[
                paddle.cos(x=angle_c * math.pi / 180.0),
                paddle.sin(x=angle_c * math.pi / 180.0),
            ],
            dtype="float64",
        )
        vector_AB = B - A
        cross_product = (
            direction_vector[0] * vector_AB[:, 1]
            - direction_vector[1] * vector_AB[:, 0]
        )
        mask = cross_product > 0
        return mask.view(-1, 1)

    def is_convex(self, polygon):
        n = len(polygon)
        for i in range(n):
            a = polygon[i]
            b = polygon[(i + 1) % n]
            c = polygon[(i + 2) % n]
            ba = a - b
            bc = c - b
            cross_product = np.cross(ba, bc)
            if cross_product < 0:
                return False
        return True

    def reorder_polygon(self, polygon):
        centroid = np.mean(polygon, axis=0)
        sorted_polygon = sorted(
            polygon, key=lambda p: np.arctan2(p[1] - centroid[1], p[0] - centroid[0])
        )
        return np.array(sorted_polygon)

    def ensure_counterclockwise(self, cells, mesh_pos):
        for i, cell in enumerate(cells):
            vertices = mesh_pos[cell]
            if not self.is_convex(vertices):
                vertices = self.reorder_polygon(vertices)
                sorted_indices = sorted(
                    range(len(cell)),
                    key=lambda k: list(map(list, vertices)).index(
                        list(mesh_pos[cell][k])
                    ),
                )
                cells[i] = np.array(cell)[sorted_indices]
        return cell

    def is_equal(self, x, pivot):
        """
        Determine if a value x is between two other values a and b.

        Parameters:
        - a (float or int): The lower bound.
        - b (float or int): The upper bound.
        - x (float or int): The value to check.

        Returns:
        - (bool): True if x is between a and b (inclusive), False otherwise.
        """
        a = abs(pivot) - float(1e-08)
        b = abs(pivot) + float(1e-08)
        if a <= abs(x) <= b:
            return True
        else:
            return False

    def make_edges_unique(self, cells_face_node, cells_node, cells_index):
        """Computes mesh edges from triangles."""
        cells_face_node_biased = (
            paddle.sort(x=cells_face_node, axis=1),
            paddle.argsort(x=cells_face_node, axis=1),
        )[0]
        senders, receivers = cells_face_node_biased[:, 0], cells_face_node_biased[:, 1]
        packed_edges = paddle.stack(x=(senders, receivers), axis=1)
        singleway_edge_index = paddle.unique(
            x=packed_edges, return_inverse=False, return_counts=False, axis=0
        ).to("int64")
        cells_face = []
        edge_indice = paddle.arange(end=tuple(singleway_edge_index.shape)[0])
        for i_edge in range(tuple(cells_face_node.shape)[0]):
            current_edge = (
                paddle.sort(x=cells_face_node[i_edge : i_edge + 1, :], axis=-1),
                paddle.argsort(x=cells_face_node[i_edge : i_edge + 1, :], axis=-1),
            )[0]
            mask = (singleway_edge_index == current_edge).astype("bool").all(axis=-1)
            cells_face.append(edge_indice[mask])
        cells_face = paddle.concat(x=cells_face).view(-1, 1)
        if tuple(cells_face.shape)[0] != tuple(cells_face_node.shape)[0]:
            raise ValueError("cells_face shape is not equal to cells_face_node shape")
        return {
            "edge_index": singleway_edge_index,
            "cells_face": cells_face,
            "cells_face_node_unbiased": cells_face_node,
            "cells_face_node_biased": packed_edges,
        }

    def create_neighbor_matrix(self, vertex_coords, edges):
        """
        Create a matrix representing the neighbors for each vertex in a graph.

        Parameters:
        vertex_coords (Tensor): A tensor of shape [n, 2] representing n vertex coordinates.
        edges (Tensor): A tensor of shape [m, 2] representing m edges,
        where each edge is a pair of vertex indices.

        Returns:
        Tensor: A matrix where each row corresponds to a vertex and contains the indices of its neighbors.
        """
        edges_mod = edges % tuple(vertex_coords.shape)[0]
        counts = paddle.zeros(shape=tuple(vertex_coords.shape)[0], dtype="int64")
        counts.put_along_axis_(
            axis=0,
            indices=edges_mod.view(-1),
            values=paddle.ones_like(x=edges_mod.view(-1)),
            reduce="add",
        )
        max_neighbors = counts.max()
        neighbor_matrix = paddle.full(
            shape=(tuple(vertex_coords.shape)[0], max_neighbors),
            fill_value=-1,
            dtype="int64",
        )
        current_count = paddle.zeros(shape=tuple(vertex_coords.shape)[0], dtype="int64")
        for edge in edges_mod:
            start, end = edge
            neighbor_matrix[start, current_count[start]] = end
            current_count[start] += 1
            neighbor_matrix[end, current_count[end]] = start
            current_count[end] += 1
        return neighbor_matrix, max_neighbors

    def generate_directed_edges(self, cells_node):
        edges = []
        for i in range(len(cells_node)):
            for j in range(i + 1, len(cells_node)):
                edge = [cells_node[i], cells_node[j]]
                reversed_edge = [cells_node[j], cells_node[i]]
                if reversed_edge not in edges:
                    edges.append(edge)
        return edges

    def compose_edge_index_x(
        self, face_node, cells_face_node_biased, cells_node, cells_index
    ):
        face_node_x = face_node.clone()
        for i in range(cells_index.max() + 1):
            mask_cell = (cells_index == i).view(-1)
            current_cells_face_node_biased = cells_face_node_biased[mask_cell]
            current_cells_node = cells_node[mask_cell]
            all_possible_edges, _ = paddle.sort(
                x=paddle.to_tensor(
                    data=self.generate_directed_edges(current_cells_node)
                ),
                axis=-1,
            ), paddle.argsort(
                x=paddle.to_tensor(
                    data=self.generate_directed_edges(current_cells_node)
                ),
                axis=-1,
            )
            for edge in all_possible_edges:
                edge = edge.unsqueeze(axis=0)
                if (edge.unsqueeze(axis=0) == current_cells_face_node_biased).astype(
                    "bool"
                ).all(axis=-1).sum() < 1:
                    face_node_x = paddle.concat(x=(face_node_x, edge), axis=0)
        return face_node_x

    def convert_to_tensors(self, input_dict):
        if isinstance(input_dict, dict):
            for key in input_dict.keys():
                value = input_dict[key]
                if isinstance(value, np.ndarray):
                    input_dict[key] = paddle.to_tensor(data=value)
                elif not isinstance(value, paddle.Tensor):
                    input_dict[key] = paddle.to_tensor(data=value)
        elif isinstance(input_dict, list):
            for i in range(len(input_dict)):
                value = input_dict[i]
                if isinstance(value, np.ndarray):
                    input_dict[i] = paddle.to_tensor(data=value)
                elif not isinstance(value, paddle.Tensor):
                    input_dict[i] = paddle.to_tensor(data=value)
        return input_dict

    def convert_to_numpy(self, input_dict):
        if isinstance(input_dict, dict):
            for key in input_dict.keys():
                value = input_dict[key]
                if isinstance(value, paddle.Tensor):
                    input_dict[key] = value.numpy()
                elif not isinstance(value, paddle.Tensor):
                    input_dict[key] = paddle.to_tensor(data=value).numpy()
        elif isinstance(input_dict, list):
            for i in range(len(input)):
                value = input_dict[i]
                if isinstance(value, paddle.Tensor):
                    input_dict[i] = value.numpy()
                elif not isinstance(value, paddle.Tensor):
                    input_dict[i] = paddle.to_tensor(data=value).numpy()
        return input_dict

    def compute_unit_normals(
        self,
        mesh_pos: paddle.Tensor,
        cells_node: paddle.Tensor,
        centroid: paddle.Tensor = None,
    ):
        cells_node = cells_node.reshape(-1, 3)
        A = mesh_pos[cells_node[:, 0]]
        B = mesh_pos[cells_node[:, 1]]
        C = mesh_pos[cells_node[:, 2]]
        AB = B - A
        AC = C - A
        N = paddle.cross(x=AB, y=AC, axis=-1)
        norm = paddle.linalg.norm(x=N, axis=-1, keepdim=True)
        unit_N = N / norm
        geo_center = paddle.mean(x=centroid, axis=0, keepdim=True)
        outward = centroid - geo_center
        mask_outward = (unit_N * outward).sum(axis=-1, keepdim=True) > 0
        unit_N = paddle.where(condition=mask_outward.repeat(1, 3), x=unit_N, y=-unit_N)
        return unit_N


class PlyMesh(Basemanager):
    """
    Tecplot .dat file is only supported with Tobias`s airfoil dataset ,No more data file supported
    """

    def __init__(self, path=None):
        mesh_pos, cells_node = DS_utils.load_mesh_ply_vtk(path["mesh_file_path"])
        self.mesh_pos = mesh_pos
        self.cells_node = cells_node
        cells_face_node = self.triangles_to_faces(paddle.to_tensor(data=cells_node))
        cells_index = (
            paddle.arange(end=tuple(cells_node.shape)[0])
            .view(-1, 1)
            .repeat(1, 3)
            .numpy()
        )
        try:
            pressuredata = np.expand_dims(np.load(path["data_file_path"]), axis=1)
        except Exception:
            pressuredata = np.zeros((tuple(mesh_pos.shape)[0], 1), dtype=np.float32)
        if tuple(mesh_pos.shape)[0] < 10000:
            self.mesh_info = {
                "node|pos": mesh_pos,
                "cell|cells_node": cells_node,
                "cells_node": cells_node.reshape(-1, 1),
                "cells_index": cells_index.reshape(-1, 1),
                "cells_face_node": cells_face_node,
                "node|pressure": np.concatenate(
                    (pressuredata[0:16], pressuredata[112:]), axis=0
                ),
            }
        else:
            mesh_idx = (
                path["mesh_file_path"].rsplit("/")[-1].split("_")[1].split(".")[0]
            )
            if path["split"] == "train":
                centroid = np.load(
                    os.path.join(path["label_dir"], f"centroid_{mesh_idx}.npy")
                )
            else:
                infer_dir = path["mesh_file_path"].rsplit("/", maxsplit=1)[0]
                centroid = np.load(os.path.join(infer_dir, f"centroid_{mesh_idx}.npy"))
            self.mesh_info = {
                "node|pos": mesh_pos,
                "cell|cells_node": cells_node,
                "cells_node": cells_node.reshape(-1, 1),
                "cells_index": cells_index.reshape(-1, 1),
                "cells_face_node": cells_face_node,
                "cell|pressure": pressuredata,
                "cell|centroid": centroid,
            }
        self.path = path

    def extract_mesh_A(self, data_index=None):
        """
        all input dataset values should be paddle tensor object
        """
        dataset = self.convert_to_tensors(self.mesh_info)
        cells_node = dataset["cells_node"][:, 0]
        cells_index = dataset["cells_index"][:, 0]
        """>>>compute centroid crds>>>"""
        mesh_pos = dataset["node|pos"]
        centroid = calc_cell_centered_with_node_attr(
            node_attr=dataset["node|pos"],
            cells_node=cells_node,
            cells_index=cells_index,
            reduce="mean",
        )
        dataset["centroid"] = centroid
        """<<<compute centroid crds<<<"""
        """ >>>   compose face  and face_center_pos >>> """
        decomposed_cells = self.make_edges_unique(
            dataset["cells_face_node"], cells_node.view(-1, 1), cells_index.view(-1, 1)
        )
        cells_face_node = decomposed_cells["cells_face_node_biased"]
        cells_face = decomposed_cells["cells_face"]
        dataset["cells_face"] = cells_face
        face_node = decomposed_cells["edge_index"].T
        dataset["face_node"] = face_node
        face_center_pos = (mesh_pos[face_node[0]] + mesh_pos[face_node[1]]) / 2.0
        dataset["face_center_pos"] = face_center_pos
        """ <<<   compose face   <<< """
        """ >>>   compute face length   >>>"""
        face_length = paddle.linalg.norm(
            x=mesh_pos[face_node[0]] - mesh_pos[face_node[1]], axis=1, keepdim=True
        )
        dataset["face_length"] = face_length
        """ <<<   compute face length   <<<"""
        """ >>> compute cells_face and neighbor_cell >>> """
        senders_cell = calc_node_centered_with_cell_attr(
            cell_attr=cells_index.view(-1),
            cells_node=cells_face.view(-1),
            cells_index=cells_index.view(-1),
            reduce="max",
            map=False,
        )
        recivers_cell = calc_node_centered_with_cell_attr(
            cell_attr=cells_index.view(-1),
            cells_node=cells_face.view(-1),
            cells_index=cells_index.view(-1),
            reduce="min",
            map=False,
        )
        neighbour_cell = paddle.stack(x=(recivers_cell, senders_cell), axis=0)
        dataset["neighbour_cell"] = neighbour_cell.to("int64")
        """ <<< compute cells_face and neighbor_cell <<< """
        """ >>> compute cell_area >>> """
        cells_node_reshape = cells_node.reshape(-1, 3)
        cells_face_node = paddle.stack(
            x=(
                cells_node_reshape[:, 0:2],
                cells_node_reshape[:, 1:3],
                paddle.stack(
                    x=(cells_node_reshape[:, 2], cells_node_reshape[:, 0]), axis=1
                ),
            ),
            axis=1,
        )
        cells_length = paddle.linalg.norm(
            x=mesh_pos[cells_face_node[:, :, 0]] - mesh_pos[cells_face_node[:, :, 1]],
            axis=-1,
            keepdim=True,
        )
        circum = cells_length.sum(axis=1, keepdim=True) * 0.5
        mul = (
            circum[:, 0]
            * (circum - cells_length)[:, 0]
            * (circum - cells_length)[:, 1]
            * (circum - cells_length)[:, 2]
        )
        valid_cells_area = paddle.sqrt(x=mul)
        dataset["cells_area"] = valid_cells_area
        """ <<< compute cell_area <<< """
        """ >>> unit normal vector >>> """
        unv = self.compute_unit_normals(mesh_pos, cells_node, centroid=centroid)
        node_unv = calc_node_centered_with_cell_attr(
            cell_attr=unv[cells_index],
            cells_node=cells_node.view(-1, 1),
            cells_index=cells_index.view(-1, 1),
            reduce="mean",
            map=False,
        )
        dataset["unit_norm_v"] = node_unv
        """ <<< unit normal vector <<< """
        bounds = np.loadtxt(
            os.path.join(self.path["aux_dir"], "watertight_global_bounds.txt")
        )
        pos = dataset["node|pos"]
        grid, sdf = DS_utils.compute_sdf_grid(
            pos, dataset["cells_node"].reshape(-1, 3), bounds, [64, 64, 64]
        )
        ply_file = self.path["mesh_file_path"]
        ao = DS_utils.compute_ao(ply_file)
        dataset["node|ao"] = ao
        output_dict = {
            "node|pos": dataset["node|pos"],
            "node|pressure": dataset["node|pressure"],
            "node|unit_norm_v": dataset["unit_norm_v"],
            "node|ao": dataset["node|ao"],
            "face|face_node": dataset["face_node"],
            "face|face_center_pos": dataset["face_center_pos"],
            "face|face_length": dataset["face_length"],
            "face|neighbour_cell": dataset["neighbour_cell"],
            "cell|cells_area": dataset["cells_area"],
            "cell|centroid": dataset["centroid"],
            "cells_node": dataset["cells_node"],
            "cells_index": dataset["cells_index"],
            "cells_face": dataset["cells_face"],
            "voxel|grid": grid,
            "voxel|sdf": sdf[:, None],
        }
        h5_dataset = output_dict
        print("{0}th mesh has been extracted".format(data_index))
        return h5_dataset

    def extract_mesh_B(self, data_index=None):
        """
        all input dataset values should be paddle tensor object
        """
        dataset = self.convert_to_tensors(self.mesh_info)
        car_model = trimesh.load(self.path["mesh_file_path"], force="mesh")
        vertices = car_model.vertices
        normals = car_model.vertex_normals
        faces = car_model.faces
        _, cells_node = vertices, faces
        self.triangles_to_faces(paddle.to_tensor(data=cells_node))
        (paddle.arange(end=tuple(cells_node.shape)[0]).view(-1, 1).repeat(1, 3).numpy())
        areadata = np.zeros([tuple(dataset["cell|centroid"].shape)[0], 1])
        centroiddata = dataset["cell|centroid"]
        centroid = centroiddata
        device = "cuda:0" if paddle.device.cuda.device_count() >= 1 else "cpu"
        normals_cuda = paddle.to_tensor(data=normals).to("float32").to(device)
        points_cuda = paddle.to_tensor(data=vertices).to("float32").to(device)
        centroid_cuda = centroiddata.to("float32").to(device)
        knn_idx = knn_scipy_batched(points_cuda, centroid_cuda, 4)
        centroid_normals = paddle.full(
            shape=[tuple(centroid.shape)[0], 3], fill_value=0.0
        )
        centroid_normals = scatter_paddle(
            normals_cuda[knn_idx[1]], index=knn_idx[0], dim=0, reduce="mean"
        ).cpu()
        factor = paddle.linalg.norm(x=centroid_normals, axis=-1, keepdim=True)
        centroid_normals = centroid_normals / factor
        centroid_normals = centroid_normals.cpu().numpy()
        bounds = np.loadtxt(os.path.join(path["aux_dir"], "global_bounds.txt"))
        pos = dataset["node|pos"]
        grid, sdf = DS_utils.compute_sdf_grid(
            pos, dataset["cells_node"].reshape(-1, 3), bounds, [64, 64, 64]
        )
        pos_tensor = paddle.to_tensor(data=centroid, dtype="float32")
        edge_index, _ = knn_graph(pos_tensor, k=4).sort(dim=0)
        edge_index = paddle.unique(x=edge_index, axis=1)
        edge_index_np = edge_index.numpy()
        output_dict = {
            "node|pos": dataset["node|pos"].cpu().numpy(),
            "cell|cells_area": areadata,
            "cell|centroid": centroid.cpu().numpy(),
            "cells_node": dataset["cells_node"].cpu().numpy(),
            "cell|unit_norm_v": centroid_normals,
            "cell|pressure": dataset["cell|pressure"].cpu().numpy(),
            "voxel|grid": grid,
            "voxel|sdf": sdf,
            "face|neighbour_cell": edge_index_np,
        }
        h5_dataset = output_dict
        print("{0}th mesh has been extracted".format(data_index))
        return h5_dataset


def random_samples_no_replacement(arr, num_samples, num_iterations):
    if num_samples * num_iterations > len(arr):
        raise ValueError(
            "Number of samples multiplied by iterations cannot be greater than the length of the array."
        )
    samples = []
    arr_copy = arr.copy()
    for _ in range(num_iterations):
        sample_indices = np.random.choice(len(arr_copy), num_samples, replace=False)
        sample = arr_copy[sample_indices]
        samples.append(sample)
        arr_copy = np.delete(arr_copy, sample_indices)
    return samples, arr_copy


def process_file(file_index, file_path, path, queue):
    file_name = os.path.basename(file_path)
    mesh_name = file_name
    path["mesh_file_path"] = file_path
    if path["mesh_file_path"].endswith("ply"):
        mesh_index = int("".join(char for char in mesh_name if char.isdigit()))
        data_name = f"press_{''.join(char for char in mesh_name if char.isdigit())}.npy"
        data_file_path = f"{path['label_dir']}/{data_name}"
        path["mesh_file_path"] = file_path
        path["data_file_path"] = data_file_path
        data = PlyMesh(path=path)
        if data.mesh_pos.shape[0] < 10000:
            h5_data = data.extract_mesh_A(data_index=mesh_index)
        else:
            h5_data = data.extract_mesh_B(data_index=mesh_index)
    else:
        raise ValueError(f"wrong mesh file at {path['mesh_file_path']}")
    queue.put((h5_data, mesh_index))


def string_to_floats(s):
    return np.asarray([float(ord(c)) for c in s])


def floats_to_string(floats):
    return "".join([chr(int(f)) for f in floats])


def writer_process(queue, split, path):
    os.makedirs(path["h5_save_path"], exist_ok=True)
    h5_writer = h5py.File(f"{path['h5_save_path']}/{split}.h5", "w")
    sdf_list = []
    while True:
        h5_data, file_index = queue.get()
        if h5_data is None:
            break
        if str(file_index) in h5_writer:
            continue
        current_traj = h5_writer.create_group(str(file_index))
        for key, value in h5_data.items():
            current_traj.create_dataset(key, data=value)
            if key == "voxel|sdf":
                sdf_list.append(paddle.to_tensor(data=value))
        print("{0}th mesh has been writed".format(file_index))
    if split == "train":
        voxel_mean, voxel_std = DS_utils.compute_mean_std(sdf_list)
        np.savetxt(
            f"{path['h5_save_path']}/voxel_mean_std.txt",
            np.array([voxel_mean.item(), voxel_std.item()]),
        )
        if path["track"] == "A":
            shutil.copy(
                f"{path['aux_dir']}/train_pressure_min_std.txt",
                f"{path['h5_save_path']}/train_pressure_min_std.txt",
            )
            shutil.copy(
                f"{path['aux_dir']}/watertight_global_bounds.txt",
                f"{path['h5_save_path']}/watertight_global_bounds.txt",
            )
        else:
            shutil.copy(
                f"{path['aux_dir']}/train_pressure_mean_std.txt",
                f"{path['h5_save_path']}/train_pressure_mean_std.txt",
            )
            shutil.copy(
                f"{path['aux_dir']}/global_bounds.txt",
                f"{path['h5_save_path']}/global_bounds.txt",
            )
    h5_writer.close()


def run_command(tfrecord_file, idx_file):
    subprocess.run(
        ["python", "-m", "tfrecord.tools.tfrecord2idx", tfrecord_file, idx_file],
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train / test a paddle model to predict frames"
    )
    parser.add_argument(
        "--msh_dir",
        default="/home/xiaoli/project/3D-ShapeNet-car/src/Dataset/rawDataset/trackB/Train/Feature/dataset_3/train_mesh_0603",
        type=str,
        help="",
    )
    parser.add_argument(
        "--label_dir",
        default="/home/xiaoli/project/3D-ShapeNet-car/src/Dataset/rawDataset/trackB/Train/Label/dataset_2",
        type=str,
        help="",
    )
    parser.add_argument(
        "--aux_dir",
        default="/home/xiaoli/project/3D-ShapeNet-car/src/Dataset/rawDataset/trackB/Test/Testset_track_B/Auxiliary",
        type=str,
        help="",
    )
    parser.add_argument(
        "--h5_save_path",
        default="/lvm_data/litianyu/mycode-new/3D-ShapeNet-car/src/Dataset/converted_dataset_test/trackB",
        type=str,
        help="",
    )
    parser.add_argument("--split", default="test", type=str, help="")
    parser.add_argument("--track", default="B", type=str, help="")
    params = parser.parse_args()
    debug_file_path = None
    path = {
        "msh_dir": params.msh_dir,
        "label_dir": params.label_dir,
        "aux_dir": params.aux_dir,
        "h5_save_path": params.h5_save_path,
        "split": params.split,
        "track": params.track,
        "plot": False,
    }
    os.makedirs(path["h5_save_path"], exist_ok=True)
    total_samples = 0
    file_paths_list = []
    for subdir, _, files in os.walk(path["msh_dir"]):
        for data_name in files:
            if data_name.endswith(".ply"):
                file_paths_list.append(os.path.join(subdir, data_name))
    np.random.shuffle(file_paths_list)
    print(f"Total samples: {len(file_paths_list)}")
    if debug_file_path is not None:
        multi_process = 1
    elif len(file_paths_list) < multiprocessing.cpu_count():
        multi_process = len(file_paths_list)
    else:
        multi_process = int(multiprocessing.cpu_count() / 2)
    global_data_index = 0
    with multiprocessing.Pool(multi_process) as pool:
        manager = multiprocessing.Manager()
        queue = manager.Queue()
        writer_proc = multiprocessing.Process(
            target=writer_process, args=(queue, params.split, path)
        )
        writer_proc.start()
        if debug_file_path is not None:
            file_path = debug_file_path
            results = [pool.apply_async(process_file, args=(0, file_path, path, queue))]
        else:
            results = [
                pool.apply_async(
                    process_file, args=(file_index, file_path, path, queue)
                )
                for file_index, file_path in enumerate(file_paths_list)
            ]
        for res in results:
            res.get()
        queue.put((None, None))
        writer_proc.join()
    print("Fininsh parsing train dataset calc mean and std")
