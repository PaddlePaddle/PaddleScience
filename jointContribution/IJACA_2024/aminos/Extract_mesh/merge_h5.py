import argparse
import os

import h5py
import numpy as np
import paddle


def load_ds_trackA_info(file_path, key_list):
    path_trackA_ds = file_path
    key_list = np.sort([int(key) for key in key_list])
    key_list = [str(key) for key in key_list]
    bounds = np.loadtxt(path_trackA_ds + "/watertight_global_bounds.txt")
    pressure_mean_std = paddle.to_tensor(
        data=np.loadtxt(path_trackA_ds + "/train_pressure_min_std.txt")
    ).to("float32")
    voxel_mean_std = paddle.to_tensor(
        data=np.loadtxt(path_trackA_ds + "/voxel_mean_std.txt")
    ).to("float32")
    pos_mean_std = np.loadtxt(path_trackA_ds + "/pos_mean_std.txt")
    normal_mean_std = np.loadtxt(path_trackA_ds + "/normal_mean_std.txt")
    PN_mean_std = paddle.to_tensor(
        data=np.concatenate([pos_mean_std, normal_mean_std], axis=-1)
    ).to("float32")
    physics_info = {
        "key_list": key_list,
        "bounds": bounds,
        "voxel_mean_std": voxel_mean_std,
        "pressure_mean_std": pressure_mean_std,
        "PN_mean_std": PN_mean_std,
    }
    return physics_info


def load_ds_trackB_info(file_path, key_list):
    path_trackB_ds = file_path
    key_list = np.sort([int(key) for key in key_list])
    key_list = [str(key) for key in key_list]
    pressure_mean_std = paddle.to_tensor(
        data=np.loadtxt(path_trackB_ds + "/train_pressure_mean_std.txt")
    ).to("float32")
    bounds = np.loadtxt(path_trackB_ds + "/global_bounds.txt")
    voxel_mean_std = paddle.to_tensor(
        data=np.loadtxt(path_trackB_ds + "/voxel_mean_std.txt")
    ).to("float32")
    PNA_mean_std = paddle.to_tensor(
        data=np.loadtxt(path_trackB_ds + "/PosNormalArea_mean_std.txt")
    ).to("float32")
    PN_mean_std = PNA_mean_std[:, :6]
    physics_info = {
        "key_list": key_list,
        "bounds": bounds,
        "voxel_mean_std": voxel_mean_std,
        "pressure_mean_std": pressure_mean_std,
        "PN_mean_std": PN_mean_std,
    }
    return physics_info


def load_extra_info(file_path, key_list, track_type="A"):
    if track_type == "A":
        physics_info = load_ds_trackA_info(file_path, key_list)
    else:
        physics_info = load_ds_trackB_info(file_path, key_list)
    return physics_info


def add_physics_info_to_group(group, physics_info):
    for key, value in physics_info.items():
        group.create_dataset(key, data=value)


def merge_h5_files(fileA_path, fileB_path, merged_file_path):
    with h5py.File(fileA_path, "r") as fileA, h5py.File(
        fileB_path, "r"
    ) as fileB, h5py.File(merged_file_path, "w") as merged_file:
        key_list_A = list(fileA.keys())
        key_list_B = list(fileB.keys())
        physics_info_A = load_extra_info(
            os.path.dirname(fileA_path), key_list_A, track_type="A"
        )
        physics_info_B = load_extra_info(
            os.path.dirname(fileB_path), key_list_B, track_type="B"
        )
        for key in fileA.keys():
            group = fileA[key]
            new_key = "A_" + key
            merged_file.copy(group, new_key)
            add_physics_info_to_group(merged_file[new_key], physics_info_A)
        for key in fileB.keys():
            group = fileB[key]
            new_key = "B_" + key
            merged_file.copy(group, new_key)
            add_physics_info_to_group(merged_file[new_key], physics_info_B)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train / test a paddle model to predict frames"
    )
    parser.add_argument(
        "--A_dir",
        default="/home/xiaoli/project/3D-ShapeNet-car/src/Dataset/converted_dataset/trackA/test.h5",
        type=str,
        help="",
    )
    parser.add_argument(
        "--B_dir",
        default="/home/xiaoli/project/3D-ShapeNet-car/src/Dataset/converted_dataset/trackB/test.h5",
        type=str,
        help="",
    )
    parser.add_argument(
        "--C_dir",
        default="/home/xiaoli/project/3D-ShapeNet-car/src/Dataset/converted_dataset/trackC/k1.h5",
        type=str,
        help="",
    )
    params = parser.parse_args()
    merge_h5_files(params.A_dir, params.B_dir, params.C_dir)
print("done")
