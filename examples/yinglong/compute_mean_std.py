import os

import h5py
import numpy as np
from tqdm import tqdm


def read_data(path: str, var="fields"):
    if path.endswith(".h5"):
        paths = [path]
    else:
        paths = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".h5"):
                    paths.append(os.path.join(root, file))
    paths.sort()
    files = []
    for path_ in paths:

        _file = h5py.File(path_, "r")
        files.append(_file[var])
    return files


def compute_mean_per_sample(data, axis=(2, 3)):
    mean = data.mean(axis=axis)
    return mean


def main_mean():
    src_root1 = "/root/ssd3/datasets/hrrr_h5_crop/train"
    src_root2 = "/root/ssd1/zhangzhimin04/workspaces/hrrr_download/hrrr_h5_crop"
    src_root3 = "/root/home/zhangzhimin04/workspaces/hrrr_download/hrrr_h5_crop"

    files = read_data(src_root1) + read_data(src_root2) + read_data(src_root3)
    means = []
    for file in tqdm(files):
        if len(file.shape) == 1:
            continue
        data = np.asarray(file)
        mean = compute_mean_per_sample(data)
        means.append(mean)
    means = np.asarray(means)
    print(means.shape)
    final_mean = means.mean(axis=(0, 1))
    print(final_mean)
    np.save("mean_crop.npy", final_mean)


def main_std():
    src_root1 = "/root/ssd3/datasets/hrrr_h5_crop/train"
    src_root2 = "/root/ssd1/zhangzhimin04/workspaces/hrrr_download/hrrr_h5_crop"
    src_root3 = "/root/home/zhangzhimin04/workspaces/hrrr_download/hrrr_h5_crop"

    mean = np.load("mean_crop.npy")
    mean = mean.reshape(1, mean.shape[0], 1, 1)

    files = read_data(src_root1) + read_data(src_root2) + read_data(src_root3)
    stds = []
    for file in tqdm(files):
        if len(file.shape) == 1:
            continue
        data = np.asarray(file)
        data = (data - mean) ** 2
        std = compute_mean_per_sample(data)
        stds.append(std)
    stds = np.asarray(stds)
    print(stds.shape)
    final_std = stds.mean(axis=(0, 1))
    print(final_std)
    final_std = final_std**0.5
    print(final_std)
    np.save("std_crop.npy", final_std)


def main_time_mean():
    src_root1 = "/root/ssd3/datasets/hrrr_h5_crop/train"
    src_root2 = "/root/ssd1/zhangzhimin04/workspaces/hrrr_download/hrrr_h5_crop"
    src_root3 = "/root/home/zhangzhimin04/workspaces/hrrr_download/hrrr_h5_crop"

    files = read_data(src_root1) + read_data(src_root2) + read_data(src_root3)
    means = []
    for file in tqdm(files):
        if len(file.shape) == 1:
            continue
        data = np.asarray(file)
        mean = compute_mean_per_sample(data, axis=(0,))
        means.append(mean)
    means = np.asarray(means)
    print(means.shape)
    final_mean = means.mean(axis=(0,))
    print(final_mean)
    np.save("time_mean_crop.npy", final_mean)


if __name__ == "__main__":
    main_time_mean()
