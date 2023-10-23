import os

import h5py
import numpy as np

source_path = "./Dataset/OriginalData/"
output_path = "./Dataset/PreparedData/top_dataset.h5"

IMAGE_H, IMAGE_W = 40, 40
N_ITERS = 100

files = os.listdir(source_path)
iters_shape = (len(files), N_ITERS, IMAGE_H, IMAGE_W)
iters_chunk_shape = (1, 1, IMAGE_H, IMAGE_W)
target_shape = (len(files), 1, IMAGE_H, IMAGE_W)
target_chunk_shape = (1, 1, IMAGE_H, IMAGE_W)

if __name__ == "__main__":
    # shape: iters (10000, 100, 40, 40)
    # shape: y (10000, 1, 40, 40)
    with h5py.File(output_path, "w") as h5f:
        iters = h5f.create_dataset("iters", iters_shape, chunks=iters_chunk_shape)
        targets = h5f.create_dataset("targets", target_shape, chunks=target_chunk_shape)

        for i, file_name in enumerate(files):
            file_path = os.path.join(source_path, file_name)
            arr = np.load(file_path)["arr_0"]
            iters[i] = arr

            th_ = arr.mean(axis=(1, 2), keepdims=True)
            targets[i] = (arr > th_).astype("float32")[[-1], :, :]
