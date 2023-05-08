# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
from multiprocessing import Pool
from typing import Any
from typing import Dict
from typing import Tuple

import h5py
import numpy as np
from paddle.io import DistributedBatchSampler
from tqdm import tqdm

import ppsci
from ppsci.utils import logger


def get_mean_std(mean_path: str, std_path: str, vars_channel: Tuple[int, ...]):
    mean = np.load(mean_path).squeeze(0).astype(np.float32)
    mean = mean[vars_channel]
    std = np.load(std_path).squeeze(0).astype(np.float32)
    std = std[vars_channel]
    return mean, std


def sample_func(
    dataset_cfg: Dict[str, Any], save_path: str, batch_idxs: Tuple[int, ...]
):
    dataset = ppsci.data.dataset.build_dataset(dataset_cfg)
    for idx in tqdm(batch_idxs):
        input_dict, label_dict, weight_dict = dataset.getitem(idx)
        fdest = h5py.File("{}/{:0>8d}.h5".format(save_path, idx), "w")
        for key, value in input_dict.items():
            fdest.create_dataset(f"input_dict/{key}", data=value, dtype="f")
        for key, value in label_dict.items():
            fdest.create_dataset(f"label_dict/{key}", data=value, dtype="f")
        if weight_dict is not None:
            for key, value in weight_dict.items():
                fdest.create_dataset(f"weight_dict/{key}", data=value, dtype="f")


def sample_data_epoch(epoch: int):
    # initialize logger
    logger.init_logger("ppsci")
    # set dataset path and save path
    train_file_path = "./datasets/era5/train"
    precip_file_path = None
    data_mean_path = "./datasets/era5/stat/global_means.npy"
    data_std_path = "./datasets/era5/stat/global_stds.npy"
    tmp_save_path = "./datasets/era5/train_split_rank0/epoch_tmp"
    save_path = f"./datasets/era5/train_split_rank0/epoch_{epoch}"
    # set hyper-parameters
    input_keys = ["input"]
    output_keys = ["output"]
    img_h, img_w = 720, 1440
    # FourCastNet use 20 atmospheric variable，their index in the dataset is from 0 to 19.
    # The variable name is 'u10', 'v10', 't2m', 'sp', 'msl', 't850', 'u1000', 'v1000', 'z000',
    # 'u850', 'v850', 'z850',  'u500', 'v500', 'z500', 't500', 'z50', 'r500', 'r850', 'tcwv'.
    # You can obtain detailed information about each variable from
    # https://cds.climate.copernicus.eu/cdsapp#!/search?text=era5&type=dataset
    vars_channel = [i for i in range(20)]
    num_trainer = 1
    rank = 0
    processes = 16

    if os.path.exists(tmp_save_path):
        shutil.rmtree(tmp_save_path)
        logger.info(f"tmp_save_path({tmp_save_path}) is arrelady exists! remove it")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        logger.info(f"save_path({save_path}) is arrelady exists! remove it")
    os.makedirs(tmp_save_path)

    data_mean, data_std = get_mean_std(data_mean_path, data_std_path, vars_channel)
    transforms = [
        {
            "SqueezeData": {},
        },
        {
            "CropData": {"xmin": (0, 0), "xmax": (img_h, img_w)},
        },
        {
            "Normalize": {"mean": data_mean, "std": data_std},
        },
    ]
    dataset_cfg = {
        "name": "ERA5Dataset",
        "file_path": train_file_path,
        "input_keys": input_keys,
        "label_keys": output_keys,
        "precip_file_path": precip_file_path,
        "vars_channel": vars_channel,
        "transforms": transforms,
    }
    dataset = ppsci.data.dataset.build_dataset(dataset_cfg)

    batch_sampler = DistributedBatchSampler(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_replicas=num_trainer,
        rank=rank,
    )
    batch_sampler.set_epoch(epoch)
    batch_idxs = []
    for data in tqdm(batch_sampler):
        batch_idxs += data

    pool = Pool(processes=processes)
    for st in range(0, len(batch_idxs), len(batch_idxs) // (processes - 1)):
        end = st + len(batch_idxs) // (processes - 1)
        result = pool.apply_async(
            sample_func, (dataset_cfg, tmp_save_path, batch_idxs[st:end])
        )
    pool.close()
    pool.join()
    if result.successful():
        logger.info("successful")
        shutil.move(tmp_save_path, save_path)
        logger.info(f"move {tmp_save_path} to {save_path}")


def main():
    epoch = 0
    sample_data_epoch(epoch)

    # if you want to sample every 5 epochs, you can use the following code
    # epoch = 150
    # for i in range(0, epoch, 5):
    #     sample_data_epoch(epoch)


if __name__ == "__main__":
    main()
