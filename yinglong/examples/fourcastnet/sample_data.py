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

import glob
import os
import shutil
from multiprocessing import Pool
from typing import Any
from typing import Dict
from typing import Tuple

import h5py
from paddle import io
from tqdm import tqdm

import examples.fourcastnet.utils as fourcast_utils
import ppsci
from ppsci.utils import logger


def sample_func(
    dataset_cfg: Dict[str, Any], save_path: str, batch_idxs: Tuple[int, ...]
):
    dataset = ppsci.data.dataset.build_dataset(dataset_cfg)
    for idx in tqdm(batch_idxs):
        input_dict, label_dict, weight_dict = dataset[idx]
        fdest = h5py.File(f"{save_path}/{idx:0>8d}.h5", "w")
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
    TRAIN_FILE_PATH = "./datasets/era5/train"
    PRECIP_FILE_PATH = None
    DATA_MEAN_PATH = "./datasets/era5/stat/global_means.npy"
    DATA_STD_PATH = "./datasets/era5/stat/global_stds.npy"
    TMP_SAVE_PATH = "./datasets/era5/train_split_rank0/epoch_tmp"
    save_path = f"./datasets/era5/train_split_rank0/epoch_{epoch}"
    # set hyper-parameters
    input_keys = ("input",)
    output_keys = ("output",)
    IMG_H, IMG_W = 720, 1440
    # FourCastNet use 20 atmospheric variable，their index in the dataset is from 0 to 19.
    # The variable name is 'u10', 'v10', 't2m', 'sp', 'msl', 't850', 'u1000', 'v1000', 'z000',
    # 'u850', 'v850', 'z850',  'u500', 'v500', 'z500', 't500', 'z50', 'r500', 'r850', 'tcwv'.
    # You can obtain detailed information about each variable from
    # https://cds.climate.copernicus.eu/cdsapp#!/search?text=era5&type=dataset
    VARS_CHANNEL = list(range(20))
    NUM_TRAINER = 1
    RANK = 0
    PROCESSES = 16

    if len(glob.glob(TMP_SAVE_PATH + "/*.h5")):
        raise FileExistsError(
            f"TMP_SAVE_PATH({TMP_SAVE_PATH}) is not an empty folder, please specify an empty folder."
        )
    if len(glob.glob(save_path + "/*.h5")):
        raise FileExistsError(
            f"save_path({save_path}) is not an empty folder, please specify an empty folder."
        )
    os.makedirs(TMP_SAVE_PATH, exist_ok=True)

    data_mean, data_std = fourcast_utils.get_mean_std(
        DATA_MEAN_PATH, DATA_STD_PATH, VARS_CHANNEL
    )
    transforms = [
        {"SqueezeData": {}},
        {"CropData": {"xmin": (0, 0), "xmax": (IMG_H, IMG_W)}},
        {"Normalize": {"mean": data_mean, "std": data_std}},
    ]
    dataset_cfg = {
        "name": "ERA5Dataset",
        "file_path": TRAIN_FILE_PATH,
        "input_keys": input_keys,
        "label_keys": output_keys,
        "PRECIP_FILE_PATH": PRECIP_FILE_PATH,
        "vars_channel": VARS_CHANNEL,
        "transforms": transforms,
    }
    dataset = ppsci.data.dataset.build_dataset(dataset_cfg)

    batch_sampler = io.DistributedBatchSampler(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_replicas=NUM_TRAINER,
        rank=RANK,
    )
    batch_sampler.set_epoch(epoch)
    batch_idxs = []
    for data in tqdm(batch_sampler):
        batch_idxs += data

    pool = Pool(processes=PROCESSES)
    for st in range(0, len(batch_idxs), len(batch_idxs) // (PROCESSES - 1)):
        end = st + len(batch_idxs) // (PROCESSES - 1)
        result = pool.apply_async(
            sample_func, (dataset_cfg, TMP_SAVE_PATH, batch_idxs[st:end])
        )
    pool.close()
    pool.join()
    if result.successful():
        logger.info("successful")
        shutil.move(TMP_SAVE_PATH, save_path)
        logger.info(f"move {TMP_SAVE_PATH} to {save_path}")


def main():
    EPOCHS = 0
    sample_data_epoch(EPOCHS)

    # if you want to sample every 5 epochs, you can use the following code
    # EPOCHS = 150
    # for epoch in range(0, EPOCHS, 5):
    #     sample_data_epoch(epoch)


if __name__ == "__main__":
    main()
