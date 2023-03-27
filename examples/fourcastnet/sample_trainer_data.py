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

import h5py
import numpy as np
from tqdm import tqdm
import glob
import os
from multiprocessing import Pool
import shutil
import time

from paddle.io import Dataset, DistributedBatchSampler


class GetDataset(Dataset):
    def __init__(self, location, two_step_training=False, precip_path=None):
        self.location = location
        self.dt = 1
        self.n_history = 0
        self._get_files_stats()
        self.two_step_training = two_step_training

        self.precip = True if precip_path is not None else False
        if self.precip:
            self.precip_paths = glob.glob(precip_path + "/*.h5")
            self.precip_paths.sort()

    def _get_files_stats(self):
        self.files_paths = glob.glob(self.location + "/*.h5")
        self.files_paths.sort()
        self.n_years = len(self.files_paths)
        with h5py.File(self.files_paths[0], "r") as _f:
            print("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_year = _f["fields"].shape[0]
            #original image shape (before padding)
            self.img_shape_x = _f["fields"].shape[
                2] - 1  #just get rid of one of the pixels
            self.img_shape_y = _f["fields"].shape[3]

        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.files = [None for _ in range(self.n_years)]
        self.precip_files = [None for _ in range(self.n_years)]
        print("Number of samples per year: {}".format(self.n_samples_per_year))
        print("Delta t: {} hours".format(6 * self.dt))
        print(
            "Including {} hours of past history in training at a frequency of {} hours".
            format(6 * self.dt * self.n_history, 6 * self.dt))

    def _open_file(self, year_idx):
        _file = h5py.File(self.files_paths[year_idx], "r")
        self.files[year_idx] = _file["fields"]
        if self.precip:
            self.precip_files[year_idx] = h5py.File(
                self.precip_paths[year_idx], "r")["tp"]

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
        year_idx = int(global_idx /
                       self.n_samples_per_year)  #which year we are on
        local_idx = int(
            global_idx % self.n_samples_per_year
        )  #which sample in that year we are on - determines indices for centering

        #open image file
        if self.files[year_idx] is None:
            self._open_file(year_idx)

        if not self.precip:
            #if we are not at least self.dt*n_history timesteps into the prediction
            if local_idx < self.dt * self.n_history:
                local_idx += self.dt * self.n_history
            #if we are on the last image in a year predict identity, else predict next timestep
            step = 0 if local_idx >= self.n_samples_per_year - self.dt else self.dt
        else:

            inp_local_idx = local_idx
            tar_local_idx = local_idx
            #if we are on the last image in a year predict identity, else predict next timestep
            step = 0 if tar_local_idx >= self.n_samples_per_year - self.dt else self.dt
            # first year has 2 missing samples in precip (they are first two time points)
            if year_idx == 0:
                lim = 1458
                local_idx = local_idx % lim
                inp_local_idx = local_idx + 2
                tar_local_idx = local_idx
                step = 0 if tar_local_idx >= lim - self.dt else self.dt

        #if two_step_training flag is true then ensure that local_idx is not the last or last but one sample in a year
        if self.two_step_training:
            if local_idx >= self.n_samples_per_year - 2 * self.dt:
                #set local_idx to last possible sample in a year that allows taking two steps forward
                local_idx = self.n_samples_per_year - 3 * self.dt

        if self.precip:

            inp = self.files[year_idx][inp_local_idx]
            if len(np.shape(inp)) == 3:
                inp = np.expand_dims(inp, 0)
            inp = inp[:, :, 0:720]
            tar = self.precip_files[year_idx][tar_local_idx + step]
            if len(np.shape(tar)) == 2:
                tar = np.expand_dims(tar, 0)
                tar = np.expand_dims(tar, 0)
            tar = tar[:, :, 0:720]
            return np.concatenate([inp, tar], axis=1)

        else:
            if self.two_step_training:
                inp = self.files[year_idx][(local_idx - self.dt * self.
                                            n_history):(local_idx + 1):self.dt]
                if len(np.shape(inp)) == 3:
                    inp = np.expand_dims(inp, 0)

                tar = self.files[year_idx][local_idx + step:local_idx + step +
                                           2]
                if len(np.shape(tar)) == 3:
                    tar = np.expand_dims(tar, 0)
                return np.concatenate([inp, tar], axis=0)
            else:
                inp = self.files[year_idx][(local_idx - self.dt * self.
                                            n_history):(local_idx + 1):self.dt]
                if len(np.shape(inp)) == 3:
                    inp = np.expand_dims(inp, 0)

                tar = self.files[year_idx][local_idx + step]
                if len(np.shape(tar)) == 3:
                    tar = np.expand_dims(tar, 0)

                return np.concatenate([inp, tar], axis=0)

    def getitem(self, global_idx):
        return self.__getitem__(global_idx)


def fun(train_data_path, save_path, batch_idxs, precip_data_path):
    dataset = GetDataset(train_data_path, precip_path=precip_data_path)
    for idx in tqdm(batch_idxs):
        data = dataset.getitem(idx)
        total, used, free = shutil.disk_usage("/")
        for i in range(100):
            total, used, free = shutil.disk_usage("/")
            if free // 2**30 < 500:
                print("No space left!!!, sleep 60s")
                time.sleep(60)
            else:
                break
        if free // 2**30 < 500:
            print("Error, No space left!!!")
            break
        fdest = h5py.File("{}/{:0>8d}.h5".format(save_path, idx), "a")
        fdest["fields"] = data


def sample_data_epoch(epoch):

    processes = 64
    train_data_path = "../afs_era5/train"
    precip_data_path = "../afs_era5/precip/train"
    save_path = "../era5_split_train/epoch_tmp"
    dst_path = "../era5_split_train/epoch_{}".format(epoch)
    if os.path.exists(save_path):
        print("save_path is arrelady exists! remove it")
        shutil.rmtree(save_path)
    if os.path.exists(dst_path):
        print("dst_path is arrelady exists! remove it")
        shutil.rmtree(dst_path)
    total, used, free = shutil.disk_usage("/")
    while free // 2**30 < 1500:
        print("No space left!!!, sleep 60s")
        time.sleep(60)
        total, used, free = shutil.disk_usage("/")

    os.makedirs(save_path)
    num_trainer = int(os.getenv("TRAINERS"))
    rank = int(os.getenv("PADDLE_TRAINER_ID"))
    print(num_trainer, rank)

    dataset = GetDataset(train_data_path)
    batch_sampler = DistributedBatchSampler(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_replicas=num_trainer,
        rank=rank, )
    batch_sampler.set_epoch(epoch)
    batch_idxs = []
    for data in tqdm(batch_sampler):
        batch_idxs += data
    # fun(train_data_path, save_path, batch_idxs, precip_data_path)
    pool = Pool(processes=processes)  # set the processes max number 3
    for i in range(0, len(batch_idxs), len(batch_idxs) // (processes - 1)):
        end = i + len(batch_idxs) // (processes - 1)
        result = pool.apply_async(fun, (train_data_path, save_path,
                                        batch_idxs[i:end], precip_data_path))
    pool.close()
    pool.join()
    if result.successful():
        print("successful")
        print("rename {} to {}!!!".format(save_path, dst_path))
        shutil.move(save_path, dst_path)
    else:
        print("error")


def main():

    epoch = 0
    sample_data_epoch(epoch)


if __name__ == "__main__":
    main()
