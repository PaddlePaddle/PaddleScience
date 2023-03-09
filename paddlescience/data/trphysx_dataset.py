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
"""
This code is refer from: 
https://github.com/zabaras/transformer-physx/blob/main/trphysx/embedding/training/enn_data_handler.py
https://github.com/zabaras/transformer-physx/blob/main/examples/rossler/rossler_module/dataset_rossler.py
"""

import numpy as np
import h5py

from paddle.io import Dataset


class LorenzDataset(Dataset):
    """ Dataset for training Lorenz model """

    def __init__(self, file_path, block_size, stride, ndata=None):
        super(LorenzDataset, self).__init__()
        self.file_path = file_path
        self.block_size = block_size
        self.stride = stride
        self.ndata = ndata

        self.read_data(file_path, block_size, stride)

    def read_data(self, file_path, block_size, stride):

        data = []
        with h5py.File(file_path, "r") as f:
            data_num = 0
            for key in f.keys():
                data_series = np.asarray(f[key], dtype='float32')
                for i in range(0, data_series.shape[0] - block_size + 1,
                               stride):
                    data.append(data_series[i:i + block_size])
                data_num += 1
                if self.ndata is not None and data_num >= self.ndata:
                    break

        data = np.asarray(data)
        self.mu = np.asarray([
            np.mean(data[:, :, 0]), np.mean(data[:, :, 1]),
            np.mean(data[:, :, 2])
        ]).reshape(1, 3)
        self.std = np.asarray([
            np.std(data[:, :, 0]), np.std(data[:, :, 1]), np.std(data[:, :, 2])
        ]).reshape(1, 3)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return {"inputs": self.data[i]}


class CylinderDataset(Dataset):
    """ Dataset for training Lorenz model """

    def __init__(self, file_path, block_size, stride, ndata=None):
        super(CylinderDataset, self).__init__()
        self.file_path = file_path
        self.block_size = block_size
        self.stride = stride
        self.ndata = ndata

        self.read_data(file_path, block_size, stride)

    def read_data(self, file_path, block_size, stride):

        data = []
        visc = []
        with h5py.File(file_path, "r") as f:
            data_num = 0
            for key in f.keys():
                visc0 = (2.0 / float(key))
                ux = np.asarray(f[key + '/ux'], dtype='float32')
                uy = np.asarray(f[key + '/uy'], dtype='float32')
                p = np.asarray(f[key + '/p'], dtype='float32')
                data_series = np.stack([ux, uy, p], axis=1)

                for i in range(0, data_series.shape[0] - block_size + 1,
                               stride):
                    data.append(data_series[i:i + block_size])
                    visc.append([visc0])

                data_num += 1
                if self.ndata is not None and data_num >= self.ndata:
                    break

        data = np.asarray(data)
        visc = np.asarray(visc, dtype='float32')

        self.mu = np.asarray([
            np.mean(data[:, :, 0]), np.mean(data[:, :, 1]),
            np.mean(data[:, :, 2]), np.mean(visc)
        ]).reshape(1, 4, 1, 1)
        self.std = np.asarray([
            np.std(data[:, :, 0]), np.std(data[:, :, 1]),
            np.std(data[:, :, 2]), np.std(visc)
        ]).reshape(1, 4, 1, 1)

        self.data = data
        self.visc = visc

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return {"inputs": self.data[i], "viscosity": self.visc[i]}


class RosslerDataset(Dataset):
    """ Dataset for training rossler model """

    def __init__(self, file_path, block_size, stride, ndata=None):
        super(RosslerDataset, self).__init__()
        self.file_path = file_path
        self.block_size = block_size
        self.stride = stride
        self.ndata = ndata

        self.read_data(file_path, block_size, stride)

    def read_data(self, file_path, block_size, stride):

        data = []
        with h5py.File(file_path, "r") as f:
            data_num = 0
            for key in f.keys():
                data_series = np.asarray(f[key], dtype='float32')
                for i in range(0, data_series.shape[0] - block_size + 1,
                               stride):
                    data.append(data_series[i:i + block_size])
                data_num += 1
                if self.ndata is not None and data_num >= self.ndata:
                    break

        data = np.asarray(data)
        self.mu = np.asarray([
            np.mean(data[:, :, 0]), np.mean(data[:, :, 1]),
            np.min(data[:, :, 2])
        ]).reshape(1, 3)
        self.std = np.asarray([
            np.std(data[:, :, 0]), np.std(data[:, :, 1]),
            np.max(data[:, :, 2]) - np.min(data[:, :, 2])
        ]).reshape(1, 3)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return {"inputs": self.data[i]}
