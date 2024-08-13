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
Code below is heavily based on [transformer-physx](https://github.com/zabaras/transformer-physx)
"""

from typing import Dict
from typing import Optional
from typing import Tuple

import h5py
import numpy as np
import paddle
from paddle import io

from ppsci.arch import base


class LorenzDataset(io.Dataset):
    """Dataset for training Lorenz model.

    Args:
        file_path (str): Data set path.
        input_keys (Tuple[str, ...]): Input keys, such as ("states",).
        label_keys (Tuple[str, ...]): Output keys, such as ("pred_states", "recover_states").
        block_size (int): Data block size.
        stride (int): Data stride.
        ndata (Optional[int]): Number of data series to use. Defaults to None.
        weight_dict (Optional[Dict[str, float]]): Weight dictionary. Defaults to None.
        embedding_model (Optional[base.Arch]): Embedding model. Defaults to None.

    Examples:
        >>> import ppsci
        >>> dataset = ppsci.data.dataset.LorenzDataset(
        ...     "file_path": "/path/to/LorenzDataset",
        ...     "input_keys": ("x",),
        ...     "label_keys": ("v",),
        ...     "block_size": 32,
        ...     "stride": 16,
        ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        file_path: str,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        block_size: int,
        stride: int,
        ndata: Optional[int] = None,
        weight_dict: Optional[Dict[str, float]] = None,
        embedding_model: Optional[base.Arch] = None,
    ):
        super().__init__()
        self.file_path = file_path
        self.input_keys = input_keys
        self.label_keys = label_keys

        self.block_size = block_size
        self.stride = stride
        self.ndata = ndata
        self.weight_dict = {key: 1.0 for key in self.label_keys}
        if weight_dict is not None:
            self.weight_dict.update(weight_dict)

        self.data = self.read_data(file_path, block_size, stride)
        self.embedding_model = embedding_model
        if embedding_model is None:
            self.embedding_data = None
        else:
            embedding_model.eval()
            with paddle.no_grad():
                data_tensor = paddle.to_tensor(self.data)
                embedding_data_tensor = embedding_model.encoder(data_tensor)
            self.embedding_data = embedding_data_tensor.numpy()

    def read_data(self, file_path: str, block_size: int, stride: int):
        data = []
        with h5py.File(file_path, "r") as f:
            data_num = 0
            for key in f.keys():
                data_series = np.asarray(f[key], dtype=paddle.get_default_dtype())
                for i in range(0, data_series.shape[0] - block_size + 1, stride):
                    data.append(data_series[i : i + block_size])
                data_num += 1
                if self.ndata is not None and data_num >= self.ndata:
                    break
        return np.asarray(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # when embedding data is None
        if self.embedding_data is None:
            data_item = self.data[i]
            input_item = {self.input_keys[0]: data_item}
            label_item = {
                self.label_keys[0]: data_item[1:, :],
                self.label_keys[1]: data_item,
            }
        else:
            data_item = self.embedding_data[i]
            input_item = {self.input_keys[0]: data_item[:-1, :]}
            label_item = {self.label_keys[0]: data_item[1:, :]}
            if len(self.label_keys) == 2:
                label_item[self.label_keys[1]] = self.data[i][1:, :]

        weight_shape = [1] * len(data_item.shape)
        weight_item = {
            key: np.full(weight_shape, value) for key, value in self.weight_dict.items()
        }
        return (input_item, label_item, weight_item)


class RosslerDataset(LorenzDataset):
    """Dataset for training Rossler model.

    Args:
        file_path (str): Data set path.
        input_keys (Tuple[str, ...]): Input keys, such as ("states",).
        label_keys (Tuple[str, ...]): Output keys, such as ("pred_states", "recover_states").
        block_size (int): Data block size.
        stride (int): Data stride.
        ndata (Optional[int]): Number of data series to use. Defaults to None.
        weight_dict (Optional[Dict[str, float]]): Weight dictionary. Defaults to None.
        embedding_model (Optional[base.Arch]): Embedding model. Defaults to None.

    Examples:
        >>> import ppsci
        >>> dataset = ppsci.data.dataset.RosslerDataset(
        ...     "file_path": "/path/to/RosslerDataset",
        ...     "input_keys": ("x",),
        ...     "label_keys": ("v",),
        ...     "block_size": 32,
        ...     "stride": 16,
        ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        file_path: str,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        block_size: int,
        stride: int,
        ndata: Optional[int] = None,
        weight_dict: Optional[Dict[str, float]] = None,
        embedding_model: Optional[base.Arch] = None,
    ):
        super().__init__(
            file_path,
            input_keys,
            label_keys,
            block_size,
            stride,
            ndata,
            weight_dict,
            embedding_model,
        )


class CylinderDataset(io.Dataset):
    """Dataset for training Cylinder model.

    Args:
        file_path (str): Data set path.
        input_keys (Tuple[str, ...]): Input keys, such as ("states","visc").
        label_keys (Tuple[str, ...]): Output keys, such as ("pred_states", "recover_states").
        block_size (int): Data block size.
        stride (int): Data stride.
        ndata (Optional[int]): Number of data series to use. Defaults to None.
        weight_dict (Optional[Dict[str, float]]): Weight dictionary. Defaults to None.
        embedding_model (Optional[base.Arch]): Embedding model. Defaults to None.
        embedding_batch_size (int, optional): The batch size of embedding model. Defaults to 64.

    Examples:
        >>> import ppsci
        >>> dataset = ppsci.data.dataset.CylinderDataset(
        ...     "file_path": "/path/to/CylinderDataset",
        ...     "input_keys": ("x",),
        ...     "label_keys": ("v",),
        ...     "block_size": 32,
        ...     "stride": 16,
        ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        file_path: str,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        block_size: int,
        stride: int,
        ndata: Optional[int] = None,
        weight_dict: Optional[Dict[str, float]] = None,
        embedding_model: Optional[base.Arch] = None,
        embedding_batch_size: int = 64,
    ):
        super().__init__()
        self.file_path = file_path
        self.input_keys = input_keys
        self.label_keys = label_keys

        self.block_size = block_size
        self.stride = stride
        self.ndata = ndata
        self.weight_dict = {key: 1.0 for key in self.label_keys}
        if weight_dict is not None:
            self.weight_dict.update(weight_dict)

        self.data, self.visc = self.read_data(file_path, block_size, stride)
        self.embedding_model = embedding_model
        if embedding_model is None:
            self.embedding_data = None
        else:
            embedding_model.eval()
            with paddle.no_grad():
                data_tensor = paddle.to_tensor(self.data)
                visc_tensor = paddle.to_tensor(self.visc)
                embedding_data = []
                for i in range(0, len(data_tensor), embedding_batch_size):
                    start, end = i, min(i + embedding_batch_size, len(data_tensor))
                    embedding_data_batch = embedding_model.encoder(
                        data_tensor[start:end], visc_tensor[start:end]
                    )
                    embedding_data.append(embedding_data_batch.numpy())
                self.embedding_data = np.concatenate(embedding_data)

    def read_data(self, file_path: str, block_size: int, stride: int):
        data = []
        visc = []
        with h5py.File(file_path, "r") as f:
            data_num = 0
            for key in f.keys():
                visc0 = 2.0 / float(key)
                ux = np.asarray(f[key + "/ux"], dtype=paddle.get_default_dtype())
                uy = np.asarray(f[key + "/uy"], dtype=paddle.get_default_dtype())
                p = np.asarray(f[key + "/p"], dtype=paddle.get_default_dtype())
                data_series = np.stack([ux, uy, p], axis=1)

                for i in range(0, data_series.shape[0] - block_size + 1, stride):
                    data.append(data_series[i : i + block_size])
                    visc.append([visc0])

                data_num += 1
                if self.ndata is not None and data_num >= self.ndata:
                    break

        data = np.asarray(data)
        visc = np.asarray(visc, dtype=paddle.get_default_dtype())
        return data, visc

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if self.embedding_data is None:
            data_item = self.data[i]
            input_item = {
                self.input_keys[0]: data_item,
                self.input_keys[1]: self.visc[i],
            }
            label_item = {
                self.label_keys[0]: data_item[1:],
                self.label_keys[1]: data_item,
            }
        else:
            data_item = self.embedding_data[i]
            input_item = {self.input_keys[0]: data_item[:-1, :]}
            label_item = {self.label_keys[0]: data_item[1:, :]}
            if len(self.label_keys) == 2:
                label_item[self.label_keys[1]] = data_item[1:, :]
        weight_shape = [1] * len(data_item.shape)
        weight_item = {
            key: np.full(weight_shape, value) for key, value in self.weight_dict.items()
        }
        return (input_item, label_item, weight_item)
