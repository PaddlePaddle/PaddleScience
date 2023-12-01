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

from __future__ import annotations

from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle
from paddle import io
from paddle import vision

from ppsci.utils import misc
from ppsci.utils import reader


class NPZDataset(io.Dataset):
    """Dataset class for .npz file.

    Args:
        file_path (str): Npz file path.
        input_keys (Tuple[str, ...]): List of input keys.
        label_keys (Tuple[str, ...], optional): List of label keys. Defaults to ().
        alias_dict (Optional[Dict[str, str]]): Dict of alias(es) for input and label keys.
            i.e. {inner_key: outer_key}. Defaults to None.
        weight_dict (Optional[Dict[str, Union[Callable, float]]]): Define the weight of
            each constraint variable. Defaults to None.
        timestamps (Optional[Tuple[float, ...]]): The number of repetitions of the data
            in the time dimension. Defaults to None.
        transforms (Optional[vision.Compose]): Compose object contains sample wise
            transform(s). Defaults to None.

    Examples:
        >>> import ppsci
        >>> dataset = ppsci.data.dataset.NPZDataset(
        ...     "/path/to/file.npz"
        ...     ("x",),
        ...     ("u",),
        ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        file_path: str,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...] = (),
        alias_dict: Optional[Dict[str, str]] = None,
        weight_dict: Optional[Dict[str, Union[Callable, float]]] = None,
        timestamps: Optional[Tuple[float, ...]] = None,
        transforms: Optional[vision.Compose] = None,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.label_keys = label_keys

        # read raw data from file
        raw_data = reader.load_npz_file(
            file_path,
            input_keys + label_keys,
            alias_dict,
        )
        # filter raw data by given timestamps if specified
        if timestamps is not None:
            if "t" in raw_data:
                # filter data according to given timestamps
                raw_time_array = raw_data["t"]
                mask = []
                for ti in timestamps:
                    mask.append(np.nonzero(np.isclose(raw_time_array, ti).flatten())[0])
                raw_data = misc.convert_to_array(
                    raw_data, self.input_keys + self.label_keys
                )
                mask = np.concatenate(mask, 0)
                raw_data = raw_data[mask]
                raw_data = misc.convert_to_dict(
                    raw_data, self.input_keys + self.label_keys
                )
            else:
                # repeat data according to given timestamps
                raw_data = misc.convert_to_array(
                    raw_data, self.input_keys + self.label_keys
                )
                raw_data = misc.combine_array_with_time(raw_data, timestamps)
                self.input_keys = ("t",) + tuple(self.input_keys)
                raw_data = misc.convert_to_dict(
                    raw_data, self.input_keys + self.label_keys
                )

        # fetch input data
        self.input = {
            key: value for key, value in raw_data.items() if key in self.input_keys
        }
        # fetch label data
        self.label = {
            key: value for key, value in raw_data.items() if key in self.label_keys
        }

        # prepare weights
        self.weight = {}
        if weight_dict is not None:
            for key, value in weight_dict.items():
                if isinstance(value, (int, float)):
                    self.weight[key] = np.full_like(
                        next(iter(self.label.values())), value
                    )
                elif callable(value):
                    func = value
                    self.weight[key] = func(self.input)
                    if isinstance(self.weight[key], (int, float)):
                        self.weight[key] = np.full_like(
                            next(iter(self.label.values())), self.weight[key]
                        )
                else:
                    raise NotImplementedError(f"type of {type(value)} is invalid yet.")

        self.transforms = transforms
        self._len = len(next(iter(self.input.values())))

    def __getitem__(self, idx):
        input_item = {key: value[idx] for key, value in self.input.items()}
        label_item = {key: value[idx] for key, value in self.label.items()}
        weight_item = {key: value[idx] for key, value in self.weight.items()}

        if self.transforms is not None:
            input_item, label_item, weight_item = self.transforms(
                input_item, label_item, weight_item
            )

        return (input_item, label_item, weight_item)

    def __len__(self):
        return self._len


class IterableNPZDataset(io.IterableDataset):
    """IterableNPZDataset for full-data loading.

    Args:
        file_path (str): Npz file path.
        input_keys (Tuple[str, ...]): List of input keys.
        label_keys (Tuple[str, ...], optional): List of label keys. Defaults to ().
        alias_dict (Optional[Dict[str, str]]): Dict of alias(es) for input and label keys.
            i.e. {inner_key: outer_key}. Defaults to None.
        weight_dict (Optional[Dict[str, Union[Callable, float]]]): Define the weight of
            each constraint variable. Defaults to None.
        timestamps (Optional[Tuple[float, ...]]): The number of repetitions of the data
            in the time dimension. Defaults to None.
        transforms (Optional[vision.Compose]): Compose object contains sample wise
            transform(s). Defaults to None.

    Examples:
        >>> import ppsci
        >>> dataset = ppsci.data.dataset.IterableNPZDataset(
        ...     "/path/to/file.npz"
        ...     ("x",),
        ...     ("u",),
        ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        file_path: str,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...] = (),
        alias_dict: Optional[Dict[str, str]] = None,
        weight_dict: Optional[Dict[str, Union[Callable, float]]] = None,
        timestamps: Optional[Tuple[float, ...]] = None,
        transforms: Optional[vision.Compose] = None,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.label_keys = label_keys

        # read raw data from file
        raw_data = reader.load_npz_file(
            file_path,
            input_keys + label_keys,
            alias_dict,
        )
        # filter raw data by given timestamps if specified
        if timestamps is not None:
            if "t" in raw_data:
                # filter data according to given timestamps
                raw_time_array = raw_data["t"]
                mask = []
                for ti in timestamps:
                    mask.append(np.nonzero(np.isclose(raw_time_array, ti).flatten())[0])
                raw_data = misc.convert_to_array(
                    raw_data, self.input_keys + self.label_keys
                )
                mask = np.concatenate(mask, 0)
                raw_data = raw_data[mask]
                raw_data = misc.convert_to_dict(
                    raw_data, self.input_keys + self.label_keys
                )
            else:
                # repeat data according to given timestamps
                raw_data = misc.convert_to_array(
                    raw_data, self.input_keys + self.label_keys
                )
                raw_data = misc.combine_array_with_time(raw_data, timestamps)
                self.input_keys = ("t",) + tuple(self.input_keys)
                raw_data = misc.convert_to_dict(
                    raw_data, self.input_keys + self.label_keys
                )

        # fetch input data
        self.input = {
            key: value for key, value in raw_data.items() if key in self.input_keys
        }
        # fetch label data
        self.label = {
            key: value for key, value in raw_data.items() if key in self.label_keys
        }

        # prepare weights
        self.weight = {}
        if weight_dict is not None:
            for key, value in weight_dict.items():
                if isinstance(value, (int, float)):
                    self.weight[key] = np.full_like(
                        next(iter(self.label.values())), value
                    )
                elif callable(value):
                    func = value
                    self.weight[key] = func(self.input)
                    if isinstance(self.weight[key], (int, float)):
                        self.weight[key] = np.full_like(
                            next(iter(self.label.values())), self.weight[key]
                        )
                else:
                    raise NotImplementedError(f"type of {type(value)} is invalid yet.")

        self.input = {key: paddle.to_tensor(value) for key, value in self.input.items()}
        self.label = {key: paddle.to_tensor(value) for key, value in self.label.items()}
        self.weight = {
            key: paddle.to_tensor(value) for key, value in self.weight.items()
        }

        self.transforms = transforms
        self._len = len(next(iter(self.input.values())))

    @property
    def num_samples(self):
        """Number of samples within current dataset."""
        return self._len

    def __iter__(self):
        if callable(self.transforms):
            input_, label_, weight_ = self.transforms(
                self.input, self.label, self.weight
            )
            yield input_, label_, weight_
        else:
            yield self.input, self.label, self.weight

    def __len__(self):
        return 1





class ScalerStd(object):
    """
    Desc: Normalization utilities with std mean
    """

    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)

    def transform(self, data):
        mean = paddle.to_tensor(self.mean).type_as(data).to(data.device) if paddle.is_tensor(data) else self.mean
        std = paddle.to_tensor(self.std).type_as(data).to(data.device) if paddle.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = paddle.to_tensor(self.mean) if paddle.is_tensor(data) else self.mean
        std = paddle.to_tensor(self.std) if paddle.is_tensor(data) else self.std
        return (data * std) + mean


class VAECustomDataset(io.Dataset):
    def __init__(self, file_path, data_type="train"):
        """

        :param file_path:
        :param data_type: train or test
        """
        super().__init__()
        all_data = np.load(file_path)
        data = all_data["data"]
        num, _, _ = data.shape
        data = data.reshape(num, -1)

        self.neighbors = all_data['neighbors']
        self.areasoverlengths = all_data['areasoverlengths']
        self.dirichletnodes = all_data['dirichletnodes']
        self.dirichleths = all_data['dirichletheads']
        self.Qs = np.zeros([all_data['coords'].shape[-1]])
        self.val_data = all_data["test_data"]

        self.data_type = data_type

        self.train_len = int(num * 0.8)
        self.test_len = num - self.train_len

        self.train_data = data[:self.train_len]
        self.test_data = data[self.train_len:]

        self.scaler = ScalerStd()
        self.scaler.fit(self.train_data)

        self.train_data = self.scaler.transform(self.train_data)
        self.test_data = self.scaler.transform(self.test_data)
        
        self.input_keys = ""
        self.label_keys = ""

    def __getitem__(self, idx):
        if self.data_type == "train":
            return self.train_data[idx]
        else:
            return self.test_data[idx]

    def __len__(self):
        if self.data_type == "train":
            return self.train_len
        else:
            return self.test_len
