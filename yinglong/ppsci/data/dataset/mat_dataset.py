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


class MatDataset(io.Dataset):
    """Dataset class for .mat file.

    Args:
        file_path (str): Mat file path.
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
        >>> dataset = ppsci.data.dataset.MatDataset(
        ...     "/path/to/file.mat"
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
        raw_data = reader.load_mat_file(
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
        self.weight = {
            key: np.ones_like(next(iter(self.label.values()))) for key in self.label
        }
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

        # TODO(sensen): Transforms may be applied on label and weight.
        if self.transforms is not None:
            input_item = self.transforms(input_item)

        return (input_item, label_item, weight_item)

    def __len__(self):
        return self._len


class IterableMatDataset(io.IterableDataset):
    """IterableMatDataset for full-data loading.

    Args:
        file_path (str): Mat file path.
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
        >>> dataset = ppsci.data.dataset.IterableMatDataset(
        ...     "/path/to/file.mat"
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
        raw_data = reader.load_mat_file(
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
        self.weight = {
            key: np.ones_like(next(iter(self.label.values()))) for key in self.label
        }
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
        yield self.input, self.label, self.weight

    def __len__(self):
        return 1
