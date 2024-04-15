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

import numpy as np
import paddle
from paddle import io
from paddle import vision


class NamedArrayDataset(io.Dataset):
    """Class for Named Array Dataset.

    Args:
        input (Dict[str, np.ndarray]): Input dict.
        label (Optional[Dict[str, np.ndarray]]): Label dict. Defaults to None.
        weight (Optional[Dict[str, np.ndarray]]): Weight dict. Defaults to None.
        transforms (Optional[vision.Compose]): Compose object contains sample wise
            transform(s). Defaults to None.

    Examples:
        >>> import ppsci
        >>> input = {"x": np.random.randn(100, 1)}
        >>> output = {"u": np.random.randn(100, 1)}
        >>> weight = {"u": np.random.randn(100, 1)}
        >>> dataset = ppsci.data.dataset.NamedArrayDataset(input, output, weight)
    """

    # Whether support batch indexing for speeding up fetching process.
    batch_index: bool = True

    def __init__(
        self,
        input: Dict[str, np.ndarray],
        label: Optional[Dict[str, np.ndarray]] = None,
        weight: Optional[Dict[str, np.ndarray]] = None,
        transforms: Optional[vision.Compose] = None,
    ):
        super().__init__()
        self.input = input
        self.label = {} if label is None else label
        self.input_keys = tuple(input.keys())
        self.label_keys = tuple(self.label.keys())
        self.weight = {} if weight is None else weight
        self.transforms = transforms
        self._len = len(next(iter(input.values())))

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


class IterableNamedArrayDataset(io.IterableDataset):
    """IterableNamedArrayDataset for full-data loading.

    Args:
        input (Dict[str, np.ndarray]): Input dict.
        label (Optional[Dict[str, np.ndarray]]): Label dict. Defaults to None.
        weight (Optional[Dict[str, np.ndarray]]): Weight dict. Defaults to None.
        transforms (Optional[vision.Compose]): Compose object contains sample wise
            transform(s). Defaults to None.

    Examples:
        >>> import ppsci
        >>> input = {"x": np.random.randn(100, 1)}
        >>> label = {"u": np.random.randn(100, 1)}
        >>> weight = {"u": np.random.randn(100, 1)}
        >>> dataset = ppsci.data.dataset.IterableNamedArrayDataset(input, label, weight)
    """

    # Whether support batch indexing for speeding up fetching process.
    batch_index: bool = False

    def __init__(
        self,
        input: Dict[str, np.ndarray],
        label: Optional[Dict[str, np.ndarray]] = None,
        weight: Optional[Dict[str, np.ndarray]] = None,
        transforms: Optional[vision.Compose] = None,
    ):
        super().__init__()
        self.input = {key: paddle.to_tensor(value) for key, value in input.items()}
        self.label = (
            {key: paddle.to_tensor(value) for key, value in label.items()}
            if label is not None
            else {}
        )
        self.input_keys = tuple(input.keys())
        self.label_keys = tuple(self.label.keys())
        self.weight = (
            {
                key: paddle.to_tensor(value, paddle.get_default_dtype())
                for key, value in weight.items()
            }
            if weight is not None
            else None
        )
        self._len = len(next(iter(self.input.values())))
        self.transforms = transforms

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


class ContinuousNamedArrayDataset(io.IterableDataset):
    """ContinuousNamedArrayDataset for iterable sampling.

    Args:
        input (Callable): Function generate input dict.
        label (Callable): Function generate label dict.
        weight (Optional[Callable]): Function generate weight dict. Defaults to None.
        transforms (Optional[vision.Compose]): Compose object contains sample wise
            transform(s). Defaults to None.

    Examples:
        >>> import ppsci
        >>> import numpy as np
        >>> input = lambda : {"x": np.random.randn(100, 1)}
        >>> label = lambda inp: {"u": np.random.randn(100, 1)}
        >>> weight = lambda inp, label: {"u": 1 - (label["u"] ** 2)}
        >>> dataset = ppsci.data.dataset.ContinuousNamedArrayDataset(input, label, weight)
        >>> input_batch, label_batch, weight_batch = next(iter(dataset))
        >>> print(input_batch["x"].shape)
        [100, 1]
        >>> print(label_batch["u"].shape)
        [100, 1]
        >>> print(weight_batch["u"].shape)
        [100, 1]
    """

    # Whether support batch indexing for speeding up fetching process.
    batch_index: bool = False

    def __init__(
        self,
        input: Callable,
        label: Callable,
        weight: Optional[Callable] = None,
        transforms: Optional[vision.Compose] = None,
    ):
        super().__init__()
        self.input_fn = input
        self.input_keys = tuple(self.input_fn().keys())

        self.label_fn = label
        input_ = self.input_fn()
        self.label_keys = tuple(self.label_fn(input_).keys())

        self.weight_fn = weight
        self.transforms = transforms

    @property
    def num_samples(self):
        """Number of samples within current dataset."""
        raise NotImplementedError(
            "ContinuousNamedArrayDataset has no fixed number of samples."
        )

    def __iter__(self):
        def to_tensor_dict(_dict):
            if _dict is None:
                return None
            return {k: paddle.to_tensor(v) for k, v in _dict.items()}

        while True:
            input_batch = self.input_fn()
            label_batch = self.label_fn(input_batch)
            if callable(self.weight_fn):
                weight_batch = self.weight_fn(input_batch, label_batch)
            else:
                weight_batch = None

            if callable(self.transforms):
                input_batch, label_batch, weight_batch = self.transforms(
                    input_batch, label_batch, weight_batch
                )
            yield to_tensor_dict(input_batch), to_tensor_dict(
                label_batch
            ), to_tensor_dict(weight_batch)

    def __len__(self):
        return 1


class ChipHeatDataset(io.Dataset):
    """ChipHeatDataset for data loading of multi-branch DeepONet model.

    Args:
        input (Dict[str, np.ndarray]): Input dict.
        label (Optional[Dict[str, np.ndarray]]): Label dict. Defaults to None.
        index (tuple[str, ...]): Key of input dict.
        data_type (str): One of key of input dict.
        weight (Optional[Dict[str, np.ndarray]]): Weight dict. Defaults to None.
        transforms (Optional[vision.Compose]): Compose object contains sample wise
            transform(s). Defaults to None.

    Examples:
        >>> import ppsci
        >>> input = {"x": np.random.randn(100, 1)}
        >>> label = {"u": np.random.randn(100, 1)}
        >>> index = ('x', 'u', 'bc', 'bc_data')
        >>> data_type = 'u'
        >>> weight = {"u": np.random.randn(100, 1)}
        >>> dataset = ppsci.data.dataset.ChipHeatDataset(input, label, index, data_type, weight)
    """

    def __init__(
        self,
        input: Dict[str, np.ndarray],
        label: Dict[str, np.ndarray],
        index: tuple[str, ...],
        data_type: str,
        weight: Optional[Dict[str, float]] = None,
        transforms: Optional[vision.Compose] = None,
    ):
        super().__init__()
        self.input = input
        self.label = label
        self.input_keys = tuple(input.keys())
        self.label_keys = tuple(label.keys())
        self.index = index
        self.data_type = data_type
        self.weight = {} if weight is None else weight
        self.transforms = transforms

    def __getitem__(self, idx):
        quotient = idx
        index_ir = dict()
        for i in self.index:
            index_ir[i] = 0

        for i in index_ir:
            num = len(self.input[i])
            index_ir[i] = quotient % num
            quotient = quotient // num

        input_item = {}
        for key in self.input:
            if key == "y":
                input_item[key] = self.input[key][index_ir["x"]]
            elif key == "u_one":
                input_item[key] = self.input[key][
                    len(self.input[self.data_type]) * index_ir["x"]
                    + index_ir[self.data_type]
                ]
            else:
                input_item[key] = self.input[key][index_ir[key]]

        label_item = {key: value for key, value in self.label.items()}
        weight_item = {key: value for key, value in self.weight.items()}

        if self.transforms is not None:
            input_item, label_item, weight_item = self.transforms(
                (input_item, label_item, weight_item)
            )

        return (input_item, label_item, weight_item)

    def __len__(self):
        _len = 1
        for i in self.index:
            _len *= len(self.input[i])
        return _len
