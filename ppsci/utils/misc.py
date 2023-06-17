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

import collections
import functools
import random
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import paddle

__all__ = [
    "all_gather",
    "AverageMeter",
    "PrettyOrderedDict",
    "Prettydefaultdict",
    "concat_dict_list",
    "convert_to_array",
    "convert_to_dict",
    "stack_dict_list",
    "combine_array_with_time",
    "set_random_seed",
    "run_on_eval_mode",
]


class AverageMeter:
    """
    Computes and stores the average and current value
    Code was based on https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self, name="", fmt="f", postfix="", need_avg=True):
        self.name = name
        self.fmt = fmt
        self.postfix = postfix
        self.need_avg = need_avg
        self.reset()

    def reset(self):
        """Reset"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def avg_info(self):
        if isinstance(self.avg, paddle.Tensor):
            self.avg = float(self.avg)
        return f"{self.name}: {self.avg:.5f}"

    @property
    def total(self):
        return f"{self.name}_sum: {self.sum:{self.fmt}}{self.postfix}"

    @property
    def total_minute(self):
        return f"{self.name} {self.sum / 60:{self.fmt}}{self.postfix} min"

    @property
    def mean(self):
        return (
            f"{self.name}: {self.avg:{self.fmt}}{self.postfix}" if self.need_avg else ""
        )

    @property
    def value(self):
        return f"{self.name}: {self.val:{self.fmt}}{self.postfix}"


class PrettyOrderedDict(collections.OrderedDict):
    def __str__(self):
        return "".join([str((k, v)) for k, v in self.items()])


class Prettydefaultdict(collections.defaultdict):
    def __str__(self):
        return "".join([str((k, v)) for k, v in self.items()])


def convert_to_dict(array: np.ndarray, keys: Tuple[str, ...]) -> Dict[str, np.ndarray]:
    """Split given array into single channel array at axis -1 in order of given keys.

    Args:
        array (np.ndarray): Array to be splited.
        keys (Tuple[str, ...]):Keys used in split.

    Returns:
        Dict[str, np.ndarray]: Splited dict.
    """
    if array.shape[-1] != len(keys):
        raise ValueError(
            f"dim of array({array.shape[-1]}) must equal to " f"len(keys)({len(keys)})"
        )

    split_array = np.split(array, len(keys), axis=-1)
    return {key: split_array[i] for i, key in enumerate(keys)}


def all_gather(
    tensor: paddle.Tensor, concat: bool = True, axis: int = 0
) -> Union[paddle.Tensor, List[paddle.Tensor]]:
    """Gather tensor from all devices, concatenate them along given axis if specified.

    Args:
        tensor (paddle.Tensor): Tensor to be gathered from all GPUs.
        concat (bool, optional): Whether to concatenate gathered Tensors. Defaults to True.
        axis (int, optional): Axis which concatenated along. Defaults to 0.

    Returns:
        Union[paddle.Tensor, List[paddle.Tensor]]: Gathered Tensors
    """
    result = []
    paddle.distributed.all_gather(result, tensor)
    if concat:
        return paddle.concat(result, axis)
    return result


def convert_to_array(dict_: Dict[str, np.ndarray], keys: Tuple[str, ...]) -> np.ndarray:
    """Concatenate arrays in axis -1 in order of given keys.

    Args:
        dict_ (Dict[str, np.ndarray]): Dict contains arrays.
        keys (Tuple[str, ...]): Concatenate keys used in concatenation.

    Returns:
        np.ndarray: Concatenated array.
    """
    return np.concatenate([dict_[key] for key in keys], axis=-1)


def concat_dict_list(
    dict_list: Tuple[Dict[str, np.ndarray], ...]
) -> Dict[str, np.ndarray]:
    """Concatenate arrays in tuple of dicts at axis 0.

    Args:
        dict_list (Tuple[Dict[str, np.ndarray], ...]): Tuple of dicts.

    Returns:
        Dict[str, np.ndarray]: A dict with concatenated arrays for each key.
    """
    ret = {}
    for key in dict_list[0].keys():
        ret[key] = np.concatenate([_dict[key] for _dict in dict_list], axis=0)
    return ret


def stack_dict_list(
    dict_list: Tuple[Dict[str, np.ndarray], ...]
) -> Dict[str, np.ndarray]:
    """Stack arrays in tuple of dicts at axis 0.

    Args:
        dict_list (Tuple[Dict[str, np.ndarray], ...]): Tuple of dicts.

    Returns:
        Dict[str, np.ndarray]: A dict with stacked arrays for each key.
    """
    ret = {}
    for key in dict_list[0].keys():
        ret[key] = np.stack([_dict[key] for _dict in dict_list], axis=0)
    return ret


def typename(obj: object) -> str:
    """Return type name of given object.

    Args:
        obj (object): Python object which is instantiated from a class.

    Returns:
        str: Class name of given object.
    """
    return obj.__class__.__name__


def combine_array_with_time(x: np.ndarray, t: Tuple[int, ...]) -> np.ndarray:
    """Combine given data x with time sequence t.
    Given x with shape (N, D) and t with shape (T, ),
    this function will repeat t_i for N times and will concat it with data x for each t_i in t,
    finally return the stacked result, whic is of shape (NxT, D+1).

    Args:
        x (np.ndarray): Points data with shape (N, D).
        t (Tuple[int, ...]): Time sequence with shape (T, ).

    Returns:
        np.ndarray: Combined data with shape of (NxT, D+1).
    """
    nx = len(x)
    tx = []
    for ti in t:
        tx.append(
            np.hstack(
                (np.full([nx, 1], float(ti), dtype=paddle.get_default_dtype()), x)
            )
        )
    tx = np.vstack(tx)
    return tx


def set_random_seed(seed: int):
    """Set numpy, random, paddle random_seed to given seed.

    Args:
        seed (int): Random seed.
    """
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_on_eval_mode(func: Callable) -> Callable:
    """A decorator automatically running given class method in eval mode and keep
    training state unchanged after function finished.

    Args:
        func (Callable): Class method which is expected running in eval mode.

    Returns:
        Callable: Decorated class method.
    """

    @functools.wraps(func)
    def function_with_eval_state(self, *args, **kwargs):
        # log original state
        train_state = self.model.training

        # switch to eval mode
        if train_state:
            self.model.eval()

        # run func in eval mode
        result = func(self, *args, **kwargs)

        # restore state
        if train_state:
            self.model.train()

        return result

    return function_with_eval_state
