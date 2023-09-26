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

import collections
import functools
import os
import random
from typing import Callable
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import paddle
from matplotlib import pyplot as plt
from paddle import distributed as dist

__all__ = [
    "all_gather",
    "AverageMeter",
    "PrettyOrderedDict",
    "Prettydefaultdict",
    "concat_dict_list",
    "convert_to_array",
    "convert_to_dict",
    "stack_dict_list",
    "cartesian_product",
    "combine_array_with_time",
    "set_random_seed",
    "run_on_eval_mode",
    "run_at_rank0",
    "plot_curve",
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
        self.history = []

    def update(self, val, n=1):
        """Update"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.history.append(val)

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
    result: List[paddle.Tensor] = []
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
    dict_list: Sequence[Dict[str, np.ndarray]]
) -> Dict[str, np.ndarray]:
    """Concatenate arrays in tuple of dicts at axis 0.

    Args:
        dict_list (Sequence[Dict[str, np.ndarray]]): Sequence of dicts.

    Returns:
        Dict[str, np.ndarray]: A dict with concatenated arrays for each key.
    """
    ret = {}
    for key in dict_list[0].keys():
        ret[key] = np.concatenate([_dict[key] for _dict in dict_list], axis=0)
    return ret


def stack_dict_list(
    dict_list: Sequence[Dict[str, np.ndarray]]
) -> Dict[str, np.ndarray]:
    """Stack arrays in tuple of dicts at axis 0.

    Args:
        dict_list (Sequence[Dict[str, np.ndarray]]): Sequence of dicts.

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
    finally return the stacked result, which is of shape (NxT, D+1).

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


def cartesian_product(*arrays: np.ndarray) -> np.ndarray:
    """Cartesian product for input sequence of array(s).
    Reference: https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points

    Assume input arrays shape are: (N_1,), (N_2,), (N_3,), ..., (N_M,),
    then the cartesian product result will be shape of (N_1*N_2*N_3*...*N_M, M).

    Returns:
        np.ndarray: Cartesian product result of shape (N_1*N_2*N_3*...*N_M, M).

    Examples:
        >>> t = np.array([1, 2])
        >>> x = np.array([10, 20])
        >>> y = np.array([100, 200])
        >>> txy = cartesian_product(t, x, y)
        >>> print(txy)
        [[  1  10 100]
         [  1  10 200]
         [  1  20 100]
         [  1  20 200]
         [  2  10 100]
         [  2  10 200]
         [  2  20 100]
         [  2  20 200]]
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


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


def run_at_rank0(func: Callable) -> Callable:
    """A decorator that allow given function run only at rank 0 to avoid
    multiple logs or other events. Usually effected in distributed environment.

    Args:
        func (Callable): Given function.

    Returns:
        Callable: Wrappered function which will only run at at rank 0,
            skipped at other rank.
    """

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        if dist.get_rank() == 0:
            return func(*args, **kwargs)

    return wrapped_func


def plot_curve(
    data: Dict[str, List],
    xlabel: str = "X",
    ylabel: str = "Y",
    output_dir: str = "./output/",
    smooth_step: int = 1,
    use_semilogy: bool = False,
) -> None:
    """Plotting curve.

    Args:
        data (Dict[str, List]): Dict of all data, keys are curves' name.
        xlabel (str, optional): Label of x-axis. Defaults to "X".
        ylabel (str, optional): Label of y-axis. Defaults to "Y".
        output_dir (str, optional): Output directory of figure. Defaults to "./output/".
        smooth_step (int, optional): How many points are squeezed to one point to smooth the curve. Defaults to 1.
        use_semilogy (bool, optional): Whether to set non-uniform coordinates for the y-axis. Defaults to False.
    """
    data_arr = np.concatenate(
        [np.asarray(arr).reshape(-1, 1) for arr in data.values()], axis=1
    )

    # smooth
    if data_arr.shape[0] % smooth_step != 0:
        data_arr = np.reshape(
            data_arr[: -(data_arr.shape[0] % smooth_step), :],
            (-1, smooth_step, data_arr.shape[1]),
        )
    else:
        data_arr = np.reshape(data_arr, (-1, smooth_step, data_arr.shape[1]))
    data_arr = np.mean(data_arr, axis=1)

    # plot
    plt.figure()
    for i in range(data_arr.shape[1]):
        if use_semilogy:
            plt.semilogy(np.arange(data_arr.shape[0]) * smooth_step, data_arr[:, i])
        else:
            plt.plot(np.arange(data_arr.shape[0]) * smooth_step, data_arr[:, i])
    plt.legend(
        list(data.keys()),
        loc="lower left",
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.yticks(size=10)
    plt.xticks(size=10)

    plt.savefig(os.path.join(output_dir, f"{xlabel}-{ylabel}_curve.jpg"))
    plt.clf()
