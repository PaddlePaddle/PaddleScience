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
import csv
import random

import numpy as np
import paddle
import paddle.distributed as dist

from ppsci.utils import logger

__all__ = [
    "all_gather",
    "AverageMeter",
    "PrettyOrderedDict",
    "Prettydefaultdict",
    "concat_dict_list",
    "convert_to_array",
    "convert_to_dict",
    "load_csv_file",
    "stack_dict_list",
    "combine_array_with_time",
    "set_random_seed",
]


class AverageMeter(object):
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
        """reset"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """update"""
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


def convert_to_dict(array, keys):
    if array.shape[-1] != len(keys):
        raise ValueError(
            f"dim of array({array.shape[-1]}) must equal to " f"len(keys)({len(keys)})"
        )

    split_array = np.split(array, len(keys), axis=-1)
    return {key: split_array[i] for i, key in enumerate(keys)}


def all_gather(tensor, concat=True, axis=0):
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


def convert_to_array(dict, keys):
    return np.concatenate([dict[key] for key in keys], axis=-1)


def concat_dict_list(dict_list):
    ret = {}
    for key in dict_list[0].keys():
        ret[key] = np.concat([_dict[key] for _dict in dict_list], axis=0)
    return ret


def stack_dict_list(dict_list):
    ret = {}
    for key in dict_list[0].keys():
        ret[key] = np.stack([_dict[key] for _dict in dict_list], axis=0)
    return ret


def typename(object):
    return object.__class__.__name__


def combine_array_with_time(x, t):
    nx = len(x)
    tx = []
    for ti in t:
        tx.append(np.hstack((np.full([nx, 1], float(ti), dtype="float32"), x)))
    tx = np.vstack(tx)
    return tx


def load_csv_file(file_path, keys, alias_dict=None, encoding="utf-8"):
    try:
        if alias_dict is None:
            alias_dict = {}
        # check if all keys in alias_dict are valid
        for original_key in alias_dict:
            if original_key not in keys:
                raise ValueError(
                    f"key({original_key}) in alias_dict "
                    f"is not found in keys({keys})"
                )
        # read all data from csv file
        with open(file_path, "r", encoding=encoding) as csv_file:
            reader = csv.DictReader(csv_file)
            raw_data_dict = collections.defaultdict(list)
            for line_idx, line_dict in enumerate(reader):
                for key, value in line_dict.items():
                    raw_data_dict[key].append(value)

                # check if all keys are available at first line
                if line_idx == 0:
                    for require_key in keys:
                        if require_key not in line_dict:
                            raise KeyError(
                                f"key({require_key}) "
                                f"not found in csvfile({file_path})"
                            )

            data_dict = {}
            for key, value in raw_data_dict.items():
                if key in alias_dict:
                    data_dict[alias_dict[key]] = np.array(value, "float32").reshape(
                        [-1, 1]
                    )
                else:
                    data_dict[key] = np.array(value, "float32").reshape([-1, 1])

        return data_dict
    except Exception as e:
        logger.error(f"{repr(e)}, {file_path} isn't a valid csv file.")
        raise


def set_random_seed(seed):
    rank = dist.get_rank()
    paddle.seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
