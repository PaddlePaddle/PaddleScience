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
from typing import Tuple
from typing import Union

import numpy as np


def uniform_sampler() -> Callable[[], int]:
    """Generate uniform sampling function from 1 to 99

    Returns:
        sampler (Callable[[], int]): uniform sampling from 1 to 99
    """
    return lambda: np.random.randint(1, 99)


def poisson_sampler(lam: int) -> Callable[[], int]:
    """Generate poisson sampling function with parameter lam with range 1 to 99

    Args:
        lam (int): poisson rate parameter

    Returns:
        sampler (Callable[[], int]): poisson sampling function with parameter lam with range 1 to 99
    """

    def func():
        iter_ = max(np.random.poisson(lam), 1)
        iter_ = min(iter_, 99)
        return iter_

    return func


def generate_sampler(sampler_type: str = "Fixed", num: int = 0) -> Callable[[], int]:
    """Generate sampler for the number of initial iteration steps

    Args:
        sampler_type (str): "Poisson" for poisson sampler; "Uniform" for uniform sampler; "Fixed" for choosing a fixed number of initial iteration steps.
        num (int): If `sampler_type` == "Poisson", `num` specifies the poisson rate parameter; If `sampler_type` == "Fixed", `num` specifies the fixed number of initial iteration steps.

    Returns:
        sampler (Callable[[], int]): sampler for the number of initial iteration steps
    """
    if sampler_type == "Poisson":
        return poisson_sampler(num)
    elif sampler_type == "Uniform":
        return uniform_sampler()
    else:
        return lambda: num


def generate_train_test(
    data_iters: np.ndarray,
    data_targets: np.ndarray,
    train_test_ratio: float,
    n_sample: int,
) -> Union[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """Generate training and testing set

    Args:
        data_iters (np.ndarray): data with 100 channels corresponding to the results of 100 steps of SIMP algorithm
        data_targets (np.ndarray): final optimization solution given by SIMP algorithm
        train_test_ratio (float): split ratio of training and testing sets, if `train_test_ratio` = 1 then only return training data
        n_sample (int): number of total samples in training and testing sets to be sampled from the h5 dataset

    Returns:
        Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]: if `train_test_ratio` = 1, return (train_inputs, train_labels), else return (train_inputs, train_labels, test_inputs, test_labels)
    """
    n_obj = len(data_iters)
    idx = np.arange(n_obj)
    np.random.shuffle(idx)
    train_idx = idx[: int(train_test_ratio * n_sample)]
    if train_test_ratio == 1.0:
        return data_iters[train_idx], data_targets[train_idx]

    test_idx = idx[int(train_test_ratio * n_sample) :]
    train_iters = data_iters[train_idx]
    train_targets = data_targets[train_idx]
    test_iters = data_iters[test_idx]
    test_targets = data_targets[test_idx]
    return train_iters, train_targets, test_iters, test_targets


def augmentation(
    input_dict: Dict[str, np.ndarray],
    label_dict: Dict[str, np.ndarray],
    weight_dict: Dict[str, np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Apply random transformation from D4 symmetry group

    Args:
        input_dict (Dict[str, np.ndarray]): input dict of np.ndarray size `(batch_size, any, height, width)`
        label_dict (Dict[str, np.ndarray]): label dict of np.ndarray size `(batch_size, 1, height, width)`
        weight_dict (Dict[str, np.ndarray]): weight dict if any
    """
    inputs = input_dict["input"]
    labels = label_dict["output"]
    assert len(inputs.shape) == 3
    assert len(labels.shape) == 3

    # random horizontal flip
    if np.random.random() > 0.5:
        inputs = np.flip(inputs, axis=2)
        labels = np.flip(labels, axis=2)
    # random vertical flip
    if np.random.random() > 0.5:
        inputs = np.flip(inputs, axis=1)
        labels = np.flip(labels, axis=1)
    # random 90* rotation
    if np.random.random() > 0.5:
        new_perm = list(range(len(inputs.shape)))
        new_perm[-2], new_perm[-1] = new_perm[-1], new_perm[-2]
        inputs = np.transpose(inputs, new_perm)
        labels = np.transpose(labels, new_perm)

    return {"input": inputs}, {"output": labels}, weight_dict
