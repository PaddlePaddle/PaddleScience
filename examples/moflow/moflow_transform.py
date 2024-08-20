# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2020 Chengxi Zang

import json

import numpy as np

from ppsci.utils import logger

zinc250_atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]


def one_hot(data, out_size=9, num_max_id=5):
    assert tuple(data.shape)[0] == out_size
    b = np.zeros((out_size, num_max_id))
    indices = np.where(data >= 6, data - 6, num_max_id - 1)
    b[np.arange(out_size), indices] = 1
    return b


def transform_fn(data):
    """
    :param data: ((9,), (4,9,9), (15,))
    :return:
    """
    node, adj, label = data
    node = one_hot(node).astype(np.float32)
    adj = np.concatenate(
        [adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)], axis=0
    ).astype(np.float32)
    return node, adj, label


def one_hot_zinc250k(data, out_size=38):
    num_max_id = len(zinc250_atomic_num_list)
    assert tuple(data.shape)[0] == out_size
    b = np.zeros((out_size, num_max_id), dtype=np.float32)
    for i in range(out_size):
        ind = zinc250_atomic_num_list.index(data[i])
        b[i, ind] = 1.0
    return b


def transform_fn_zinc250k(data):
    node, adj, label = data
    node = one_hot_zinc250k(node).astype(np.float32)
    adj = np.concatenate(
        [adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)], axis=0
    ).astype(np.float32)
    return node, adj, label


def get_val_ids(file_path, data_name):
    logger.message("loading train/valid split information from: {}".format(file_path))
    with open(file_path) as json_data:
        data = json.load(json_data)
    if data_name == "qm9":
        val_ids = [(int(idx) - 1) for idx in data["valid_idxs"]]
    elif data_name == "zinc250k":
        val_ids = [(idx - 1) for idx in data]
    return val_ids
