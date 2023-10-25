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

import numpy as np
import paddle


def augmentation(input_dict, label_dict, weight_dict=None):
    """Apply random transformation from D4 symmetry group

    Args:
        input_dict (Dict[str, Tensor]): input dict of Tensor size `(batch_size, any, height, width)`
        label_dict (Dict[str, Tensor]): label dict of Tensor size `(batch_size, 1, height, width)`
        weight_dict (Dict[str, Tensor]): weight dict if any

    Returns:
        Tuples: (transformed_input_dict, transformed_label_dict, transformed_weight_dict)
    """
    X = paddle.to_tensor(input_dict["input"])
    Y = paddle.to_tensor(label_dict["output"])
    n_obj = len(X)
    indices = np.arange(n_obj)
    np.random.shuffle(indices)

    if len(X.shape) == 3:
        # random horizontal flip
        if np.random.random() > 0.5:
            X = paddle.flip(X, axis=2)
            Y = paddle.flip(Y, axis=2)
        # random vertical flip
        if np.random.random() > 0.5:
            X = paddle.flip(X, axis=1)
            Y = paddle.flip(Y, axis=1)
        # random 90* rotation
        if np.random.random() > 0.5:
            new_perm = list(range(len(X.shape)))
            new_perm[1], new_perm[2] = new_perm[2], new_perm[1]
            X = paddle.transpose(X, perm=new_perm)
            Y = paddle.transpose(Y, perm=new_perm)
        X = X.reshape([1] + X.shape)
        Y = Y.reshape([1] + Y.shape)
    else:
        # random horizontal flip
        batch_size = X.shape[0]
        mask = np.random.random(size=batch_size) > 0.5
        X[mask] = paddle.flip(X[mask], axis=3)
        Y[mask] = paddle.flip(Y[mask], axis=3)
        # random vertical flip
        mask = np.random.random(size=batch_size) > 0.5
        X[mask] = paddle.flip(X[mask], axis=2)
        Y[mask] = paddle.flip(Y[mask], axis=2)
        # random 90* rotation
        mask = np.random.random(size=batch_size) > 0.5
        new_perm = list(range(len(X.shape)))
        new_perm[2], new_perm[3] = new_perm[3], new_perm[2]
        X[mask] = paddle.transpose(X[mask], perm=new_perm)
        Y[mask] = paddle.transpose(Y[mask], perm=new_perm)

    return {"input": X}, {"output": Y}, weight_dict
