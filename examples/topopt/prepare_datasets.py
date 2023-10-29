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

import h5py
import numpy as np


def generate_train_test(data_path, train_test_ratio, n_sample):
    h5data = h5py.File(data_path, "r")
    X = h5data["iters"]
    Y = h5data["targets"]
    idx = np.arange(10000)
    np.random.shuffle(idx)
    train_idx = idx <= train_test_ratio * n_sample
    if train_test_ratio == 1:
        X_train = []
        Y_train = []
        for i in range(10000):
            if train_idx[i]:
                X_train.append(np.array(X[i]))
                Y_train.append(np.array(Y[i]))
        return X_train, Y_train
    else:
        X_train = []
        X_test = []
        Y_train = []
        Y_test = []
        for i in range(10000):
            if train_idx[i]:
                X_train.append(np.array(X[i]))
                Y_train.append(np.array(Y[i]))
            else:
                X_test.append(np.array(X[i]))
                Y_test.append(np.array(Y[i]))
        return X_train, Y_train, X_test, Y_test
