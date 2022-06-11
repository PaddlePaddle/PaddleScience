# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import scipy.io as io


class Dataset:
    def __init__(self, t_range, Neta, Nf_train):
        self.t_range = t_range
        self.Neta = Neta
        self.Nf_train = Nf_train

    def build_data(self):
        data = io.loadmat('./VIV_Training.mat')
        t, eta, f = data['t'], data['eta_y'], data['f_y']

        t_0 = t.min(0)
        tmin = np.reshape(t_0, (-1, 1))
        t_1 = t.max(0)
        tmax = np.reshape(t_1, (-1, 1))

        N = t.shape[0]
        N = 160
        idx = np.random.choice(N, self.Neta, replace=False)
        t_eta = t[idx]
        eta = eta[idx]

        #idx = np.random.choice(N, self.Nf_train, replace=False)
        t_f = t[idx]
        f = f[idx]

        return t_eta, eta, t_f, f, tmin, tmax
