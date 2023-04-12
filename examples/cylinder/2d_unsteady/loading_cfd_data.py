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

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DataLoader:
    def __init__(
        self, path=None, N_f=20000, N_b=1000, time_start=0, time_end=0.5, time_nsteps=31
    ):
        """
        N_f: Num of residual points
        N_b: Num of boundary points
        time_start: unsteady time start point
        time_end: unsteady time end point
        time_nsteps: interval of time sampling
        """

        self.N_f = N_f
        self.N_b = N_b

        # 0S
        time_start = time_start
        # 0.5S
        time_end = time_end
        time_nsteps = time_nsteps
        self.random_time_steps = None

        self.path = path
        # * 100 to adapt the probe8.*.csv filename
        self.scale = 1
        time_points = (
            np.linspace(time_start, time_end, time_nsteps, endpoint=True) * self.scale
        )
        self.discretized_time = time_points.astype(int)

    def select_discretized_time(self, num_time=20, reused=True):
        # num_time, number of time for training in each epoch
        if self.random_time_steps and reused:
            return self.random_time_steps
        else:
            self.random_time_steps = np.random.choice(self.discretized_time, num_time)
        self.random_time_steps.sort()
        return self.random_time_steps

    def select_ordered_time(self, num_time=20, reused=True):
        # num_time, number of time for training in each epoch
        if self.random_time_steps is not None and reused:
            return self.random_time_steps
        self.discretized_time.sort()
        return self.discretized_time

    def reading_data_from_csv(self, path, filename):
        full_filename = path + filename
        data_pd = pd.read_csv(full_filename, encoding="gbk")
        return data_pd

    def replicate_time_list(self, time_list, domain_shape, spatial_data):
        all_t = []
        count = 0
        for t in time_list:
            tmp_t = [t / self.scale] * domain_shape
            all_t.append(tmp_t)
        replicated_t = np.array(all_t).reshape(-1, 1)

        for index in range(len(spatial_data)):
            all_data = []
            tmp = spatial_data[index].tolist()
            for t in time_list:
                all_data.append(tmp)
            spatial_d = np.array(all_data).reshape(-1, 1)
            spatial_data[index] = spatial_d

        return replicated_t, spatial_data

    def loading_train_inside_domain_data(
        self, time_list, flatten=False, dtype="float32"
    ):
        # load train_domain points
        # domain_train.csv, title is p,U:0,U:1,U:2,Points:0,Points:1,Points:2
        filename = "domain_train.csv"
        path = self.path
        domain_data_from_pd = self.reading_data_from_csv(path, filename)
        # p,U:0,U:1,U:2,Points:0,Points:1,Points:2
        domain_data = np.array(domain_data_from_pd)
        # idx = np.random.choice(domain_data.shape[0], self.N_f , replace=True)
        # domain_data = domain_data[idx]

        # t, x, y
        # x = np.concatenate((x, domain_data[:,4].reshape((-1,1))))
        # y = np.concatenate((y, domain_data[:,5].reshape((-1,1))))

        # t, x, y
        x = domain_data[:, 4].reshape((-1, 1))
        y = domain_data[:, 5].reshape((-1, 1))
        t, xy = self.replicate_time_list(time_list, x.shape[0], [x, y])
        t = t.astype(dtype)
        xy[0] = xy[0].astype(dtype)
        xy[1] = xy[1].astype(dtype)
        print("residual data shape:", t.shape[0])
        if flatten == True:
            return t.flatten(), xy[0].flatten(), xy[1].flatten()
        else:
            return t, xy[0], xy[1]

    def loading_outlet_data(self, time_list, flatten=False, dtype="float32"):
        filename = "domain_outlet.csv"
        path = self.path

        outlet_data_from_pd = self.reading_data_from_csv(path, filename)
        # p,U:0,U:1,U:2,Points:0,Points:1,Points:2
        outlet_data = np.array(outlet_data_from_pd)

        # p, t, x, y
        p = outlet_data[:, 0].reshape((-1, 1))
        x = outlet_data[:, 4].reshape((-1, 1))
        y = outlet_data[:, 5].reshape((-1, 1))

        print("outlet data shape:", outlet_data.shape[0])
        t, pxy = self.replicate_time_list(time_list, outlet_data.shape[0], [p, x, y])

        pxy[0] = pxy[0].astype(dtype)
        pxy[1] = pxy[1].astype(dtype)
        pxy[2] = pxy[2].astype(dtype)
        t = t.astype(dtype)

        if flatten == True:
            return pxy[0].flatten(), t.flatten(), pxy[1].flatten(), pxy[2].flatten()
        else:
            return pxy[0], t, pxy[1], pxy[2]

    def loading_inlet_data(self, time_list, path):
        #  title is p,U:0,U:1,U:2,Points:0,Points:1,Points:2
        filename = "domain_inlet.csv"

        # u, v, x, y
        return self.loading_data(time_list, path, filename)

    def loading_cylinder_data(self, time_list, path):
        #  title is p,U:0,U:1,U:2,Points:0,Points:1,Points:2
        filename = "domain_cylinder.csv"

        # u, v, x, y
        return self.loading_data(time_list, path, filename)

    def loading_side_data(self, time_list, path):
        #  title is p,U:0,U:1,U:2,Points:0,Points:1,Points:2
        filename = "domain_side.csv"

        # u, v, x, y
        return self.loading_data(time_list, path, filename)

    def loading_boundary_data(
        self, time_list, num_random=None, flatten=False, dtype="float32"
    ):
        inlet_bc = self.loading_inlet_data(time_list, self.path)
        # side_bc = self.loading_side_data(time_list, self.path)
        cylinder_bc = self.loading_cylinder_data(time_list, self.path)

        u = np.concatenate((inlet_bc[0], cylinder_bc[0])).astype(dtype)
        v = np.concatenate((inlet_bc[1], cylinder_bc[1])).astype(dtype)
        x = np.concatenate((inlet_bc[2], cylinder_bc[2])).astype(dtype)
        y = np.concatenate((inlet_bc[3], cylinder_bc[3])).astype(dtype)
        t, uvxy = self.replicate_time_list(time_list, u.shape[0], [u, v, x, y])

        t = t.astype(dtype)
        uvxy[0] = uvxy[0].astype(dtype)
        uvxy[1] = uvxy[1].astype(dtype)
        uvxy[2] = uvxy[2].astype(dtype)
        uvxy[3] = uvxy[3].astype(dtype)

        if flatten == True:
            return (
                uvxy[0].flatten(),
                uvxy[1].flatten(),
                t.flatten(),
                uvxy[2].flatten(),
                uvxy[3].flatten(),
            )
        else:
            return uvxy[0], uvxy[1], t, uvxy[2], uvxy[3]

    def loading_data(self, time_list, path, filename, num_random=None):
        # boudnary datra: cylinder/inlet/side
        # domain_xx.csv, title is p,U:0,U:1,U:2,Points:0,Points:1,Points:2
        # path = self.path

        boundary_data = None
        data_from_pd = self.reading_data_from_csv(path, filename)
        # p,U:0,U:1,U:2,Points:0,Points:1,Points:2
        boundary_data = np.array(data_from_pd)

        if num_random:
            idx = np.random.choice(boundary_data.shape[0], num_random, replace=False)
            boundary_data = boundary_data[idx]
        # u, v, t, x, y
        u = boundary_data[:, 1].reshape((-1, 1))
        v = boundary_data[:, 2].reshape((-1, 1))
        x = boundary_data[:, 4].reshape((-1, 1))
        y = boundary_data[:, 5].reshape((-1, 1))

        print("boundary  data shape:", boundary_data.shape[0])
        return u, v, x, y

    def loading_supervised_data(self, time_list, flatten=False, dtype="float32"):
        path = self.path

        supervised_data = None
        full_supervised_data = None
        for time in time_list:
            filename = "/probe/probe0." + str(time) + ".csv"
            data_from_pd = self.reading_data_from_csv(path, filename)
            # p,U:0,U:1,U:2,Points:0,Points:1,Points:2

            t_len = data_from_pd.shape[0]
            supervised_t = np.array([time / self.scale] * t_len).reshape((-1, 1))
            if supervised_data is None:
                supervised_data = np.array(data_from_pd)
                t_data = np.concatenate((supervised_t, supervised_data), axis=1)
                full_supervised_data = t_data
            else:
                next_data = np.array(data_from_pd)
                t_data = np.concatenate((supervised_t, next_data), axis=1)
                full_supervised_data = np.concatenate((full_supervised_data, t_data))

        print("supervised data shape:", full_supervised_data.shape[0])
        # p, u, v, t, x, y
        p = full_supervised_data[:, 1].reshape((-1, 1)).astype(dtype)
        u = full_supervised_data[:, 2].reshape((-1, 1)).astype(dtype)
        v = full_supervised_data[:, 3].reshape((-1, 1)).astype(dtype)
        t = full_supervised_data[:, 0].reshape((-1, 1)).astype(dtype)
        x = full_supervised_data[:, 6].reshape((-1, 1)).astype(dtype)
        y = full_supervised_data[:, 7].reshape((-1, 1)).astype(dtype)

        if flatten == True:
            return (
                p.flatten(),
                u.flatten(),
                v.flatten(),
                t.flatten(),
                x.flatten(),
                y.flatten(),
            )
        else:
            return p, u, v, t, x, y

    def loading_initial_data(self, time_list, flatten=False, dtype="float32"):
        # "p","U:0","U:1","U:2","vtkOriginalPointIds","Points:0","Points:1","Points:2"
        path = self.path

        initial_data = None
        time = time_list[0]
        filename = "/initial/ic0." + str(time) + ".csv"
        data_from_pd = self.reading_data_from_csv(path, filename)
        t_len = data_from_pd.shape[0]
        initial_t = np.array([time / self.scale] * t_len).reshape((-1, 1))
        initial_data = np.array(data_from_pd)
        initial_t_data = np.concatenate((initial_t, initial_data), axis=1)

        print("initial data shape:", initial_data.shape[0])
        # p, u, v, t, x, y
        p = initial_t_data[:, 1].reshape((-1, 1)).astype(dtype)
        u = initial_t_data[:, 2].reshape((-1, 1)).astype(dtype)
        v = initial_t_data[:, 3].reshape((-1, 1)).astype(dtype)
        t = initial_t_data[:, 0].reshape((-1, 1)).astype(dtype)
        x = initial_t_data[:, 5].reshape((-1, 1)).astype(dtype)
        y = initial_t_data[:, 6].reshape((-1, 1)).astype(dtype)
        if flatten == True:
            return (
                p.flatten(),
                u.flatten(),
                v.flatten(),
                t.flatten(),
                x.flatten(),
                y.flatten(),
            )
        else:
            return p, u, v, t, x, y
