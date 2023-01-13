# import glob
# import os
# import time

# import numpy as np
# import pandas as pd

# dir_name = '/workspace/hesensen/PaddleScience_dev_3d/examples/cylinder/3d_unsteady_continuous/data/all/'
# #file_pattern = 'point_70.000000_9.000000_27.000000.plt'


# scale = 10.0
# pressure_base = 101325.0
# # time_step = 1 # need change to 0.1
# time_start_per_file = 2000.0
# domain_coordinate_interval_dict = {1:[0,800], 2:[0,400], 3:[0,300]}


# def normalize(max_domain, min_domain, array, index):
#     #array_min = min(array[:,index])
#     #array_max = max(array[:,index])
#     diff = max_domain - min_domain
#     if abs(diff) < 0.00001:
#         array[:,index] = 0.0
#     else:
#         array[:,index] = (array[:, index] - min_domain)/diff


# def scale_value(array, scale, index):
#     array[:, index] = array[:, index] * scale


# #'2002.000000,10.000000,14.000000,8.000000,0.937419,-0.129196,-0.008992,101325.328523,4.933044'
# def load_ic_data(t):
#     print("Loading IC data ......")
#     # Get list of all files in a given directory sorted by name
#     list_of_files = sorted(filter(os.path.isfile, glob.glob(dir_name + '*')))
#     # Iterate over sorted list of files and print the file paths one by one.
#     # (63916, 9)
#     flow_array = None
#     for file_path in list_of_files:
#         # print(file_path)
#         # filename = file_path.strip('.plt').split('_')
#         # data = pd.read_table(file_path, sep=',', header=None)
#         with open(file_path, "r") as fp:
#             # line = fp.readline()
#             line = fp.readlines(0)[1]   # 2000.1开始
#             data_list = line.strip("\n").split(",")
#         txyz_uvwpe = np.array([data_list], dtype="float32")
#         if flow_array is None:
#             flow_array = txyz_uvwpe
#         else:
#             flow_array = np.concatenate((flow_array, txyz_uvwpe))

#     print("load_ic_data", flow_array.shape)
#     # Set ic t=0
#     flow_array[:, 0] = 0
#     # Scale xyz by 10.0
#     for i in [1, 2, 3]:
#         scale_value(flow_array, scale, i)
#     # Normalize x,y,z to [0,1]
#     for coordinate, interval in domain_coordinate_interval_dict.items():
#         min_domain = interval[0]
#         max_domain = interval[1]
#         normalize(min_domain, max_domain, flow_array, coordinate)
#     # Cast pressure baseline
#     flow_array[:, 7] = flow_array[:, 7] - pressure_base

#     # txyzuvwpe
#     print("IC data shape: {}".format(flow_array.shape))
#     return flow_array.astype(np.float32)


# def load_supervised_data(t_start, t_end, t_step, t_ic, num_points):
#     print("Loading Supervised data ......")
#     # row_index = int((t_start - time_start_per_file) / time_step)
#     row_index = round((t_start - time_start_per_file) / t_step)  # time_step=0.1,row_index从第2行开始
#     # row_length = int((t_end - t_start) / time_step)
#     row_length = int((t_end - t_start + t_step) / t_step)
#     print(row_index, row_length) # 1 5

#     # Get list of all files in a given directory sorted by name
#     list_of_files = sorted(filter(os.path.isfile, glob.glob(dir_name + '*')))
#     random_files = np.random.permutation(list_of_files)


#     # Iterate over sorted list of files and print the file paths one by one.
#     # (63916, 9)
#     flow_array = None
#     for file_path in random_files[:num_points]: # 读每一个坐标上的文件
#         # print(file_path)
#         # filename = file_path.strip('.plt').split('_')
#         # data = pd.read_table(file_path, sep=',', header=None)
#         with open(file_path) as fp:
#             for index, line in enumerate(fp): # 遍历每一个文件在不同时间上的值
#                 if index in range(row_index, row_index + row_length): # [1, 6]，只看t_start~t_end之间的值
#                     data_list = line.strip('\n').split(',')
#                     txyz_uvwpe = np.array([data_list], dtype=float)
#                     if flow_array is None:
#                         flow_array = txyz_uvwpe
#                     else:
#                         flow_array = np.concatenate((flow_array, txyz_uvwpe))

#     # Normalize t to [0, ..]
#     flow_array[:, 0] = (flow_array[:, 0] - t_ic) / t_step
#     # print(flow_array.shape) # (8000, 9)
#     # print(flow_array[:, 0].min(), flow_array[:, 0].max(), t_ic) # 0.9999999999990905 4.0000000000009095 2000.0
#     # exit(0)

#     # Scale xyz by 10.0
#     for i in [1, 2, 3]:
#         scale_value(flow_array, scale, i)
#     # Normalize x,y,z to [0,1]
#     for coordinate, interval in domain_coordinate_interval_dict.items():
#         min_domain = interval[0]
#         max_domain = interval[1]
#         normalize(min_domain, max_domain, flow_array, coordinate)
#     # Cast pressure baseline
#     flow_array[:, 7] = flow_array[:, 7] - pressure_base

#     # txyzuvwpe
#     print("Supervised data shape: {}".format(flow_array.shape))
#     return flow_array.astype(np.float32)


# if __name__ == '__main__':
#     start = time.perf_counter()
#     ic_t = 2000.0
#     #flow_array = load_ic_data(ic_t)
#     ic_array = load_ic_data(ic_t)
#     flow_array = load_supervised_data(2000.0, 2000.4, 100)
#     end = time.perf_counter()
#     print("spent:{}s".format(end-start))

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
"""
Created in Oct. 2022
@author: Hui Xiang, Yanbo Zhang
"""

import os
import time
import glob
import numpy as np
import pandas as pd
import math
import copy
from pyevtk.hl import pointsToVTK

dir_ic = '/workspace/hesensen/PaddleScience_cqp/examples/cylinder/3d_unsteady_continuous/data/ic_data/'
# dir_sp = '/home/aistudio/work/data/supervised_data/'
dir_sp_mode1 = '/workspace/hesensen/PaddleScience_cqp/examples/cylinder/3d_unsteady_continuous/data/new_sp_data/'

#file_pattern = 'point_70.000000_9.000000_27.000000.dat'


scale = 1.0
pressure_base = 101325.0
# time_step = 1 # need change to 0.1
time_start_per_file = 200000
domain_coordinate_interval_dict = {1: [0,1600], 2: [0,800], 3: [0,320]}

def normalize(max_domain, min_domain, array, index):
    #array_min = min(array[:,index])
    #array_max = max(array[:,index])
    diff = max_domain - min_domain
    if abs(diff) < 0.0000001:
        array[:, index] = 0.0
    else:
        array[:, index] = (array[:, index] - min_domain) / diff


def scale_value(array, scale, index):
    array[:, index] = array[:, index] * scale

# 2.000000000000000000e+05,1.000000000000000000e+01,1.580000000000000000e+02,2.100000000000000000e+02,9.929000000000000326e-02,-1.369999999999999961e-04,2.400000000000000061e-05,1.013256406250000000e+05
def load_ic_data(t):
    print("Loading IC data ......")
    # Get list of all files in a given directory sorted by name
    list_of_files = sorted(filter(os.path.isfile, glob.glob(dir_ic + '*')))

    # Iterate over sorted list of files and print the file paths one by one.
    flow_list = []
    flow_array = None
    for file_path in list_of_files:
        with open(file_path) as fp:
            line = fp.readline()
            data_list = line.strip('\n').split(',')
        flow_list.append(data_list)
    flow_array = np.array(flow_list, dtype=float)

    # Set ic t=0
    flow_array[:, 0] = 0

    # Normalize x,y,z to [0,1]
    for  coordinate, interval in domain_coordinate_interval_dict.items():
        min_domain = interval[0]
        max_domain = interval[1]
        normalize(max_domain, min_domain, flow_array, coordinate)

    # Cast pressure baseline
    # flow_array[:,7] = flow_array[:,7] - pressure_base
    flow_array[:, 7] = flow_array[:, 7] / pressure_base - 1

    flow_array = flow_array.astype(np.float32)
    # txyzuvwpe
    print("IC data shape: {}".format(flow_array.shape))
    t = flow_array[:, 0].reshape((-1, 1))
    x = flow_array[:, 1].reshape((-1, 1))
    y = flow_array[:, 2].reshape((-1, 1))
    z = flow_array[:, 3].reshape((-1, 1))
    u = flow_array[:, 4].reshape((-1, 1))
    v = flow_array[:, 5].reshape((-1, 1))
    w = flow_array[:, 6].reshape((-1, 1))
    p = flow_array[:, 7].reshape((-1, 1))
    #return np.transpose(flow_array.astype(np.float32))
    return np.concatenate([t, x, y, z, u, v, w, p], axis=1)
    # return t, x, y, z, u, v, w, p


# t_step = 50, t_start=200050, t_end=200250, take 3 steps
def load_supervised_data(t_start, t_end, t_step, t_ic, num_points):
    print("Loading Supervised data ......")
    row_index = round((t_start - time_start_per_file) / t_step)  # row_index = 1, strat from second row
    row_length = int((t_end - t_start) / t_step) + 1

    # Get list of all files in a given directory sorted by name
    list_of_files = sorted(filter(os.path.isfile, glob.glob(dir_sp_mode1 + '*')))
    random_files = np.random.permutation(list_of_files)
    # t_step = 50
    # Iterate over sorted list of files and print the file paths one by one.
    flow_list = []
    flow_array = None
    # for file_path in random_files[:num_points]:
    # read all supervised_data
    for file_path in random_files:
        with open(file_path) as fp:
            for index, line in enumerate(fp):
                if index in range(row_index, row_index + row_length):
                    data_list = line.strip('\n').split(',')
                    flow_list.append(data_list)
    flow_array = np.array(flow_list, dtype=float)

    # Normalize x,y,z to [0,1]
    for coordinate, interval in domain_coordinate_interval_dict.items():
        min_domain = interval[0]
        max_domain = interval[1]
        normalize(max_domain, min_domain, flow_array, coordinate)

    # Cast pressure baseline
    flow_array[:, 7] = flow_array[:, 7] / pressure_base - 1

    flow_array = flow_array.astype(np.float32)

    # txyzuvwpe
    print("Supervised data shape: {}".format(flow_array.shape))
    t = (flow_array[:, 0].reshape((-1, 1)) - t_ic) / t_step
    x = flow_array[:, 1].reshape((-1, 1))
    y = flow_array[:, 2].reshape((-1, 1))
    z = flow_array[:, 3].reshape((-1, 1))
    u = flow_array[:, 4].reshape((-1, 1))
    v = flow_array[:, 5].reshape((-1, 1))
    w = flow_array[:, 6].reshape((-1, 1))
    p = flow_array[:, 7].reshape((-1, 1))
    return np.concatenate([t, x, y, z, u, v, w, p], axis=1)
    # return t, x, y, z, u, v, w, p

if __name__ == '__main__':
    start = time.perf_counter()
    ic_t = 2000.1
    t_step = 0.1
    ic_array = load_ic_data(ic_t)
    flow_array = load_supervised_data(2000.1, 2000.4, t_step, ic_t, 10000)
    _, x, y, z, u, v, w, _ = ic_array
    pointsToVTK("./ini_points", x.flatten(), y.flatten(), z.flatten(), data={"u": u.flatten(), "v": v.flatten(), "w":w.flatten()})
    _, x1, y1, z1, u1, v1, w1, _ = flow_array
    pointsToVTK("./sup_points", x1.flatten(), y1.flatten(), z1.flatten(), data={"u1": u1.flatten(), "v1": v1.flatten(), "w1": w1.flatten()})
    end = time.perf_counter()
    print("spent:{}s".format(end-start))
