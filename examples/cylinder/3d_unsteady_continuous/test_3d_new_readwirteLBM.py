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

import copy
from abc import abstractmethod
from typing import Union

import numpy as np
import paddle
import paddle.distributed as dist
import paddlescience as psci
from paddle.optimizer import lr
from pyevtk.hl import pointsToVTK
from matplotlib import pyplot as plt

import sample_boundary_training_data as sample_data
import vtk
from load_lbm_data import load_ic_data, load_supervised_data, load_all_LBM_data

paddle.seed(42)
np.random.seed(42)
# paddle.enable_static()

# Dimensionless Trick for Navier Stokes equations
Re = 3900
U0 = 0.1
Dcylinder = 80.0
rho = 1.0
nu = rho * U0 * Dcylinder / Re

t_star = Dcylinder / U0 # 800
xyz_star = Dcylinder    # 80
uvw_star = U0           # 0.1
p_star = rho * U0 * U0  # 0.01

# time array
ic_t = 200000.0
t_start = 200000.0
t_end = 204950.0
t_step = 50.0
time_num = int((t_end - t_start) / t_step) + 1
time_list = np.linspace(int((t_start - ic_t) / t_step), int((t_end - ic_t) / t_step), time_num, endpoint=True).astype(int)
time_tmp = time_list * t_step
# time_index = np.random.choice(time_list, int(time_num / 2.5), replace=False)
# time_index.sort()
time_index = np.array(time_list)
time_array = time_index * t_step
print(f"time_num = {time_num}, time_list = {time_list}, time_tmp = {time_tmp}")
print(f"time_index = {time_index}, time_array = {time_array}")

P_txyzuvwp = np.zeros([time_num, 8])

i = 0
for t_i in time_index:
    num_cords, txyz_uvwpe = load_all_LBM_data([t_i], ic_t)
    init_t = txyz_uvwpe[:, 0]; print(f"init_t={init_t.shape} {init_t.mean().item():.10f}")
    init_x = txyz_uvwpe[:, 1]; print(f"init_x={init_x.shape} {init_x.mean().item():.10f}")
    init_y = txyz_uvwpe[:, 2]; print(f"init_y={init_y.shape} {init_y.mean().item():.10f}")
    init_z = txyz_uvwpe[:, 3]; print(f"init_z={init_z.shape} {init_z.mean().item():.10f}")
    init_u = txyz_uvwpe[:, 4]; print(f"init_u={init_u.shape} {init_u.mean().item():.10f}")
    init_v = txyz_uvwpe[:, 5]; print(f"init_v={init_v.shape} {init_v.mean().item():.10f}")
    init_w = txyz_uvwpe[:, 6]; print(f"init_w={init_w.shape} {init_w.mean().item():.10f}")
    init_p = txyz_uvwpe[:, 7]; print(f"init_p={init_p.shape} {init_p.mean().item():.10f}")

    print('num_cords = ', num_cords)

    cords = np.stack((init_x[0:num_cords], init_y[0:num_cords], init_z[0:num_cords]), axis=1)
    solution = np.stack((init_u, init_v, init_w, init_p), axis=1)
    # psci.visu.save_vtk_cord(filename="./vtk/cylinder3d_2023_1_31_LBM", time_array=time_array, cord=cords, data=solution)
    filename = f"./vtk/cylinder3d_2023_1_31_LBM_{t_i}"
    data = solution[0*num_cords: 1*num_cords]
    psci.visu.__save_vtk_raw(filename=filename, cordinate=cords, data=data)
    print(f"Saved: ./vtk/cylinder3d_2023_1_31_LBM_{t_i}")

    P_txyzuvwp[t_i, 0] = init_t[154386]
    P_txyzuvwp[t_i, 1] = init_x[154386]
    P_txyzuvwp[t_i, 2] = init_y[154386]
    P_txyzuvwp[t_i, 3] = init_z[154386]
    P_txyzuvwp[t_i, 4] = init_u[154386]
    P_txyzuvwp[t_i, 5] = init_v[154386]
    P_txyzuvwp[t_i, 6] = init_w[154386]
    P_txyzuvwp[t_i, 7] = init_p[154386]

    # for ii in range(num_cords):
    #     if np.isclose(init_x[ii], 790.0) and np.isclose(init_y[ii], 404.0) and np.isclose(init_z[ii], 160.0):
    #         print('ii = ', ii)
    #         exit(0)

print('P_txyzuvwp[0, 1] = ', P_txyzuvwp[0, 1])
print('P_txyzuvwp[0, 2] = ', P_txyzuvwp[0, 2])
print('P_txyzuvwp[0, 3] = ', P_txyzuvwp[0, 3])

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成figure对象
fig = plt.figure()
# 生成axes对象
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

t = np.linspace(t_start - ic_t, t_end - ic_t, time_num, endpoint=True)
u = P_txyzuvwp[:, 4]
v = P_txyzuvwp[:, 5]
w = P_txyzuvwp[:, 6]
print(P_txyzuvwp.shape, t.shape, u.shape, v.shape, w.shape)
# 绘制散点
axes.plot(t, u, c="red", label=r'$u$', ls='-', alpha=0.6, lw=2, zorder=2) 
axes.plot(t, v, c="green", label=r'$v$', ls='-.', alpha=0.6, lw=2, zorder=2) 
axes.plot(t, w, c="blue", label=r'$w$', ls=':', alpha=1, lw=1, zorder=1) 
# 设置图像标题
axes.set_title("曲线图")

axes.legend()
# 显示图像
plt.show()
plt.savefig('cylinder3d_p154386.jpg')
print('Saved: cylinder3d_p154386.jpg')