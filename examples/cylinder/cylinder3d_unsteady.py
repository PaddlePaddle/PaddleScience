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

# discrete time method

import paddlescience as psci
import numpy as np
import paddle

paddle.seed(1)
np.random.seed(1)

# paddle.enable_static()
paddle.disable_static()

# load real data
def GetRealPhyInfo(time, need_cord=False, need_physic=False):
    real_data = np.load("flow_unsteady_re200/flow_re200_" + str(time) + "_xyzuvwp.npy")
    real_data = real_data.astype(np.float32)
    if need_cord is False and need_physic is False:
        print("Error: you need to get cord or get physic infomation")
        exit()
    elif need_cord is True and need_physic is True:
        return real_data
    elif need_cord is True and need_physic is False:
        return real_data[:, 0:3]
    elif need_cord is False and need_physic is True:
        return real_data[:, 3:7]
    else:
        pass

# get init physic infomation
def GenInitPhyInfo(xyz):
    uvw = np.zeros((len(xyz), 3)).astype(np.float32)
    for i in range(len(xyz)):
        if abs(xyz[i][0] - (-8)) < 1e-4:
            uvw[i][0] = 1.0
    return uvw

# define start time and time step
start_time = 100
time_step = 1

cc = (0.0, 0.0)
cr = 0.5
geo = psci.geometry.CylinderInCube(
    origin=(-8, -8, -2),
    extent=(25, 8, 2),
    circle_center=cc,
    circle_radius=cr)

geo.add_boundary(name="left", criteria=lambda x, y, z: abs(x + 8.0) < 1e-4)
geo.add_boundary(name="right", criteria=lambda x, y, z: abs(x - 25.0) < 1e-4)
geo.add_boundary(
    name="circle",
    criteria=lambda x, y, z: ((x - cc[0])**2 + (y - cc[1])**2 - cr**2) < 1e-4)

# discretize geometry
geo_disc = geo.discretize(npoints=80000, method="sampling")

# the real_cord need to be added in geo_disc
real_cord = GetRealPhyInfo(start_time, need_cord=True)
geo_disc.user = real_cord

# N-S equation
pde = psci.pde.NavierStokes(
    nu=0.005,
    rho=1.0,
    dim=3,
    time_dependent=True,
    weight=[0.01, 0.01, 0.01, 0.01])

pde.set_time_interval([0.0, 105.0])

# boundary condition on left side: u=10, v=w=0
bc_left_u = psci.bc.Dirichlet('u', rhs=1.0, weight=1.0)
bc_left_v = psci.bc.Dirichlet('v', rhs=0.0, weight=1.0)
bc_left_w = psci.bc.Dirichlet('w', rhs=0.0, weight=1.0)

# boundary condition on right side: p=0
bc_right_p = psci.bc.Dirichlet('p', rhs=0.0, weight=1.0)

# boundary on circle
bc_circle_u = psci.bc.Dirichlet('u', rhs=0.0, weight=1.0)
bc_circle_v = psci.bc.Dirichlet('v', rhs=0.0, weight=1.0)
bc_circle_w = psci.bc.Dirichlet('w', rhs=0.0, weight=1.0)

# add bounday and boundary condition
pde.add_bc("left", bc_left_u, bc_left_v, bc_left_w)
pde.add_bc("right", bc_right_p)
pde.add_bc("circle", bc_circle_u, bc_circle_v, bc_circle_w)

# pde discretization 
# time_step = start_time - 0
pde_disc = pde.discretize(
    time_method="implicit", time_step=start_time, geo_disc=geo_disc)

# Network
net = psci.network.FCNet(
    num_ins=3, num_outs=4, num_layers=10, hidden_size=50, activation='tanh')

# Loss
loss = psci.loss.L2(p=2)

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Solver parameter
solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)
solver.feed_data_interior_cur(GenInitPhyInfo(pde_disc.geometry.interior))  # add u(0) interior
solver.feed_data_user_cur(GenInitPhyInfo(real_cord))  # add u(0) user 
solver.feed_data_user_next(GetRealPhyInfo(start_time, need_physic=True))  # add u(start_time) user

# num_epoch in train
train_epoch = 20

# Solver time: 0 -> start_time
start_time_result = solver.solve(num_epoch = train_epoch)
file_path = "train_flow_unsteady_re200/fac3d_train_rslt_" + str(start_time)
psci.visu.save_vtk(filename=file_path, geo_disc=pde_disc.geometry, data=start_time_result)
start_time_interior = np.array(start_time_result[0])
start_time_usr = np.array(start_time_result[-1])

# time step need to be modified
pde_disc = pde.discretize(
    time_method="implicit", time_step=time_step, geo_disc=geo_disc)

solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)

# Solver time: [start_time, start_time+time_step, start_time+2*time_step, ...]
num_time_step = 10
current_interior = start_time_interior[:, 0:3]
current_user = start_time_usr[:, 0:3]
#current_user = GetRealPhyInfo(start_time, need_physic=True)[:, 0:3]
for i in range(num_time_step):
    next_time = start_time + (i + 1) * time_step
    solver.feed_data_interior_cur(current_interior)  # add u(n) interior
    solver.feed_data_user_cur(current_user)  # add u(n) user 
    solver.feed_data_user_next(GetRealPhyInfo(next_time, need_physic=True))  # add u(n+1) user
    next_uvwp = solver.solve(num_epoch = train_epoch)
    # Save vtk
    file_path = "train_flow_unsteady_re200/fac3d_train_rslt_" + str(next_time)
    psci.visu.save_vtk(filename=file_path, geo_disc=pde_disc.geometry, data=next_uvwp)
    next_interior = np.array(next_uvwp[0])
    next_user = np.array(next_uvwp[-1])
    current_interior = next_interior[:, 0:3]
    current_user = next_user[:, 0:3]
    #current_user = GetRealPhyInfo(next_time, need_physic=True)[:, 0:3]
