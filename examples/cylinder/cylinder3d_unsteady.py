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
def GetRealPhyInfo(time):
    real_data = np.load("flow_unsteady_re200/flow_re200_" + str(time) + "_xyzuvwp.npy")
    real_data = real_data.astype(np.float32)
    print(real_data.shape[0])
    return real_data

def GenInitPhyInfo(xyz):
    uvw = np.zeros((len(xyz), 3)).astype(np.float32)
    for i in range(len(xyz)):
        if abs(xyz[i][0] - (-8)) < 1e-4:
            uvw[i][0] = 1.0
    return uvw

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
start_time = 100
time_step = 1
# Get real data
real_data = GetRealPhyInfo(start_time)
real_cord = real_data[:, 0:3]
real_sol = real_data[:, 3:7]
geo_disc = geo.discretize(npoints=40000, method="sampling")
geo_disc.user = real_cord

# N-S
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

# discretization
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

# Solver train t0 -> t1
solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)
ui_cur = GenInitPhyInfo(pde_disc.geometry.interior)
solver.feed_data_interior_cur(ui_cur)  # add u(n) interior
us_cur = GenInitPhyInfo(real_cord)
solver.feed_data_user_cur(us_cur)  # add u(n) user 
solver.feed_data_user_next(real_sol)  # add u(n+1) user

train_epoch = 2000
print("################ start time=%d train task ############"%start_time)
uvw_t1 = solver.solve(num_epoch = train_epoch)
file_path = "train_flow_unsteady_re200/fac3d_train_rslt_" + str(start_time)
psci.visu.save_vtk(filename=file_path, geo_disc=pde_disc.geometry, data=uvw_t1)
t1_interior = uvw_t1[0]
t1_interior = np.array(t1_interior)
t1_user = uvw_t1[-1]
t1_user = np.array(t1_user)

# discretization modify the time_step
pde_disc = pde.discretize(
    time_method="implicit", time_step=time_step, geo_disc=geo_disc)

solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)

# Solver train t1 -> tn
time_steps = 5
current_interior = t1_interior[:, 0:3]
current_user = t1_user[:, 0:3]
for i in range(time_steps):
    current_time = start_time + (i + 1) * time_step
    print("############# start time=%d train task ############" %current_time)
    solver.feed_data_interior_cur(current_interior)  # add u(n) interior
    solver.feed_data_user_cur(current_user)  # add u(n) user 
    real_xyzuvwp = GetRealPhyInfo(current_time)
    real_uvwp = real_xyzuvwp[:, 3:7]
    solver.feed_data_user_next(real_uvwp)  # add u(n+1) user
    next_uvwp = solver.solve(num_epoch = train_epoch)
    # Save vtk
    file_path = "train_flow_unsteady_re200/fac3d_train_rslt_" + str(current_time)
    psci.visu.save_vtk(filename=file_path, geo_disc=pde_disc.geometry, data=next_uvwp)
    tn_interior = next_uvwp[0]
    tn_interior = np.array(tn_interior)
    tn_user = next_uvwp[-1]
    tn_user = np.array(tn_user)
    current_interior = tn_interior[:, 0:3]
    current_user = tn_user[:, 0:3]
