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

import paddlescience as psci
import numpy as np
import paddle

paddle.seed(1)
np.random.seed(1)

paddle.enable_static()

#paddle.disable_static()


def GenInitPhyInfo(xyz):
    uvwp = np.zeros((len(xyz), 3)).astype(np.float32)
    for i in range(len(xyz)):
        if abs(xyz[i][0] - (-8)) < 1e-4:
            uvwp[i][0] = 1.0
    return uvwp


def GetRealPhyInfo(time):
    use_real_data = False
    if use_real_data is True:
        xyzuvwp = np.load("csv/flow_re20_" + str(time) + "_xyzuvwp.npy")
    else:
        xyzuvwp = np.ones((1000, 7)).astype(np.float32)
    return xyzuvwp


circle_center = (0.0, 0.0)
circle_radius = 0.5
geo = psci.geometry.CylinderInCube(
    origin=(-8, -8, -0.5),
    extent=(25, 8, 0.5),
    circle_center=circle_center,
    circle_radius=circle_radius)

geo.add_boundary(name="top", criteria=lambda x, y, z: z == 0.5)
geo.add_boundary(name="down", criteria=lambda x, y, z: z == -0.5)
geo.add_boundary(name="left", criteria=lambda x, y, z: x == -8)
geo.add_boundary(name="right", criteria=lambda x, y, z: x == 25)
geo.add_boundary(name="front", criteria=lambda x, y, z: y == -8)
geo.add_boundary(name="back", criteria=lambda x, y, z: y == 8)
geo.add_boundary(
    name="circle",
    criteria=lambda x, y, z: (x - circle_center[0])**2 + (y - circle_center[1])**2 == circle_radius**2
)

# N-S
pde = psci.pde.NavierStokes(nu=0.05, rho=1.0, dim=3, time_dependent=True)

# set bounday condition
bc_top_u = psci.bc.Dirichlet('u', rhs=1.0)
bc_top_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_top_w = psci.bc.Dirichlet('w', rhs=0.0)

bc_down_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_down_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_down_w = psci.bc.Dirichlet('w', rhs=0.0)

bc_left_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_left_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_left_w = psci.bc.Dirichlet('w', rhs=0.0)

bc_right_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_right_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_right_w = psci.bc.Dirichlet('w', rhs=0.0)

bc_front_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_front_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_front_w = psci.bc.Dirichlet('w', rhs=0.0)

bc_back_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_back_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_back_w = psci.bc.Dirichlet('w', rhs=0.0)

# TODO 3. circle boundry
bc_circle_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_circle_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_circle_w = psci.bc.Dirichlet('w', rhs=0.0)

pde.add_geometry(geo)

# add bounday and boundary condition
pde.add_bc("top", bc_top_u, bc_top_v, bc_top_w)
pde.add_bc("down", bc_down_u, bc_down_v, bc_down_w)
pde.add_bc("left", bc_left_u, bc_left_v, bc_left_w)
pde.add_bc("right", bc_right_u, bc_right_v, bc_right_w)
pde.add_bc("front", bc_front_u, bc_front_v, bc_front_w)
pde.add_bc("back", bc_back_u, bc_back_v, bc_back_w)
pde.add_bc("circle", bc_circle_u, bc_circle_v, bc_circle_w)

# Discretization
pde_disc = psci.discretize(
    pde,
    time_method="implicit",
    time_step=0.5,
    space_npoints=60000,
    space_method="sampling")

# Get real data
real_xyzuvwp = GetRealPhyInfo(0.5)
real_xyz = real_xyzuvwp[:, 0:3]
real_uvwp = real_xyzuvwp[:, 3:7]

# load real physic data in geo
pde_disc.geometry.data = real_xyz

# Network
# TODO: remove num_ins and num_outs
net = psci.network.FCNet(
    num_ins=3, num_outs=4, num_layers=10, hidden_size=50, activation='tanh')

# Loss, TO rename
# bc_weight = GenBCWeight(geo.space_domain, geo.bc_index)
loss = psci.loss.L2()

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Add (u0,v0,w0) message for implicit method
uvw = GenInitPhyInfo(pde_disc.geometry.interior)
n = pde_disc.geometry.interior.shape[0]

# Solver train t0 -> t1
solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)
solver.feed_data_n(uvw)  # add u(n)
solver.feed_data(real_uvwp)  # add real data 
print("###################### start time=0.5 train task ############")
uvw_t1 = solver.solve(num_epoch=2)
uvw_t1 = uvw_t1[0]
uvw_t1 = np.array(uvw_t1)
print(uvw_t1.shape)

# Solver train t1 -> tn
time_step = 9
current_uvw = uvw_t1[:, 0:3]
for i in range(time_step):
    current_time = 0.5 + (i + 1) * 0.5
    print("###################### start time=%f train task ############" %
          current_time)
    solver.feed_data_n(current_uvw)  # add u(n)
    real_xyzuvwp = GetRealPhyInfo(current_time)
    real_uvwp = real_xyzuvwp[:, 3:7]
    solver.feed_data(real_uvwp)  # add real data 
    next_uvwp = solver.solve(num_epoch=2)
    # Save vtk
    psci.visu.save_vtk(geo_disc=pde_disc.geometry, data=next_uvwp)
    next_uvwp = next_uvwp[0]
    next_uvwp = np.array(next_uvwp)
    current_uvw = next_uvwp[:, 0:3]
