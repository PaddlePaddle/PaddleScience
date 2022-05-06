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

# paddle.enable_static()
paddle.disable_static()


# load real data
def GetRealPhyInfo(time):
    use_real_data = True
    if use_real_data is True:
        real_data = np.load("flow_unsteady/flow_re200_" + format(time, '.2f') +
                            "_xyzuvwp.npy")
        real_data = real_data.astype(np.float32)
    else:
        real_data = np.ones((1000, 7)).astype(np.float32)
    return real_data


cc = (0.0, 0.0)
cr = 0.5
geo = psci.geometry.CylinderInCube(
    origin=(-8, -8, -0.5),
    extent=(25, 8, 0.5),
    circle_center=cc,
    circle_radius=cr)

geo.add_boundary(name="left", criteria=lambda x, y, z: abs(x + 8.0) < 1e-4)
geo.add_boundary(name="right", criteria=lambda x, y, z: abs(x - 25.0) < 1e-4)
geo.add_boundary(
    name="circle",
    criteria=lambda x, y, z: ((x - cc[0])**2 + (y - cc[1])**2 - cr**2) < 1e-4)

# discretize geometry
time_step = 0.25

# Get real data
real_data = GetRealPhyInfo(time_step)
real_cord = real_data[:, 0:3]
real_solu = real_data[:, 3:7]

geo_disc = geo.discretize(npoints=10000, method="sampling")
geo_disc.user = real_cord

geo_disc.interior = np.load("in.npy").astype("float32")
geo_disc.boundary["left"] = np.load("bl.npy").astype("float32")
geo_disc.boundary["right"] = np.load("br.npy").astype("float32")
geo_disc.boundary["circle"] = np.load("bc.npy").astype("float32")

# N-S
pde = psci.pde.NavierStokes(
    nu=0.05,
    rho=1.0,
    dim=3,
    time_dependent=True,
    weight=[4.0, 0.01, 0.01, 0.01])
pde.set_time_interval([0.0, 10.0])

# boundary condition on left side: u=10, v=w=0
bc_left_u = psci.bc.Dirichlet('u', rhs=10.0)
bc_left_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_left_w = psci.bc.Dirichlet('w', rhs=0.0)

# boundary condition on right side: p=0
bc_right_p = psci.bc.Dirichlet('p', rhs=0.0)

# boundary on circle
bc_circle_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_circle_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_circle_w = psci.bc.Dirichlet('w', rhs=0.0)

# add bounday and boundary condition
pde.add_bc("left", bc_left_u, bc_left_v, bc_left_w)
pde.add_bc("right", bc_right_p)
pde.add_bc("circle", bc_circle_u, bc_circle_v, bc_circle_w)

# discretization
pde_disc = pde.discretize(
    time_method="implicit", time_step=time_step, geo_disc=geo_disc)

# Network
net = psci.network.FCNet(
    num_ins=3, num_outs=4, num_layers=10, hidden_size=50, activation='tanh')

# Loss
loss = psci.loss.L2(p=2)

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Solver 
solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)

n = len(pde_disc.geometry.interior)
ui_cur = np.zeros((n, 3)).astype(np.float32)
solver.feed_data_interior_cur(ui_cur)  # add u(n) interior

# 
n = 10000
us_cur = np.zeros((n, 3)).astype(np.float32)
for i in range(n):
    x = real_cord[i, 0]
    if abs(x + 8.0) < 1e-4:
        us_cur[i, 0] = 10.0
us_next = real_solu
solver.feed_data_user_cur(us_cur)  # add u(n) user 
solver.feed_data_user_next(us_next)  # add u(n+1) user

print("###################### start time=0.25 train task ############")
uvw_t1 = solver.solve(num_epoch=2000)
psci.visu.save_vtk(geo_disc=pde_disc.geometry, data=uvw_t1)
