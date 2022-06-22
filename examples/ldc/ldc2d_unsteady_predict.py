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

# set geometry and boundary
geo = psci.geometry.Rectangular(origin=(-0.10, -0.10), extent=(0.10, 0.10))

# discretize geometry
npoints = 10201
geo_disc = geo.discretize(npoints=npoints, method="uniform")

# N-S
pde = psci.pde.NavierStokes(nu=0.01, rho=1.0, dim=2, time_dependent=True)
pde.set_time_interval([0.0, 0.5])

# discretization pde
pde_disc = pde.discretize(time_step=0.1, geo_disc=geo_disc)

# Network
net = psci.network.FCNet(
    num_ins=3, num_outs=3, num_layers=10, hidden_size=50, activation='tanh')

# Algorithm
algo = psci.algorithm.PINNs(net=net)

# Solver
solver = psci.solver.Solver(pde=pde_disc, algo=algo)
solution = solver.predict(
    dynamic_net_file='checkpoint/dynamic_net_params_10000.pdparams')

psci.visu.save_vtk(
    time_array=pde_disc.time_array, geo_disc=pde_disc.geometry, data=solution)
