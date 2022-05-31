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
import os
import wget
import zipfile

paddle.seed(1)
np.random.seed(1)

#paddle.enable_static()
paddle.disable_static()

# define start time
start_time = 100

cc = (0.0, 0.0)
cr = 0.5
geo = psci.geometry.CylinderInCube(
    origin=(-8, -8, -2), extent=(25, 8, 2), circle_center=cc, circle_radius=cr)

# discretize geometry
geo_disc = geo.discretize(npoints=[200, 50, 4], method="uniform")

# N-S equation
pde = psci.pde.NavierStokes(
    nu=0.01,
    rho=1.0,
    dim=3,
    time_dependent=True,
    weight=[0.01, 0.01, 0.01, 0.01])

pde.set_time_interval([100.0, 110.0])

# pde discretization 
# TODO(need to be moved in prediction)
pde_disc = pde.discretize(
    time_method="implicit", time_step=1, geo_disc=geo_disc)

# Network
net = psci.network.FCNet(
    num_ins=3, num_outs=4, num_layers=10, hidden_size=50, activation='tanh')

# Loss
# TODO(need to be moved in prediction)
loss = psci.loss.L2(p=2, data_weight=100.0)

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer 
# TODO(need to be moved in prediction)
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Solver parameter
solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)

# dynamic graph
if paddle.in_dynamic_mode():
    next_uvwp = solver.predict(
        dynamic_net_file='checkpoint/dynamic_net_params_1000.pdparams',
        dynamic_opt_file='checkpoint/dynamic_opt_params_1000.pdopt')
else:
    next_uvwp = solver.predict(
        static_model_file='checkpoint/static_model_params_1000.pdparams')

# save vtk
if paddle.in_dynamic_mode():
    file_path = "predict_cylinder_unsteady_re100/rslt_dynamic_" + str(100)
else:
    file_path = "predict_cylinder_unsteady_re100/rslt_static_" + str(100)
psci.visu.save_vtk(
    filename=file_path, geo_disc=pde_disc.geometry, data=next_uvwp)

# save npy
result = next_uvwp[-1]
result = np.array(result)
np.save("predict_cylinder_unsteady_re100/predict_result.npy", result)
