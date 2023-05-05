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
import pytest
from tool import compare
import sys


def kovasznay(static=True):
    paddle.seed(1)
    np.random.seed(1)

    if not static:
        paddle.disable_static()
    else:
        paddle.enable_static()

    # constants
    Re = 40.0
    r = Re / 2 - np.sqrt(Re**2 / 4.0 + 4.0 * np.pi**2)

    # Kovasznay solution
    ref_sol_u = lambda x, y: 1.0 - np.exp(r * x) * np.cos(2.0 * np.pi * y)
    ref_sol_v = lambda x, y: r / (2 * np.pi) * np.exp(r * x) * np.sin(2.0 * np.pi * y)
    ref_sol_p = lambda x, y: 1.0 / 2.0 - 1.0 / 2.0 * np.exp(2.0 * r * x)

    # set geometry and boundary
    geo = psci.geometry.Rectangular(origin=(-0.5, -0.5), extent=(1.5, 1.5))
    geo.add_boundary(
        name="boarder",
        criteria=lambda x, y: (x == -0.5) | (x == 1.5) | (y == -0.5) | (y == 1.5)
    )

    # discretization
    npoints = 10
    geo_disc = geo.discretize(npoints=npoints, method="uniform")

    # N-S equation
    pde = psci.pde.NavierStokes(nu=1.0 / Re, rho=1.0, dim=2)

    # set boundary condition
    bc_border_u = psci.bc.Dirichlet('u', ref_sol_u)
    bc_border_v = psci.bc.Dirichlet('v', ref_sol_v)
    bc_border_p = psci.bc.Dirichlet('p', ref_sol_p)

    # add bounday and boundary condition
    pde.add_bc("boarder", bc_border_u)
    pde.add_bc("boarder", bc_border_v)
    pde.add_bc("boarder", bc_border_p)

    # discretization pde
    pde_disc = pde.discretize(geo_disc=geo_disc)

    # Network
    net = psci.network.FCNet(
        num_ins=2,
        num_outs=3,
        num_layers=10,
        hidden_size=50,
        activation='tanh')

    # Loss
    loss = psci.loss.L2()

    # Algorithm
    algo = psci.algorithm.PINNs(net=net, loss=loss)

    # Optimizer
    opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

    # Solver
    solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)
    solution = solver.solve(num_epoch=25)

    return solution


standard_value = np.load("./standard/kovasznay.npz", allow_pickle=True)


@pytest.mark.kovasznay
@pytest.mark.skipif(
    paddle.distributed.get_world_size() != 1, reason="skip serial case")
def test_kovasznay_0():
    """
    test kovasznay
    """
    dyn_solution = standard_value['dyn_solution'].tolist()
    stc_solution = standard_value['stc_solution'].tolist()
    dynamic_rslt = kovasznay(static=False)
    static_rslt = kovasznay()
    # compare(dynamic_rslt, static_rslt)

    compare(dyn_solution, dynamic_rslt, delta=1e-7)
    compare(stc_solution, static_rslt, delta=1e-7)


@pytest.mark.kovasznay
@pytest.mark.skipif(
    paddle.distributed.get_world_size() != 2, reason="skip distributed case")
def test_kovasznay_1():
    """
    test kovasznay
    distributed case: padding
    """
    solution = standard_value['dst_solution'].tolist()
    static_rslt = kovasznay()
    compare(solution, static_rslt, delta=1e-7)


if __name__ == '__main__':
    code = pytest.main([sys.argv[0]])
    sys.exit(code)
