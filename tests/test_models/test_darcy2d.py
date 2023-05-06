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


def darcy2d(static=True):

    paddle.seed(1)
    np.random.seed(1)

    if not static:
        paddle.disable_static()
    else:
        paddle.enable_static()

    psci.config.set_dtype("float32")

    # ref solution
    ref_sol = lambda x, y: np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)

    # ref rhs
    ref_rhs = lambda x, y: 8.0 * np.pi**2 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)

    # set geometry and boundary
    geo = psci.geometry.Rectangular(origin=(0.0, 0.0), extent=(1.0, 1.0))

    geo.add_boundary(name="top", criteria=lambda x, y: y == 1.0)
    geo.add_boundary(name="down", criteria=lambda x, y: y == 0.0)
    geo.add_boundary(name="left", criteria=lambda x, y: x == 0.0)
    geo.add_boundary(name="right", criteria=lambda x, y: x == 1.0)

    # discretize geometry
    npoints = 10
    geo_disc = geo.discretize(npoints=npoints, method="uniform")

    # Poisson
    pde = psci.pde.Poisson(dim=2, rhs=ref_rhs)

    # set bounday condition
    bc_top = psci.bc.Dirichlet('u', rhs=ref_sol)
    bc_down = psci.bc.Dirichlet('u', rhs=ref_sol)
    bc_left = psci.bc.Dirichlet('u', rhs=ref_sol)
    bc_right = psci.bc.Dirichlet('u', rhs=ref_sol)

    # add bounday and boundary condition
    pde.add_bc("top", bc_top)
    pde.add_bc("down", bc_down)
    pde.add_bc("left", bc_left)
    pde.add_bc("right", bc_right)

    # discretization pde
    pde_disc = pde.discretize(geo_disc=geo_disc)

    # Network
    # TODO: remove num_ins and num_outs
    net = psci.network.FCNet(
        num_ins=2, num_outs=1, num_layers=5, hidden_size=20, activation='tanh')

    # Loss
    loss = psci.loss.L2()

    # Algorithm
    algo = psci.algorithm.PINNs(net=net, loss=loss)

    # Optimizer
    opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

    # Solver
    solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)
    solution = solver.solve(num_epoch=25)

    # MSE
    # TODO: solution array to dict: interior, bc
    cord = pde_disc.geometry.interior
    ref = ref_sol(cord[:, 0], cord[:, 1])
    mse2 = np.linalg.norm(solution[0][:, 0] - ref, ord=2)**2

    n = 1
    for cord in pde_disc.geometry.boundary.values():
        ref = ref_sol(cord[:, 0], cord[:, 1])
        mse2 += np.linalg.norm(solution[n][:, 0] - ref, ord=2)**2
        n += 1

    mse = mse2 / npoints
    return solution, mse


standard_value = np.load("./standard/darcy2d.npz", allow_pickle=True)


@pytest.mark.darcy2d
@pytest.mark.skipif(
    paddle.distributed.get_world_size() != 1, reason="skip serial case")
def test_darcy2d_0():
    """
    test darcy2d
    """
    dyn_solution, dyn_mse = standard_value['dyn_solution'].tolist(
    ), standard_value['dyn_mse']
    stc_solution, stc_mse = standard_value['stc_solution'].tolist(
    ), standard_value['stc_mse']
    dynamic_rslt, dynamic_mse = darcy2d(static=False)
    static_rslt, static_mse = darcy2d()
    #compare(dynamic_rslt, static_rslt)
    #compare(dynamic_mse, static_mse)
    compare(dyn_solution, dynamic_rslt, delta=1e-7)
    compare(dyn_mse, dynamic_mse, mode="equal")
    compare(stc_solution, static_rslt, delta=1e-7)
    compare(stc_mse, static_mse, mode="equal")


@pytest.mark.darcy2d
@pytest.mark.skipif(
    paddle.distributed.get_world_size() != 2, reason="skip distributed case")
def test_darcy2d_1():
    """
    test darcy2d
    distributed case
    """
    standard_value = np.load("./standard/darcy2d.npz", allow_pickle=True)
    solution, mse = standard_value['dst_solution'].tolist(), standard_value[
        'dst_mse']
    static_rslt, static_mse = darcy2d()
    compare(solution, static_rslt, delta=1e-7)
    compare(mse, static_mse, delta=1e-7)


if __name__ == '__main__':
    code = pytest.main([sys.argv[0]])
    sys.exit(code)
