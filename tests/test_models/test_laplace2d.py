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


def laplace2d(static=True):
    paddle.seed(1)
    np.random.seed(1)

    if not static:
        paddle.disable_static()
    else:
        paddle.enable_static()

    # analytical solution
    ref_sol = lambda x, y: np.cos(x) * np.cosh(y)

    # set geometry and boundary
    geo = psci.geometry.Rectangular(origin=(0.0, 0.0), extent=(1.0, 1.0))
    geo.add_boundary(
        name="around",
        criteria=lambda x, y: (y == 1.0) | (y == 0.0) | (x == 0.0) | (x == 1.0))

    # discretize geometry
    npoints = 10
    geo_disc = geo.discretize(npoints=npoints, method="uniform")

    # Laplace
    pde = psci.pde.Laplace(dim=2)

    # set bounday condition
    bc_around = psci.bc.Dirichlet('u', rhs=ref_sol)

    # add bounday and boundary condition
    pde.add_bc("around", bc_around)

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


standard_value = np.load("./standard/laplace2d.npz", allow_pickle=True)


@pytest.mark.laplace2d
@pytest.mark.skipif(
    paddle.distributed.get_world_size() != 1, reason="skip serial case")
def test_laplace2d_0():
    """
    test laplace2d
    """
    dyn_solution, dyn_mse = standard_value['dyn_solution'].tolist(
    ), standard_value['dyn_mse']
    stc_solution, stc_mse = standard_value['stc_solution'].tolist(
    ), standard_value['stc_mse']
    dynamic_rslt, dynamic_mse = laplace2d(static=False)
    static_rslt, static_mse = laplace2d()
    # compare(dynamic_rslt, static_rslt)
    # compare(dynamic_mse, static_mse)

    compare(dyn_solution, dynamic_rslt, mode="equal")
    compare(dyn_mse, dynamic_mse, mode="equal")
    compare(stc_solution, static_rslt, mode="equal")
    compare(stc_mse, static_mse, mode="equal")


@pytest.mark.laplace2d
@pytest.mark.skipif(
    paddle.distributed.get_world_size() != 2, reason="skip distributed case")
def test_laplace2d_1():
    """
    test laplace2d
    """
    solution, mse = standard_value['dst_solution'].tolist(), standard_value[
        'dst_mse']
    static_rslt, static_mse = laplace2d()
    compare(solution, static_rslt, mode="equal")
    compare(mse, static_mse, mode="equal")


if __name__ == '__main__':
    code = pytest.main([sys.argv[0]])
    sys.exit(code)
