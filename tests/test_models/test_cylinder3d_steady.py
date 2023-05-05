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
import sys
from tool import compare


def cylinder3d_steady(static=True):
    paddle.seed(1)
    np.random.seed(1)

    if not static:
        paddle.disable_static()
    else:
        paddle.enable_static()

    # load real data
    real_data = np.load(
        "../../examples/cylinder/3d_steady/re20_5.0.npy").astype("float32")
    real_cord = real_data[:, 0:3]
    real_sol = real_data[:, 3:7]

    cc = (0.0, 0.0)
    cr = 0.5
    geo = psci.geometry.CylinderInCube(
        origin=(-8, -8, -0.5),
        extent=(25, 8, 0.5),
        circle_center=cc,
        circle_radius=cr)

    geo.add_boundary(name="left", criteria=lambda x, y, z: abs(x + 8.0) < 1e-4)
    geo.add_boundary(
        name="right", criteria=lambda x, y, z: abs(x - 25.0) < 1e-4)
    geo.add_boundary(
        name="circle",
        criteria=lambda x, y, z: ((x - cc[0])**2 + (y - cc[1])**2 - cr**2) < 1e-4
    )

    # discretize geometry
    geo_disc = geo.discretize(npoints=600, method="sampling")
    geo_disc.user = real_cord

    # N-S
    pde = psci.pde.NavierStokes(
        nu=0.05,
        rho=1.0,
        dim=3,
        time_dependent=False,
        weight=[4.0, 0.01, 0.01, 0.01])

    # boundary condition on left side: u=1, v=w=0
    bc_left_u = psci.bc.Dirichlet('u', rhs=1.0)
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

    # discritize pde
    pde_disc = pde.discretize(geo_disc=geo_disc)

    # network
    net = psci.network.FCNet(
        num_ins=3,
        num_outs=4,
        num_layers=10,
        hidden_size=50,
        activation='tanh')

    # loss
    loss = psci.loss.L2(p=2)

    # Algorithm
    algo = psci.algorithm.PINNs(net=net, loss=loss)

    # Optimizer
    opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

    # Solver
    solver = psci.solver.Solver(pde=pde, algo=algo, opt=opt)

    solver.feed_data_user(real_sol)  # add real solution

    res_shpae = [(2044, 4), (58, 4), (58, 4), (4, 4), (10000, 4)]
    solution = solver.solve(num_epoch=10)
    res = [np.sum(item, axis=0) for item in solution]
    return sum(res)


standard = np.load("./standard/cylinder3d_steady.npz", allow_pickle=True)


@pytest.mark.cylinder3d_steady
@pytest.mark.skipif(
    paddle.distributed.get_world_size() != 1, reason="skip serial case")
def test_cylinder3d_steady_0():
    """
    test cylinder3d_steady
    """
    dyn_standard, stc_standard = standard['dyn_solution'], standard[
        'stc_solution']
    dyn_rslt = cylinder3d_steady(static=False)
    stc_rslt = cylinder3d_steady()
    #compare(dyn_rslt, stc_rslt)
    compare(dyn_standard, dyn_rslt, mode="equal")
    compare(stc_standard, stc_rslt, mode="equal")


@pytest.mark.cylinder3d_steady
@pytest.mark.skipif(
    paddle.distributed.get_world_size() != 2, reason="skip distributed case")
def test_cylinder3d_steady_1():
    """
    test cylinder3d_steady
    distributed case: padding
    """
    dst_standard = standard["dst_solution"]
    dst_rslt = cylinder3d_steady()
    compare(dst_standard, dst_rslt, mode="equal")


if __name__ == '__main__':
    code = pytest.main([sys.argv[0]])
    sys.exit(code)
