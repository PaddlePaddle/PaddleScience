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
import pytest


def jud_pinns(pde_disc):
    pde_inputs = [pde_disc.geometry.interior]
    for k, v in pde_disc.geometry.boundary.items():
        pde_inputs.append(v)

    net = psci.network.FCNet(2, 1, 2, 1)
    loss = psci.loss.L2()
    algo = psci.algorithm.PINNs(net=net, loss=loss)
    inputs, inputs_attr = algo.create_inputs_from_pde(pde_disc)
    assert inputs == pde_inputs
    assert len(inputs_attr) == 4


@pytest.mark.algorithm_PINNs
def test_PINNs0():
    """
    pde: laplace
    """
    geo = psci.geometry.Rectangular(origin=(0.0, 0.0), extent=(1.0, 1.0))
    geo.add_boundary(name="top", criteria=lambda x, y: (y == 1.0))
    geo_disc = geo.discretize(method="uniform", npoints=10)
    pde = psci.pde.Laplace(dim=2, weight=1.0)
    bc1 = psci.bc.Dirichlet('u', rhs=0)
    bc2 = psci.bc.Dirichlet('v', rhs=0)
    pde.add_bc("top", bc1, bc2)
    pde_disc = pde.discretize(geo_disc=geo_disc)
    jud_pinns(pde_disc)


@pytest.mark.algorithm_PINNs
def test_PINNs1():
    """
    pde: poisson
    """
    geo = psci.geometry.Rectangular(origin=(0.0, 0.0), extent=(1.0, 1.0))
    geo.add_boundary(name="top", criteria=lambda x, y: (y == 1.0))
    geo.add_boundary(name="down", criteria=lambda x, y: (y == 0.0))
    geo_disc = geo.discretize(method="uniform", npoints=40)
    pde = psci.pde.Poisson(dim=2, weight=1.0)
    bc1 = psci.bc.Dirichlet('u', rhs=0)
    bc2 = psci.bc.Dirichlet('v', rhs=0)
    bc3 = psci.bc.Dirichlet('u', rhs=lambda x, y: x + y)
    pde.add_bc("top", bc1, bc2)
    pde.add_bc("down", bc3)
    pde_disc = pde.discretize(geo_disc=geo_disc)
    jud_pinns(pde_disc)


@pytest.mark.algorithm_PINNs
def test_PINNs2():
    """
    pde: NavierStokes
    """
    geo = psci.geometry.Rectangular(
        origin=(0.0, 0.0, 0.0), extent=(1.0, 1.0, 0.1))
    geo.add_boundary(name="top", criteria=lambda x, y, z: (z == 0.1))
    geo.add_boundary(name="down", criteria=lambda x, y, z: (z == 0.0))
    geo_disc = geo.discretize(method="uniform", npoints=10000)
    pde = psci.pde.NavierStokes(dim=3, weight=1.0)
    bc1 = psci.bc.Dirichlet('u', rhs=0)
    bc2 = psci.bc.Dirichlet('v', rhs=0)
    bc3 = psci.bc.Dirichlet('w', rhs=lambda x, y, z: x + y + np.sin(z))
    pde.add_bc("top", bc1, bc2)
    pde.add_bc("down", bc3)
    pde_disc = pde.discretize(geo_disc=geo_disc)
    jud_pinns(pde_disc)
