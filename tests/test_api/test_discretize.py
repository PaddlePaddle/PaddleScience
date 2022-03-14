"""
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""

import numpy as np
import paddlescience as psci
import pytest


def geo_discrete(space, steps):
    """
    calculate geometry discrete
    """
    origin = space[0]
    extent = space[1]
    length, width = extent[0] - origin[0], extent[1] - origin[1]
    domains = []
    bc, index = [], 0
    for i in range(steps[1]):
        for j in range(steps[0]):
            d_x = origin[0] + length / (steps[0] - 1) * j
            d_y = origin[1] + width / (steps[1] - 1) * i
            domains.append([d_x, d_y])
            if d_x in (origin[0], extent[0]) or d_y in (origin[1], extent[1]):
                bc.append(index)
            index += 1
    return domains, bc


@pytest.mark.api_discrete
def test_discrete0():
    """
    pde = laplace2d
    """
    geo = psci.geometry.Rectangular(
        space_origin=(0.0, 0.0), space_extent=(1.0, 1.0))
    pdes = psci.pde.Laplace2D()
    pdes, geo = psci.discretize(pdes, geo, space_nsteps=(2, 2))
    bc = geo.get_bc_index()
    sd = geo.get_space_domain()

    sd_t, bc_t = geo_discrete([(0.0, 0.0), (1.0, 1.0)], (2, 2))
    assert np.allclose(bc, bc_t)
    assert np.allclose(sd, sd_t)


@pytest.mark.api_discrete
def test_discrete1():
    """
    pde = NavierStokes
    """
    geo = psci.geometry.Rectangular(
        space_origin=(0.0, 0.0), space_extent=(1.0, 1.0))
    pdes = psci.pde.NavierStokes(0.01, 1.0)
    pdes, geo = psci.discretize(pdes, geo, space_nsteps=(2, 2))
    bc = geo.get_bc_index()
    sd = geo.get_space_domain()

    sd_t, bc_t = geo_discrete([(0.0, 0.0), (1.0, 1.0)], (2, 2))
    assert np.allclose(bc, bc_t)
    assert np.allclose(sd, sd_t)


@pytest.mark.api_discrete
def test_discrete2():
    """
    pde = NavierStokes
    space_nsteps=(20, 20)
    """
    geo = psci.geometry.Rectangular(
        space_origin=(0.0, 0.0), space_extent=(1.0, 1.0))
    pdes = psci.pde.NavierStokes(0.01, 1.0)
    pdes, geo = psci.discretize(pdes, geo, space_nsteps=(20, 20))
    bc = geo.get_bc_index()
    sd = geo.get_space_domain()

    sd_t, bc_t = geo_discrete([(0.0, 0.0), (1.0, 1.0)], (20, 20))
    assert np.allclose(bc, bc_t)
    assert np.allclose(sd, sd_t)


@pytest.mark.api_discrete
def test_discrete3():
    """
    pde = NavierStokes
    space_nsteps=(10, 20)
    """
    geo = psci.geometry.Rectangular(
        space_origin=(0.0, 0.0), space_extent=(1.0, 1.0))
    pdes = psci.pde.NavierStokes(0.01, 1.0)
    pdes, geo = psci.discretize(pdes, geo, space_nsteps=(10, 20))
    bc = geo.get_bc_index()
    sd = geo.get_space_domain()

    sd_t, bc_t = geo_discrete([(0.0, 0.0), (1.0, 1.0)], (10, 20))
    assert np.allclose(bc, bc_t)
    assert np.allclose(sd, sd_t)


@pytest.mark.api_discrete
def test_discrete4():
    """
    pde = NavierStokes
    space_nsteps=(10, 20)
    space_origin=(-1.0, -1.0)
    """
    geo = psci.geometry.Rectangular(
        space_origin=(-1.0, -1.0), space_extent=(1.0, 1.0))
    pdes = psci.pde.NavierStokes(0.01, 1.0)
    pdes, geo = psci.discretize(pdes, geo, space_nsteps=(10, 20))
    bc = geo.get_bc_index()
    sd = geo.get_space_domain()

    sd_t, bc_t = geo_discrete([(-1.0, -1.0), (1.0, 1.0)], (10, 20))
    assert np.allclose(bc, bc_t)
    assert np.allclose(sd, sd_t)


@pytest.mark.api_discrete
def test_discrete5():
    """
    pde = NavierStokes
    space_nsteps=(10, 20)
    space_origin=(-1.0, -1.0)
    space_extent=(2.0, 1.0)
    """
    geo = psci.geometry.Rectangular(
        space_origin=(0.0, -1.0), space_extent=(2.0, 1.0))
    pdes = psci.pde.NavierStokes(0.01, 1.0)
    pdes, geo = psci.discretize(pdes, geo, space_nsteps=(4, 3))
    bc = geo.get_bc_index()
    sd = geo.get_space_domain()

    sd_t, bc_t = geo_discrete([(0.0, -1.0), (2.0, 1.0)], (4, 3))
    assert np.allclose(bc, bc_t)
    assert np.allclose(sd, sd_t)
