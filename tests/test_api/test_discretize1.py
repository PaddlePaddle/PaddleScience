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

import numpy as np
import paddlescience as psci
import pytest


def cal_discrete(space, time, time_steps, space_steps):
    """
    calculate geometry discrete
    with time dependent
    """
    dims = len(space[0])
    steps = []
    for i in range(dims):
        steps.append(
            np.linspace(
                space[0][i], space[1][i], space_steps[i], endpoint=True))

    space_domain = None
    bc_index = None
    if dims == 2:
        mesh = np.meshgrid(steps[1], steps[0], sparse=False, indexing='ij')
        space_domain = np.stack(
            (mesh[1].reshape(-1), mesh[0].reshape(-1)), axis=-1)
        nx, ny = space_steps
        idx = 0
        bc_index = np.zeros((nx * ny - (nx - 2) * (ny - 2)))
        for j in range(ny):
            for i in range(nx):
                if j == 0 or j == ny - 1 or i == 0 or i == nx - 1:
                    bc_index[idx] = j * nx + i
                    idx += 1
    elif dims == 3:
        mesh = np.meshgrid(
            steps[2], steps[1], steps[0], sparse=False, indexing='ij')
        space_domain = np.stack(
            (mesh[2].reshape(-1), mesh[1].reshape(-1), mesh[0].reshape(-1)),
            axis=-1)
        nx, ny, nz = space_steps
        bc_index = np.zeros((nx * ny * nz - (nx - 2) * (ny - 2) * (ny - 2)))
        idx = 0
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    if (k == 0 or k == nz - 1 or j == 0 or j == ny - 1 or
                            i == 0 or i == nx - 1):
                        bc_index[idx] = k * nx * ny + j * nx + i
                        idx += 1

    bc_offset = np.arange(time_steps).repeat(len(bc_index))
    bc_offset = bc_offset * len(space_domain)
    bc_index = np.tile(bc_index, time_steps)
    bc_index = bc_index + bc_offset
    ic_index = np.arange(len(space_domain))

    domain = []
    time_domain = np.linspace(time[0], time[1], time_steps, endpoint=True)
    for time in time_domain:
        current_time = time * np.ones((len(space_domain), 1), dtype=np.float32)
        current_domain = np.concatenate((current_time, space_domain), axis=-1)
        domain.append(current_domain.tolist())
    domain = np.array(domain).reshape(time_steps * space_domain.shape[0],
                                      space_domain.shape[-1] + 1)
    domain_size = len(domain)
    return space_domain, time_domain, bc_index, ic_index, domain, domain_size


@pytest.mark.api_discrete
def test_discrete0():
    """
    pde = NavierStokes
    geo: 2d
    time: 0~0.5
    """

    geo = psci.geometry.Rectangular(
        time_dependent=True,
        time_origin=0,
        time_extent=0.5,
        space_origin=(0.0, 0.0),
        space_extent=(1.0, 1.0))
    pdes = psci.pde.NavierStokes(0.01, 1.0, time_dependent=True)
    pdes, geo = psci.discretize(pdes, geo, time_nsteps=6, space_nsteps=(2, 2))
    bc = geo.get_bc_index()
    ic = geo.get_ic_index()
    sd = geo.get_space_domain()
    ds = geo.get_domain_size()
    td = geo.get_time_domain()
    domain = geo.get_domain()

    sd_t, td_t, bc_t, ic_t, domain_t, ds_t = cal_discrete(
        [(0.0, 0.0), (1.0, 1.0)], [0, 0.5], 6, (2, 2))
    assert np.allclose(bc, bc_t)
    assert np.allclose(sd, sd_t)
    assert np.allclose(ic, ic_t)
    assert np.allclose(td, td_t)
    assert np.allclose(ds, ds_t)
    assert np.allclose(domain, domain_t)


@pytest.mark.api_discrete
def test_discrete1():
    """
    pde = NavierStokes
    geo: 2d
    time: 0~0.5
    space_nsteps=(4, 5)
    space_origin=(-1.0, -1.0)
    space_extent=(2.0, 1.0)
    """

    geo = psci.geometry.Rectangular(
        time_dependent=True,
        time_origin=0,
        time_extent=0.5,
        space_origin=(-1.0, -1.0),
        space_extent=(2.0, 1.0))
    pdes = psci.pde.NavierStokes(0.01, 1.0, time_dependent=True)
    pdes, geo = psci.discretize(pdes, geo, time_nsteps=6, space_nsteps=(4, 5))
    bc = geo.get_bc_index()
    ic = geo.get_ic_index()
    sd = geo.get_space_domain()
    ds = geo.get_domain_size()
    td = geo.get_time_domain()
    domain = geo.get_domain()

    sd_t, td_t, bc_t, ic_t, domain_t, ds_t = cal_discrete(
        [(-1.0, -1.0), (2.0, 1.0)], [0, 0.5], 6, (4, 5))
    assert np.allclose(bc, bc_t)
    assert np.allclose(sd, sd_t)
    assert np.allclose(ic, ic_t)
    assert np.allclose(td, td_t)
    assert np.allclose(ds, ds_t)
    assert np.allclose(domain, domain_t)


@pytest.mark.api_discrete
def test_discrete2():
    """
    pde = NavierStokes3d
    geo: 3d
    time: 0~0.5
    """

    geo = psci.geometry.Rectangular(
        time_dependent=True,
        time_origin=0,
        time_extent=0.5,
        space_origin=(0.0, 0.0, 0.0),
        space_extent=(1.0, 1.0, 1.0))
    pdes = psci.pde.NavierStokes(0.01, 1.0, time_dependent=True)
    pdes, geo = psci.discretize(
        pdes, geo, time_nsteps=6, space_nsteps=(3, 3, 3))
    bc = geo.get_bc_index()
    ic = geo.get_ic_index()
    sd = geo.get_space_domain()
    ds = geo.get_domain_size()
    td = geo.get_time_domain()
    domain = geo.get_domain()

    sd_t, td_t, bc_t, ic_t, domain_t, ds_t = cal_discrete(
        [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)], [0, 0.5], 6, (3, 3, 3))
    assert np.allclose(bc, bc_t)
    assert np.allclose(sd, sd_t)
    assert np.allclose(ic, ic_t)
    assert np.allclose(td, td_t)
    assert np.allclose(ds, ds_t)
    assert np.allclose(domain, domain_t)


@pytest.mark.api_discrete
def test_discrete3():
    """
    pde = NavierStokes3d
    geo: 3d
    time: 0~1
    time_steps=11
    """

    geo = psci.geometry.Rectangular(
        time_dependent=True,
        time_origin=0,
        time_extent=0.5,
        space_origin=(0.0, 0.0, 0.0),
        space_extent=(1.0, 1.0, 1.0))
    pdes = psci.pde.NavierStokes(0.01, 1.0, time_dependent=True)
    pdes, geo = psci.discretize(
        pdes, geo, time_nsteps=11, space_nsteps=(3, 3, 3))
    bc = geo.get_bc_index()
    ic = geo.get_ic_index()
    sd = geo.get_space_domain()
    ds = geo.get_domain_size()
    td = geo.get_time_domain()
    domain = geo.get_domain()

    sd_t, td_t, bc_t, ic_t, domain_t, ds_t = cal_discrete(
        [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)], [0, 0.5], 11, (3, 3, 3))
    assert np.allclose(bc, bc_t)
    assert np.allclose(sd, sd_t)
    assert np.allclose(ic, ic_t)
    assert np.allclose(td, td_t)
    assert np.allclose(ds, ds_t)
    assert np.allclose(domain, domain_t)
