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
import sympy
import numpy as np
import pytest


def jud_ns(geo_disc,
           bc,
           time_dependent=True,
           time_interval=None,
           time_step=None,
           time_method=None):

    dim = geo_disc.interior.shape[1]

    pde = psci.pde.NavierStokes(dim=dim, time_dependent=time_dependent)
    for k in geo_disc.boundary.keys():
        pde.add_bc(k, *bc[k])
    if time_interval:
        pde.set_time_interval(time_interval)
    pde_disc = None
    if time_dependent:
        pde_disc = pde.discretize(
            time_method=time_method, time_step=time_step, geo_disc=geo_disc)
    else:
        pde_disc = pde.discretize(geo_disc=geo_disc)

    indvar, dvar = list(), list()

    if dim == 2:
        dvar_name = ['u', 'v', 'p']
        if time_dependent and time_method is None:
            indvar = sympy.symbols('t x y')
            for var in dvar_name:
                dvar.append(sympy.Function(var)(*indvar))
        else:
            indvar = sympy.symbols('x y')
            for var in dvar_name:
                dvar.append(sympy.Function(var)(*indvar))

    elif dim == 3:
        dvar_name = ['u', 'v', 'w', 'p']
        if time_dependent and time_method is None:
            indvar = sympy.symbols('t x y z')
            for var in dvar_name:
                dvar.append(sympy.Function(var)(*indvar))
        else:
            indvar = sympy.symbols('x y z')
            for var in dvar_name:
                dvar.append(sympy.Function(var)(*indvar))

    assert pde_disc.indvar == list(indvar)
    assert pde_disc.dvar == dvar
    for k in pde_disc.bc.keys():
        length = len(pde_disc.bc[k])
        for i in range(length):
            assert isinstance(pde_disc.bc[k][i], type(bc[k][i]))
    if time_interval:
        n = int((time_interval[1] - time_interval[0]) / time_step) + 1
        time_array = np.linspace(time_interval[0], time_interval[1], n)
        assert np.allclose(pde_disc.time_array, time_array)


@pytest.mark.pde_navierstokes
def test_navierstokes0():
    """
    2d
    time_dependent=False
    """
    geo = psci.geometry.Rectangular(origin=(0.0, 0.0), extent=(1.0, 1.0))
    geo.add_boundary(name="top", criteria=lambda x, y: (y == 1.0))
    geo_disc = geo.discretize(method="uniform", npoints=10)

    bc1 = psci.bc.Dirichlet('u', rhs=0)
    bc2 = psci.bc.Dirichlet('v', rhs=0)
    bc = {'top': [bc1, bc2]}
    jud_ns(geo_disc, bc, time_dependent=False)


@pytest.mark.pde_navierstokes
def test_navierstokes1():
    """
    2d
    time_dependent=True
    time_method=None
    time_interval = [0, 0.5]
    time_step=0.1
    """
    geo = psci.geometry.Rectangular(origin=(0.0, 0.0), extent=(1.0, 1.0))
    geo.add_boundary(name="top", criteria=lambda x, y: (y == 1.0))
    geo_disc = geo.discretize(method="uniform", npoints=10)

    bc1 = psci.bc.Dirichlet('u', rhs=0)
    bc2 = psci.bc.Dirichlet('v', rhs=0)
    bc = {'top': [bc1, bc2]}
    jud_ns(
        geo_disc,
        bc,
        time_dependent=True,
        time_interval=[0., 0.5],
        time_step=0.1)


@pytest.mark.pde_navierstokes
def test_navierstokes2():
    """
    2d
    time_dependent=True
    time_method=implicit
    time_interval = [0, 150]
    time_step=50
    """
    geo = psci.geometry.Rectangular(origin=(0.0, 0.0), extent=(1.0, 1.0))
    geo.add_boundary(name="top", criteria=lambda x, y: (y == 1.0))
    geo_disc = geo.discretize(method="uniform", npoints=10)

    bc1 = psci.bc.Dirichlet('u', rhs=0)
    bc2 = psci.bc.Dirichlet('v', rhs=0)
    bc = {'top': [bc1, bc2]}
    jud_ns(
        geo_disc,
        bc,
        time_dependent=True,
        time_interval=[0., 150.],
        time_step=50.0,
        time_method="implicit")


@pytest.mark.pde_navierstokes
def test_navierstokes3():
    """
    3d
    time_dependent=False
    """
    geo = psci.geometry.Rectangular(
        origin=(0.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0))
    geo.add_boundary(name="top", criteria=lambda x, y, z: (z == 1.0))
    geo_disc = geo.discretize(method="uniform", npoints=40)
    bc1 = psci.bc.Dirichlet('u', rhs=0)
    bc2 = psci.bc.Dirichlet('v', rhs=0)
    bc3 = psci.bc.Dirichlet('w', rhs=0)
    bc = {'top': [bc1, bc2, bc3]}
    jud_ns(geo_disc, bc, time_dependent=False)


@pytest.mark.pde_navierstokes
def test_navierstokes4():
    """
    3d
    time_dependent=True
    time_method=None
    time_interval = [0, 2.]
    time_step=0.1
    """
    geo = psci.geometry.Rectangular(
        origin=(0.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0))
    geo.add_boundary(name="top", criteria=lambda x, y, z: (z == 1.0))
    geo_disc = geo.discretize(method="uniform", npoints=40)
    bc1 = psci.bc.Dirichlet('u', rhs=0)
    bc2 = psci.bc.Dirichlet('v', rhs=0)
    bc3 = psci.bc.Dirichlet('w', rhs=0)
    bc = {'top': [bc1, bc2, bc3]}
    jud_ns(
        geo_disc,
        bc,
        time_dependent=True,
        time_interval=[0., 2.0],
        time_step=0.1)


@pytest.mark.pde_navierstokes
def test_navierstokes5():
    """
    3d
    time_dependent=True
    time_method=implicit
    time_interval = [0, 2.]
    time_step=0.1
    """
    geo = psci.geometry.Rectangular(
        origin=(0.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0))
    geo.add_boundary(name="top", criteria=lambda x, y, z: (z == 1.0))
    geo_disc = geo.discretize(method="uniform", npoints=40)
    bc1 = psci.bc.Dirichlet('u', rhs=0)
    bc2 = psci.bc.Dirichlet('v', rhs=0)
    bc3 = psci.bc.Dirichlet('w', rhs=0)
    bc = {'top': [bc1, bc2, bc3]}
    jud_ns(
        geo_disc,
        bc,
        time_dependent=True,
        time_interval=[0., 2.0],
        time_step=0.1,
        time_method="implicit")
