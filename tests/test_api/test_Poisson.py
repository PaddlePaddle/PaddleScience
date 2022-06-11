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
import pytest
import sympy
import types
import numpy as np


def jud_poisson(geo_disc, bc, rhs=None):
    """
    test poisson equation
    """
    dim = geo_disc.interior.shape[1]
    pde = psci.pde.Poisson(dim=dim, rhs=rhs)
    for k in geo_disc.boundary.keys():
        pde.add_bc(k, *bc[k])
    pde_disc = pde.discretize(geo_disc=geo_disc)

    indvar, dvar, equations = None, None, 0
    if dim == 2:
        indvar = sympy.symbols('x y')
        dvar = sympy.Function('u')(*indvar)
        for i in range(dim):
            equations += dvar.diff(indvar[i]).diff(indvar[i])

    elif dim == 3:
        indvar = sympy.symbols('x y z')
        dvar = sympy.Function('u')(*indvar)
        for i in range(dim):
            equations += dvar.diff(indvar[i]).diff(indvar[i])

    data = geo_disc.interior.T.tolist()
    rhs_disc = pde_disc.rhs[0](*data)

    assert pde.indvar == list(indvar)
    assert pde.dvar == [dvar]
    assert pde.equations == [equations]
    assert type(pde.rhs[0]) == types.LambdaType
    assert pde.weight == 1.
    for k in pde_disc.bc.keys():
        length = len(pde_disc.bc[k])
        for i in range(length):
            assert isinstance(pde_disc.bc[k][i], type(bc[k][i]))
    assert np.allclose(pde_disc.rhs_disc['interior'], rhs_disc)


@pytest.mark.pde_poisson
def test_poisson0():
    """
    2d
    """
    geo = psci.geometry.Rectangular(origin=(0.0, 0.0), extent=(1.0, 1.0))
    geo.add_boundary(name="top", criteria=lambda x, y: (y == 1.0))
    geo_disc = geo.discretize(method="uniform", npoints=10)

    bc1 = psci.bc.Dirichlet('u', rhs=0)
    bc2 = psci.bc.Dirichlet('v', rhs=0)
    bc = {'top': [bc1, bc2]}
    rhs = lambda x, y: np.sin(x) * np.sin(y)
    jud_poisson(geo_disc, bc, rhs)


@pytest.mark.pde_poisson
def test_poisson1():
    """
    3d
    """
    geo = psci.geometry.Rectangular(
        origin=(0.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0))
    geo.add_boundary(name="top", criteria=lambda x, y, z: (z == 1.0))
    geo_disc = geo.discretize(method="uniform", npoints=40)

    bc1 = psci.bc.Dirichlet('u', rhs=0)
    bc2 = psci.bc.Dirichlet('v', rhs=0)
    bc3 = psci.bc.Dirichlet('w', rhs=0)
    bc = {'top': [bc1, bc2, bc3]}
    rhs = lambda x, y, z: np.sin(x) * np.sin(y) * np.tanh(z)
    jud_poisson(geo_disc, bc, rhs)


@pytest.mark.pde_poisson
def test_poisson2():
    """
    3d
    two bc, set bc rhs as lambda
    """
    geo = psci.geometry.Rectangular(
        origin=(0.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0))
    geo.add_boundary(name="top", criteria=lambda x, y, z: (z == 1.0))
    geo.add_boundary(name="down", criteria=lambda x, y, z: (z == 0.0))
    geo_disc = geo.discretize(method="uniform", npoints=40)

    bc1 = psci.bc.Dirichlet('u', rhs=0)
    bc2 = psci.bc.Dirichlet('v', rhs=0)
    bc3 = psci.bc.Dirichlet('w', rhs=0)
    bc4 = psci.bc.Dirichlet('u', rhs=lambda x, y, z: x * y * z)
    bc5 = psci.bc.Dirichlet('v', rhs=lambda x, y, z: x + y + z)
    bc6 = psci.bc.Dirichlet('w', rhs=lambda x, y, z: x * y - z)

    bc = {'top': [bc1, bc2, bc3], 'down': [bc4, bc5, bc6]}
    rhs = lambda x, y, z: np.sin(x) * np.cos(y) * np.tan(z)
    jud_poisson(geo_disc, bc, rhs)
