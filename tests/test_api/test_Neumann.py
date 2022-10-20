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
import pytest


@pytest.mark.bc_Neumann
def test_Neumann0():
    """
    test0
    """
    indvar = sympy.symbols('x y')
    n = sympy.Symbol('n')
    bc_v = psci.bc.Neumann('v', rhs=0.0)
    r = bc_v.discretize(indvar)
    eq = sympy.Function('v')(*indvar)
    fo = sympy.Derivative(eq, n)
    assert r.formula == fo


@pytest.mark.bc_Neumann
def test_Neumann1():
    """
    test1
    """
    indvar = sympy.symbols('t x y z')
    n = sympy.Symbol('n')
    bc_u = psci.bc.Neumann('u', rhs=lambda x, y, z: x * y * z)
    r = bc_u.discretize(indvar)
    eq = sympy.Function('u')(*indvar)
    fo = sympy.Derivative(eq, n)
    assert r.formula == fo


@pytest.mark.bc_Neumann
def test_Neumann2():
    """
    test2
    """
    indvar = sympy.symbols('t x y z')
    n = sympy.Symbol('n')
    bc_u = psci.bc.Neumann('u', rhs=lambda x, y, z: x * y * z, weight=0.5)
    r = bc_u.discretize(indvar)
    eq = sympy.Function('u')(*indvar)
    fo = sympy.Derivative(eq, n)
    assert r.formula == fo
    assert bc_u.weight == 0.5
