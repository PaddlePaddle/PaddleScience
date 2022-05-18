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


@pytest.mark.bc_Dirichlet
def test_Dirichlet0():
    """
    test0
    """
    indvar = sympy.symbols('x y')
    ic_v = psci.bc.Dirichlet('v', rhs=0.0)
    r = ic_v.discretize(indvar)
    eq = sympy.Function('v')(*indvar)
    assert r.formula == eq


@pytest.mark.bc_Dirichlet
def test_Dirichlet1():
    """
    test1
    """
    indvar = sympy.symbols('t x y z')
    ic_u = psci.bc.Dirichlet('u', rhs=lambda x, y, z: x * y * z)
    r = ic_u.discretize(indvar)
    eq = sympy.Function('u')(*indvar)
    assert r.formula == eq


@pytest.mark.bc_Dirichlet
def test_Dirichlet2():
    """
    test2
    """
    indvar = sympy.symbols('t x y z')
    ic_u = psci.bc.Dirichlet('u', rhs=lambda x, y, z: x * y * z, weight=0.5)
    r = ic_u.discretize(indvar)
    eq = sympy.Function('u')(*indvar)
    assert r.formula == eq
    assert ic_u.weight == 0.5
