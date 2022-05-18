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
import numpy as np
import os


@pytest.mark.visu_save_vtk
def test_save_vtk0():
    """
    pde: laplace
    time_array=None
    """
    geo = psci.geometry.Rectangular(origin=(0.0, 0.0), extent=(1.0, 1.0))
    geo.add_boundary(name="top", criteria=lambda x, y: (y == 1.0))
    geo_disc = geo.discretize(method="uniform", npoints=10)

    l1 = len(geo_disc.interior)
    solution = [np.random.rand(l1, 3)]
    for k, v in geo_disc.boundary.items():
        solution.append(np.random.rand(len(v), 3))
    psci.visu.save_vtk(geo_disc=geo_disc, data=solution)
    assert os.path.exists("output-t1-p0.vtu")
    os.remove("output-t1-p0.vtu")


@pytest.mark.visu_save_vtk
def test_save_vtk1():
    """
    pde: laplace
    time_array=list
    """
    geo = psci.geometry.Rectangular(origin=(0.0, 0.0), extent=(1.0, 1.0))
    geo.add_boundary(name="top", criteria=lambda x, y: (y == 1.0))
    geo_disc = geo.discretize(method="uniform", npoints=10)

    l1 = len(geo_disc.interior)
    solution = [np.random.rand(l1, 3)]
    for k, v in geo_disc.boundary.items():
        solution.append(np.random.rand(len(v), 3))
    time_array = np.arange(10)
    psci.visu.save_vtk(geo_disc=geo_disc, data=solution, time_array=time_array)
    for i in range(1, len(time_array)):
        assert os.path.exists("output-t%s-p0.vtu" % i)
        os.remove("output-t%s-p0.vtu" % i)
