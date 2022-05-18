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
"""
test_rectangular
"""


@pytest.mark.geometry_rectangular
def test_rectangular0():
    """
    2d
    discretize_method="uniform"
    """
    geo2d = psci.geometry.Rectangular(origin=(0.0, 0.0), extent=(1.0, 1.0))
    geo2d.add_boundary("top", criteria=lambda x, y: y == 1.0)
    geo2d.add_boundary("down", criteria=lambda x, y: y == 0.0)
    geo_disc = geo2d.discretize(method="uniform", npoints=10)
    assert geo_disc.interior.shape == (3, 2)
    for k, v in geo_disc.boundary.items():
        assert v.shape == (3, 2)
    assert len(geo2d.criteria) == 2
    geo2d.delete_boundary("top")
    assert len(geo2d.criteria) == 1
    geo2d.clear_boundary()
    assert len(geo2d.criteria) == 0


@pytest.mark.geometry_rectangular
def test_rectangular1():
    """
    2d
    discretize_method="sampling"
    """
    geo2d = psci.geometry.Rectangular(origin=(0.0, 0.0), extent=(2.0, 1.0))
    geo2d.add_boundary("top", criteria=lambda x, y: y == 1.0)
    geo2d.add_boundary("down", criteria=lambda x, y: y == 0.0)
    geo2d.add_boundary("left", criteria=lambda x, y: x == 0.0)
    geo2d.add_boundary("right", criteria=lambda x, y: x == 2.0)
    geo_disc = geo2d.discretize(method="sampling", npoints=10)
    assert geo_disc.interior.shape[1] == 2
    for k, v in geo_disc.boundary.items():
        assert v.shape[1] == 2
    assert len(geo2d.criteria) == 4
    geo2d.delete_boundary("top")
    assert len(geo2d.criteria) == 3
    geo2d.clear_boundary()
    assert len(geo2d.criteria) == 0


@pytest.mark.geometry_rectangular
def test_rectangular2():
    """
    3d
    discretize_method="uniform"
    npoints: sequence
    """
    geo3d = psci.geometry.Rectangular(
        origin=(0.0, 0.0, 0.0), extent=(1.0, 2.0, 4.0))
    geo3d.add_boundary("top", criteria=lambda x, y, z: z == 4.0)
    geo3d.add_boundary("down", criteria=lambda x, y, z: z == 0.0)
    geo_disc = geo3d.discretize(method="uniform", npoints=(5, 4, 3))
    assert geo_disc.interior.shape == (20, 3)
    for k, v in geo_disc.boundary.items():
        assert v.shape == (20, 3)
    assert len(geo3d.criteria) == 2
    geo3d.delete_boundary("top")
    assert len(geo3d.criteria) == 1
    geo3d.clear_boundary()
    assert len(geo3d.criteria) == 0


@pytest.mark.geometry_rectangular
def test_rectangular3():
    """
    3d
    discretize_method="sampling"
    """
    geo3d = psci.geometry.Rectangular(
        origin=(0.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0))
    geo3d.add_boundary("top", criteria=lambda x, y, z: z == 1.0)
    geo3d.add_boundary("down", criteria=lambda x, y, z: z == 0.0)
    geo_disc = geo3d.discretize(method="sampling", npoints=400)
    assert geo_disc.interior.shape[1] == 3
    assert geo_disc.boundary["top"].shape[1] == 3
    assert geo_disc.boundary["down"].shape[1] == 3
    assert len(geo3d.criteria) == 2
    geo3d.delete_boundary("top")
    assert len(geo3d.criteria) == 1
    geo3d.clear_boundary()
    assert len(geo3d.criteria) == 0


#TODO:
# sampling mathod: npoints as list
