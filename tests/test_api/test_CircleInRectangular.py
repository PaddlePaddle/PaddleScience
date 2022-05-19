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
test_CircleInRectangular
"""


@pytest.mark.geometry_CircleInRectangular
def test_CircleInRectangular0():
    """
    set rectangular boundary
    """
    geo = psci.geometry.CircleInRectangular(
        origin=(0.0, 0.0),
        extent=(1.0, 1.0),
        circle_center=(0.5, 0.5),
        circle_radius=0.1)
    geo.add_boundary("top", criteria=lambda x, y: y == 1.0)
    geo.add_boundary("down", criteria=lambda x, y: y == 0.0)
    geo.add_boundary("left", criteria=lambda x, y: x == 0.0)
    geo.add_boundary("right", criteria=lambda x, y: x == 1.0)
    geo_disc = geo.discretize(method='sampling', npoints=10)

    assert geo_disc.interior.shape[1] == 2
    assert geo_disc.boundary["top"].shape[1] == 2
    assert geo_disc.boundary["down"].shape[1] == 2
    assert geo_disc.boundary["left"].shape[1] == 2
    assert geo_disc.boundary["right"].shape[1] == 2
    assert len(geo.criteria) == 4
    geo.delete_boundary("top")
    assert len(geo.criteria) == 3
    geo.clear_boundary()
    assert len(geo.criteria) == 0


@pytest.mark.geometry_CircleInRectangular
def test_CircleInRectangular1():
    """
    set rectangular boundary
    set circle boundary
    """
    geo = psci.geometry.CircleInRectangular(
        origin=(0.0, 0.0),
        extent=(1.0, 1.0),
        circle_center=(0.5, 0.5),
        circle_radius=0.1)
    geo.add_boundary("top", criteria=lambda x, y: y == 1.0)
    geo.add_boundary("down", criteria=lambda x, y: y == 0.0)
    geo.add_boundary("left", criteria=lambda x, y: x == 0.0)
    geo.add_boundary("right", criteria=lambda x, y: x == 1.0)
    geo.add_boundary(
        "circle",
        criteria=lambda x, y: ((x - 0.5)**2 + (y - 0.5)**2 - 0.1**2) < 1e-4)
    geo_disc = geo.discretize(method='sampling', npoints=30000)

    assert geo_disc.interior.shape[1] == 2
    assert geo_disc.boundary["top"].shape[1] == 2
    assert geo_disc.boundary["down"].shape[1] == 2
    assert geo_disc.boundary["left"].shape[1] == 2
    assert geo_disc.boundary["right"].shape[1] == 2
    assert geo_disc.boundary["circle"].shape[1] == 2
    assert len(geo.criteria) == 5
    geo.delete_boundary("top")
    assert len(geo.criteria) == 4
    geo.clear_boundary()
    assert len(geo.criteria) == 0
