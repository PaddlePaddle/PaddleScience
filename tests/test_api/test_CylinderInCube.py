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
test_CylinderInCube
"""


@pytest.mark.geometry_CylinderInCube
def test_CylinderInCube0():
    """
    set rectangular boundary
    """
    cc = (1.0, 0.0)
    cr = 0.5
    geo = psci.geometry.CylinderInCube(
        origin=(-8, -8, -0.5),
        extent=(25, 8, 0.5),
        circle_center=cc,
        circle_radius=cr)

    geo.add_boundary(name="left", criteria=lambda x, y, z: abs(x + 8.0) < 1e-4)
    geo.add_boundary(
        name="right", criteria=lambda x, y, z: abs(x - 25.0) < 1e-4)

    # discretize geometry
    geo_disc = geo.discretize(npoints=3000, method="sampling")

    assert geo_disc.interior.shape[1] == 3
    assert geo_disc.boundary["left"].shape[1] == 3
    assert geo_disc.boundary["right"].shape[1] == 3
    assert len(geo.criteria) == 2
    geo.delete_boundary("left")
    assert len(geo.criteria) == 1
    geo.clear_boundary()
    assert len(geo.criteria) == 0


@pytest.mark.geometry_CylinderInCube
def test_CylinderInCube1():
    """
    set rectangular boundary
    set circle boundary
    """
    cc = (0.0, 0.0)
    cr = 0.5
    geo = psci.geometry.CylinderInCube(
        origin=(-8, -8, -0.5),
        extent=(25, 8, 0.5),
        circle_center=cc,
        circle_radius=cr)

    geo.add_boundary(name="left", criteria=lambda x, y, z: abs(x + 8.0) < 1e-4)
    geo.add_boundary(
        name="right", criteria=lambda x, y, z: abs(x - 25.0) < 1e-4)
    geo.add_boundary(
        name="circle",
        criteria=lambda x, y, z: ((x - cc[0])**2 + (y - cc[1])**2 - cr**2) < 1e-4
    )

    # discretize geometry
    geo_disc = geo.discretize(npoints=3000, method="sampling")

    assert geo_disc.interior.shape[1] == 3
    assert geo_disc.boundary["left"].shape[1] == 3
    assert geo_disc.boundary["right"].shape[1] == 3
    assert geo_disc.boundary["circle"].shape[1] == 3
    assert len(geo.criteria) == 3
    geo.delete_boundary("circle")
    assert len(geo.criteria) == 2
    geo.clear_boundary()
    assert len(geo.criteria) == 0
