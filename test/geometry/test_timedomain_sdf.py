# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

from ppsci import geometry


def test_sdf_of_TimeXGeometry():
    timedomain = geometry.TimeDomain(0, 1, time_step=0.3)
    geom = geometry.Rectangle((0, 0), (1, 1))
    time_geom = geometry.TimeXGeometry(timedomain, geom)

    interior_points = time_geom.sample_interior(
        timedomain.num_timestamps * 3000, compute_sdf_derivatives=True
    )

    assert "sdf" in interior_points
    assert "sdf__x" in interior_points
    assert "sdf__y" in interior_points

    interior_points = {"x": np.linspace(-1, 1, dtype="float32").reshape((-1, 1))}
    geom = geometry.PointCloud(interior_points, ("x",))
    time_geom = geometry.TimeXGeometry(timedomain, geom)

    interior_points = time_geom.sample_interior(
        timedomain.num_timestamps * 10, compute_sdf_derivatives=True
    )

    assert "sdf" not in interior_points
    assert "sdf__x" not in interior_points
    assert "sdf__y" not in interior_points


if __name__ == "__main__":
    pytest.main()
