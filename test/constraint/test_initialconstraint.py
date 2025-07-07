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

import pytest

from ppsci import constraint
from ppsci import geometry
from ppsci import loss


def test_sdf_weight_of_initialconstraint():
    rect = geometry.TimeXGeometry(
        geometry.TimeDomain(0, 1),
        geometry.Rectangle((0, 0), (1, 1)),
    )
    ic = constraint.InitialConstraint(
        {"u": lambda out: out["u"]},
        {"u": 0},
        rect,
        {
            "dataset": "IterableNamedArrayDataset",
            "iters_per_epoch": 1,
            "batch_size": 16,
        },
        loss.MSELoss("mean"),
        weight_dict={"u": "sdf"},
        name="IC",
    )  # doctest: +SKIP

    input, _, _ = next(iter(ic.data_iter))
    assert "sdf" in input.keys()


if __name__ == "__main__":
    pytest.main()
