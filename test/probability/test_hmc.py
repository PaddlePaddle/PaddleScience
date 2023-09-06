# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import pytest

from ppsci.probability.hmc import HamiltonianMonteCarlo

paddle.seed(1024)


@pytest.mark.parametrize("true_mean", [5.0])
@pytest.mark.parametrize("true_std", [1.0])
def test_HamiltonianMonteCarlo(true_mean, true_std):
    dist = paddle.distribution.Normal(true_mean, true_std)
    HMC = HamiltonianMonteCarlo(dist, path_len=1.5, step_size=0.25)
    trial = HMC.run_chain(2500, paddle.to_tensor(0.0))

    assert paddle.allclose(trial.mean(), true_mean, rtol=0.05)
    assert paddle.allclose(paddle.std(trial), true_std, rtol=0.05)


if __name__ == "__main__":
    pytest.main()
