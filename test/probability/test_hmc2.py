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
from paddle.distribution import Bernoulli
from paddle.distribution import Normal

from ppsci.probability.hmc import HamiltonianMonteCarlo

paddle.seed(1024)


# Example case on Pyro
def test_HamiltonianMonteCarlo2():
    true_coefs = paddle.to_tensor([1.0, 2.0, 3.0])
    data = paddle.randn((2000, 3))
    dim = 3

    labels = (
        Bernoulli(paddle.nn.functional.sigmoid(paddle.matmul(data, true_coefs)))
        .sample([1])
        .squeeze()
    )
    rv_beta = Normal(paddle.zeros(dim), paddle.ones(dim))

    def log_prior(**kwargs):
        return paddle.sum(rv_beta.log_prob(kwargs["beta"]))

    def log_likelihood(**kwargs):
        p = paddle.nn.functional.sigmoid(paddle.matmul(data, kwargs["beta"]))
        return paddle.sum(labels * paddle.log(p) + (1 - labels) * paddle.log(1 - p))

    # log posterior
    def log_posterior(**kwargs):
        return log_prior(**kwargs) + log_likelihood(**kwargs)

    initial_params = {"beta": paddle.to_tensor([0.5, 0.5, 0.5])}

    HMC = HamiltonianMonteCarlo(
        log_posterior, path_len=0.040, step_size=0.0025, num_warmup_steps=500
    )
    trial = HMC.run_chain(500, initial_params)

    means = trial["beta"].mean(axis=0)
    assert paddle.allclose(means[0], paddle.to_tensor(true_coefs[0]), rtol=0.2)
    assert paddle.allclose(means[1], paddle.to_tensor(true_coefs[1]), rtol=0.2)
    assert paddle.allclose(means[2], paddle.to_tensor(true_coefs[2]), rtol=0.2)


if __name__ == "__main__":
    pytest.main()
