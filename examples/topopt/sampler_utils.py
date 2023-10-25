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

import numpy as np


def uniform_sampler() -> callable:
    return lambda: np.random.randint(1, 99)


def poisson_sampler(lam: int) -> callable:
    def func():
        iter_ = max(np.random.poisson(lam), 1)
        iter_ = min(iter_, 99)
        return iter_

    return func


def generate_sampler(sampler_type: str = "Fixed", num: int = 0) -> callable:
    """Generate sampler for the number of initial iteration steps

    Args:
        sampler_type (str): "Poisson" for poisson sampler; "Uniform" for uniform sampler; "Fixed" for choosing a fixed number of initial iteration steps.
        num (int): If `sampler_type` == "Poisson", `num` specifies the poisson rate parameter; If `sampler_type` == "Fixed", `num` specifies the fixed number of initial iteration steps.

    Returns:
        sampler (callable): sampler for the number of initial iteration steps
    """
    if sampler_type == "Poisson":
        return poisson_sampler(num)
    elif sampler_type == "Uniform":
        return uniform_sampler()
    else:
        return lambda: num
