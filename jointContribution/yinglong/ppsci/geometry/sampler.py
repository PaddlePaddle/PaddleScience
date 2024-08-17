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

"""
Code below is heavily based on [https://github.com/lululxvi/deepxde](https://github.com/lululxvi/deepxde)
"""

import numpy as np
import paddle
import skopt
from typing_extensions import Literal


def sample(
    n_samples: int, ndim: int, method: Literal["pseudo", "LHS"] = "pseudo"
) -> np.ndarray:
    """Generate pseudorandom or quasirandom samples in [0, 1]^ndim.

    Args:
        n_samples (int): The number of samples.
        ndim (int): Number of dimension.
        method (str): One of the following: "pseudo" (pseudorandom), "LHS" (Latin
            hypercube sampling), "Halton" (Halton sequence), "Hammersley" (Hammersley
            sequence), or "Sobol" (Sobol sequence).
    Returns:
        np.ndarray: Generated random samples with shape of [n_samples, ndim].
    """
    if method == "pseudo":
        return pseudorandom(n_samples, ndim)
    if method in ["LHS", "Halton", "Hammersley", "Sobol"]:
        return quasirandom(n_samples, ndim, method)
    raise ValueError(f"Sampling method({method}) is not available.")


def pseudorandom(n_samples: int, ndim: int) -> np.ndarray:
    """Pseudo random."""
    # If random seed is set, then the rng based code always returns the same random
    # number, which may not be what we expect.
    # rng = np.random.default_rng(config.random_seed)
    # return rng.random(size=(n_samples, ndim), dtype=dtype=paddle.get_default_dtype())
    return np.random.random(size=(n_samples, ndim)).astype(
        dtype=paddle.get_default_dtype()
    )


def quasirandom(
    n_samples: int, ndim: int, method: Literal["pseudo", "LHS"]
) -> np.ndarray:
    """Quasi random"""
    # Certain points should be removed:
    # - Boundary points such as [..., 0, ...]
    # - Special points [0, 0, 0, ...] and [0.5, 0.5, 0.5, ...], which cause error in
    #   Hypersphere.random_points() and Hypersphere.random_boundary_points()
    skip = 0
    if method == "LHS":
        sampler = skopt.sampler.Lhs()
    elif method == "Halton":
        # 1st point: [0, 0, ...]
        sampler = skopt.sampler.Halton(min_skip=1, max_skip=1)
    elif method == "Hammersley":
        # 1st point: [0, 0, ...]
        if ndim == 1:
            sampler = skopt.sampler.Hammersly(min_skip=1, max_skip=1)
        else:
            sampler = skopt.sampler.Hammersly()
            skip = 1
    elif method == "Sobol":
        # 1st point: [0, 0, ...], 2nd point: [0.5, 0.5, ...]
        sampler = skopt.sampler.Sobol(randomize=False)
        if ndim < 3:
            skip = 1
        else:
            skip = 2
    space = [(0.0, 1.0)] * ndim
    return np.asarray(
        sampler.generate(space, n_samples + skip)[skip:],
        dtype=paddle.get_default_dtype(),
    )
