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
"""
Code below is heavily based on https://github.com/lululxvi/deepxde
"""

import numpy as np
import skopt

from .. import config


def sample(n_samples, dimension, sampler="pseudo"):
    """Generate pseudorandom or quasirandom samples in [0, 1]^dimension.

    Args:
        n_samples (int): The number of samples.
        dimension (int): Space dimension.
        sampler (str): One of the following: "pseudo" (pseudorandom), "LHS" (Latin
            hypercube sampling), "Halton" (Halton sequence), "Hammersley" (Hammersley
            sequence), or "Sobol" (Sobol sequence).
    Returns:
        np.ndarray: Generated random samples with shape of [n_samples, dimension].
    """
    if sampler == "pseudo":
        return pseudorandom(n_samples, dimension)
    if sampler in ["LHS", "Halton", "Hammersley", "Sobol"]:
        return quasirandom(n_samples, dimension, sampler)
    raise ValueError("f{sampler} sampling is not available.")


def pseudorandom(n_samples, dimension):
    """Pseudo random."""
    # If random seed is set, then the rng based code always returns the same random
    # number, which may not be what we expect.
    # rng = np.random.default_rng(config.random_seed)
    # return rng.random(size=(n_samples, dimension), dtype=config._dtype)
    return np.random.random(size=(n_samples, dimension)).astype(config._dtype)


def quasirandom(n_samples, dimension, sampler):
    """Quasi random"""
    # Certain points should be removed:
    # - Boundary points such as [..., 0, ...]
    # - Special points [0, 0, 0, ...] and [0.5, 0.5, 0.5, ...], which cause error in
    #   Hypersphere.random_points() and Hypersphere.random_boundary_points()
    skip = 0
    if sampler == "LHS":
        sampler = skopt.sampler.Lhs()
    elif sampler == "Halton":
        # 1st point: [0, 0, ...]
        sampler = skopt.sampler.Halton(min_skip=1, max_skip=1)
    elif sampler == "Hammersley":
        # 1st point: [0, 0, ...]
        if dimension == 1:
            sampler = skopt.sampler.Hammersly(min_skip=1, max_skip=1)
        else:
            sampler = skopt.sampler.Hammersly()
            skip = 1
    elif sampler == "Sobol":
        # 1st point: [0, 0, ...], 2nd point: [0.5, 0.5, ...]
        sampler = skopt.sampler.Sobol(randomize=False)
        if dimension < 3:
            skip = 1
        else:
            skip = 2
    space = [(0.0, 1.0)] * dimension
    return np.asarray(
        sampler.generate(space, n_samples + skip)[skip:], dtype=config._dtype)
