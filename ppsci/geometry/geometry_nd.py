"""Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
"""
Code below is heavily based on https://github.com/lululxvi/deepxde
"""


import itertools

import numpy as np
from scipy import stats
from sklearn import preprocessing

from ppsci.geometry import geometry
from ppsci.geometry import sampler
from ppsci.utils import misc


class Hypercube(geometry.Geometry):
    def __init__(self, xmin, xmax):
        if len(xmin) != len(xmax):
            raise ValueError("Dimensions of xmin and xmax do not match.")

        self.xmin = np.array(xmin, dtype="float32")
        self.xmax = np.array(xmax, dtype="float32")
        if np.any(self.xmin >= self.xmax):
            raise ValueError("xmin >= xmax")

        self.side_length = self.xmax - self.xmin
        super().__init__(
            len(xmin), (self.xmin, self.xmax), np.linalg.norm(self.side_length)
        )
        self.volume = np.prod(self.side_length)

    def is_inside(self, x):
        return np.logical_and(
            np.all(x >= self.xmin, axis=-1), np.all(x <= self.xmax, axis=-1)
        )

    def on_boundary(self, x):
        _on_boundary = np.logical_or(
            np.any(np.isclose(x, self.xmin), axis=-1),
            np.any(np.isclose(x, self.xmax), axis=-1),
        )
        return np.logical_and(self.is_inside(x), _on_boundary)

    def boundary_normal(self, x):
        _n = -np.isclose(x, self.xmin).astype("float32") + np.isclose(x, self.xmax)
        # For vertices, the normal is averaged for all directions
        idx = np.count_nonzero(_n, axis=-1) > 1
        if np.any(idx):
            print(
                f"Warning: {self.__class__.__name__} boundary_normal called on vertices. "
                "You may use PDE(..., exclusions=...) to exclude the vertices."
            )
            l = np.linalg.norm(_n[idx], axis=-1, keepdims=True)
            _n[idx] /= l
        return _n

    def uniform_points(self, n, boundary=True):
        dx = (self.volume / n) ** (1 / self.ndim)
        xi = []
        for i in range(self.ndim):
            ni = int(np.ceil(self.side_length[i] / dx))
            if boundary:
                xi.append(
                    np.linspace(self.xmin[i], self.xmax[i], num=ni, dtype="float32")
                )
            else:
                xi.append(
                    np.linspace(
                        self.xmin[i],
                        self.xmax[i],
                        num=ni + 1,
                        endpoint=False,
                        dtype="float32",
                    )[1:]
                )
        x = np.array(list(itertools.product(*xi)))
        if len(x) > n:
            x = x[0:n]
        return x

    def random_points(self, n, random="pseudo"):
        x = sampler.sample(n, self.ndim, random)
        # print(f"Hypercube's range: {self.__class__.__name__}", self.xmin, self.xmax)
        return (self.xmax - self.xmin) * x + self.xmin

    def random_boundary_points(self, n, random="pseudo"):
        x = sampler.sample(n, self.ndim, random)
        # Randomly pick a dimension
        rand_dim = np.random.randint(self.ndim, size=n)
        # Replace value of the randomly picked dimension with the nearest boundary value (0 or 1)
        x[np.arange(n), rand_dim] = np.round(x[np.arange(n), rand_dim])
        return (self.xmax - self.xmin) * x + self.xmin

    def periodic_point(self, x, component):
        y = misc.convert_to_array(x, self.dim_keys)
        _on_xmin = np.isclose(y[:, component], self.xmin[component])
        _on_xmax = np.isclose(y[:, component], self.xmax[component])
        y[:, component][_on_xmin] = self.xmax[component]
        y[:, component][_on_xmax] = self.xmin[component]
        return misc.convert_to_dict(y, self.dim_keys)


class Hypersphere(geometry.Geometry):
    def __init__(self, center, radius):
        self.center = np.array(center, dtype="float32")
        self.radius = radius
        super().__init__(
            len(center), (self.center - radius, self.center + radius), 2 * radius
        )

        self._r2 = radius**2

    def is_inside(self, x):
        return np.linalg.norm(x - self.center, axis=-1) <= self.radius

    def on_boundary(self, x):
        return np.isclose(np.linalg.norm(x - self.center, axis=-1), self.radius)

    def boundary_normal(self, x):
        _n = x - self.center
        l = np.linalg.norm(_n, axis=-1, keepdims=True)
        _n = _n / l * np.isclose(l, self.radius)
        return _n

    def random_points(self, n, random="pseudo"):
        # https://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability
        if random == "pseudo":
            U = np.random.rand(n, 1).astype("float32")
            X = np.random.normal(size=(n, self.ndim)).astype("float32")
        else:
            rng = sampler.sample(n, self.ndim + 1, random)
            U, X = rng[:, 0:1], rng[:, 1:]  # Error if X = [0, 0, ...]
            X = stats.norm.ppf(X).astype("float32")
        X = preprocessing.normalize(X)
        X = U ** (1 / self.ndim) * X
        return self.radius * X + self.center

    def random_boundary_points(self, n, random="pseudo"):
        # http://mathworld.wolfram.com/HyperspherePointPicking.html
        if random == "pseudo":
            X = np.random.normal(size=(n, self.ndim)).astype("float32")
        else:
            U = sampler.sample(
                n, self.ndim, random
            )  # Error for [0, 0, ...] or [0.5, 0.5, ...]
            X = stats.norm.ppf(U).astype("float32")
        X = preprocessing.normalize(X)
        return self.radius * X + self.center
