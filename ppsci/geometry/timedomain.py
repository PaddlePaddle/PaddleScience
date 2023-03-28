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

from ppsci.geometry import geometry
from ppsci.geometry import geometry_1d
from ppsci.geometry import geometry_2d
from ppsci.geometry import geometry_3d
from ppsci.geometry import geometry_nd
from ppsci.geometry import mesh
from ppsci.utils import misc


class TimeDomain(geometry_1d.Interval):
    def __init__(self, t0, t1, time_step=None, timestamps=None):
        super().__init__(t0, t1)
        self.t0 = t0
        self.t1 = t1
        self.time_step = time_step
        self.timestamps = np.array(timestamps).reshape([-1])
        if time_step is not None:
            if time_step <= 0:
                raise ValueError(f"time_step({time_step}) must larger than 0.")
            self.num_timestamp = (
                None if time_step is None else (int(np.ceil(self.diam / time_step)) + 1)
            )
        elif timestamps is not None:
            self.num_timestamp = len(timestamps)

    def on_initial(self, t):
        return np.isclose(t, self.t0).flatten()


class TimeXGeometry(geometry.Geometry):
    def __init__(self, timedomain, geometry):
        self.timedomain = timedomain
        self.geometry = geometry
        self.ndim = geometry.ndim + timedomain.ndim

    @property
    def dim_keys(self):
        return ["t"] + self.geometry.dim_keys

    def on_boundary(self, x):
        # [N, txyz]
        return self.geometry.on_boundary(x[:, 1:])

    def on_initial(self, x):
        # [N, 1]
        return self.timedomain.on_initial(x[:, :1])

    def boundary_normal(self, x):
        # x: [N, txy(z)]
        normal = self.geometry.boundary_normal(x[:, 1:])
        return np.hstack((x[:, :1], normal))

    def uniform_points(self, n, boundary=True):
        """Uniform points on the spatio-temporal domain.

        Geometry volume ~ bbox.
        Time volume ~ diam.
        """
        if self.timedomain.time_step is not None:
            # exclude start time t0
            nt = int(np.ceil(self.timedomain.diam / self.timedomain.time_step))
            nx = int(np.ceil(n / nt))
        elif self.timedomain.timestamps is not None:
            # exclude start time t0
            nt = self.timedomain.num_timestamp - 1
        else:
            nx = int(
                np.ceil(
                    (
                        n
                        * np.prod(self.geometry.bbox[1] - self.geometry.bbox[0])
                        / self.timedomain.diam
                    )
                    ** 0.5
                )
            )
            nt = int(np.ceil(n / nx))
        x = self.geometry.uniform_points(nx, boundary=boundary)
        nx = len(x)
        if boundary and (
            self.timedomain.time_step is None and self.timedomain.timestamps is None
        ):
            t = self.timedomain.uniform_points(nt, boundary=True)
        else:
            t = np.linspace(
                self.timedomain.t1,
                self.timedomain.t0,
                num=nt,
                endpoint=boundary,
                dtype="float32",
            )[:, None][::-1]
        tx = []
        for ti in t:
            tx.append(np.hstack((np.full([nx, 1], float(ti), dtype="float32"), x)))
        tx = np.vstack(tx)
        if len(tx) > n:
            tx = tx[:n]
        return tx

    def random_points(self, n, random="pseudo"):
        # time evenly and geometry random, if time_step if specified
        if self.timedomain.time_step is not None:
            nt = int(np.ceil(self.timedomain.diam / self.timedomain.time_step))
            t = np.linspace(
                self.timedomain.t1,
                self.timedomain.t0,
                num=nt,
                endpoint=False,
                dtype="float32",
            )[:, None][
                ::-1
            ]  # [nt, 1]
            nx = int(np.ceil(n / nt))
            x = self.geometry.random_points(nx, random)
            tx = []
            for ti in t:
                tx.append(np.hstack((np.full([nx, 1], float(ti), dtype="float32"), x)))
            tx = np.vstack(tx)
            if len(tx) > n:
                tx = tx[:n]
            return tx
        elif self.timedomain.timestamps is not None:
            nt = self.timedomain.num_timestamp - 1
            t = self.timedomain.timestamps[1:]
            nx = int(np.ceil(n / nt))
            x = self.geometry.random_points(nx, random)
            tx = []
            for ti in t:
                tx.append(np.hstack((np.full([nx, 1], float(ti), dtype="float32"), x)))
            tx = np.vstack(tx)
            if len(tx) > n:
                tx = tx[:n]
            return tx

        if isinstance(self.geometry, geometry_1d.Interval):
            geom = geometry_2d.Rectangle(
                [self.timedomain.t0, self.geometry.l],
                [self.timedomain.t1, self.geometry.r],
            )
            return geom.random_points(n, random=random)

        if isinstance(self.geometry, geometry_2d.Rectangle):
            geom = geometry_3d.Cuboid(
                [self.timedomain.t0, self.geometry.xmin[0], self.geometry.xmin[1]],
                [self.timedomain.t1, self.geometry.xmax[0], self.geometry.xmax[1]],
            )
            return geom.random_points(n, random=random)

        if isinstance(self.geometry, (geometry_3d.Cuboid, geometry_nd.Hypercube)):
            geom = geometry_nd.Hypercube(
                np.append(self.timedomain.t0, self.geometry.xmin),
                np.append(self.timedomain.t1, self.geometry.xmax),
            )
            return geom.random_points(n, random=random)

        x = self.geometry.random_points(n, random=random)
        t = self.timedomain.random_points(n, random=random)
        t = np.random.permutation(t)
        return np.hstack((t, x))

    def uniform_boundary_points(self, n):
        """Uniform boundary points on the spatio-temporal domain.

        Geometry surface area ~ bbox.
        Time surface area ~ diam.
        """
        if self.geometry.ndim == 1:
            nx = 2
        else:
            s = 2 * sum(
                map(
                    lambda l: l[0] * l[1],
                    itertools.combinations(
                        self.geometry.bbox[1] - self.geometry.bbox[0], 2
                    ),
                )
            )
            nx = int((n * s / self.timedomain.diam) ** 0.5)
        nt = int(np.ceil(n / nx))
        x = self.geometry.uniform_boundary_points(nx)
        nx = len(x)
        t = np.linspace(
            self.timedomain.t1,
            self.timedomain.t0,
            num=nt,
            endpoint=False,
            dtype="float32",
        )[:, None][::-1]
        tx = []
        for ti in t:
            tx.append(np.hstack((np.full([nx, 1], float(ti), dtype="float32"), x)))
        tx = np.vstack(tx)
        if len(tx) > n:
            tx = tx[:n]
        return tx

    def random_boundary_points(self, n, random="pseudo"):
        if self.timedomain.time_step is not None:
            # exclude start time t0
            nt = int(np.ceil(self.timedomain.diam / self.timedomain.time_step))
            t = np.linspace(
                self.timedomain.t1,
                self.timedomain.t0,
                num=nt,
                endpoint=False,
                dtype="float32",
            )[:, None][::-1]
            nx = int(np.ceil(n / nt))

            if isinstance(self.geometry, mesh.Mesh):
                x, _n, a = self.geometry.random_boundary_points(nx, random=random)
            else:
                x = self.geometry.random_boundary_points(nx, random=random)
            # TODO(sensen): decouple uniform and random boundary sample.
            # x = self.geometry.uniform_boundary_points(nx)

            t_x = []
            if isinstance(self.geometry, mesh.Mesh):
                t_normal = []
                t_area = []

            for ti in t:
                t_x.append(np.hstack((np.full([nx, 1], float(ti), dtype="float32"), x)))
                if isinstance(self.geometry, mesh.Mesh):
                    t_normal.append(
                        np.hstack((np.full([nx, 1], float(ti), dtype="float32"), _n))
                    )
                    t_area.append(
                        np.hstack((np.full([nx, 1], float(ti), dtype="float32"), a))
                    )

            t_x = np.vstack(t_x)
            if isinstance(self.geometry, mesh.Mesh):
                t_normal = np.vstack(t_normal)
                t_area = np.vstack(t_area)

            if len(t_x) > n:
                t_x = t_x[:n]
                if isinstance(self.geometry, mesh.Mesh):
                    t_normal = t_normal[:n]
                    t_area = t_area[:n]

            if isinstance(self.geometry, mesh.Mesh):
                return t_x, t_normal, t_area
            else:
                return t_x
        elif self.timedomain.timestamps is not None:
            # exclude start time t0
            nt = self.timedomain.num_timestamp - 1
            t = self.timedomain.timestamps[1:]
            nx = int(np.ceil(n / nt))

            if isinstance(self.geometry, mesh.Mesh):
                x, _n, a = self.geometry.random_boundary_points(nx, random=random)
            else:
                x = self.geometry.random_boundary_points(nx, random=random)
            # TODO(sensen): decouple uniform and random boundary sample.
            # x = self.geometry.uniform_boundary_points(nx)

            t_x = []
            if isinstance(self.geometry, mesh.Mesh):
                t_normal = []
                t_area = []

            for ti in t:
                t_x.append(np.hstack((np.full([nx, 1], float(ti), dtype="float32"), x)))
                if isinstance(self.geometry, mesh.Mesh):
                    t_normal.append(
                        np.hstack((np.full([nx, 1], float(ti), dtype="float32"), _n))
                    )
                    t_area.append(
                        np.hstack((np.full([nx, 1], float(ti), dtype="float32"), a))
                    )

            t_x = np.vstack(t_x)
            if isinstance(self.geometry, mesh.Mesh):
                t_normal = np.vstack(t_normal)
                t_area = np.vstack(t_area)

            if len(t_x) > n:
                t_x = t_x[:n]
                if isinstance(self.geometry, mesh.Mesh):
                    t_normal = t_normal[:n]
                    t_area = t_area[:n]

            if isinstance(self.geometry, mesh.Mesh):
                return t_x, t_normal, t_area
            else:
                return t_x
        else:
            if isinstance(self.geometry, mesh.Mesh):
                x, _n, a = self.geometry.random_boundary_points(n, random=random)
            else:
                x = self.geometry.random_boundary_points(n, random=random)

            t = self.timedomain.random_points(n, random=random)
            t = np.random.permutation(t)

            t_x = np.hstack((t, x))

            if isinstance(self.geometry, mesh.Mesh):
                t_normal = np.hstack((_n, t))
                t_area = np.hstack((_n, t))
                return t_x, t_normal, t_area
            else:
                return t_x

    def uniform_initial_points(self, n):
        x = self.geometry.uniform_points(n, True)
        t = self.timedomain.t0
        if len(x) > n:
            x = x[:n]
        return np.hstack((np.full([n, 1], t, dtype="float32"), x))

    def random_initial_points(self, n, random="pseudo"):
        x = self.geometry.random_points(n, random=random)
        t = self.timedomain.t0
        return np.hstack((np.full([n, 1], t, dtype="float32"), x))

    def periodic_point(self, x, component):
        xp = self.geometry.periodic_point(x[:, 1:], component)
        return np.hstack((x[:, :1], xp))

    def sample_initial_interior(
        self, n: int, random: str = "pseudo", criteria=None, evenly=False
    ):
        """Sample random points in the time-geometry and return those meet criteria."""
        x = np.empty(shape=(n, self.ndim), dtype="float32")
        _size, _ntry, _nsuc = 0, 0, 0
        while _size < n:
            if evenly:
                points = self.uniform_initial_points(n)
            else:
                points = self.random_initial_points(n, random)

            if criteria is not None:
                criteria_mask = criteria(*np.split(points, self.ndim, axis=1)).flatten()
                points = points[criteria_mask]

            if len(points) > n - _size:
                points = points[: n - _size]
            x[_size : _size + len(points)] = points

            _size += len(points)
            _ntry += 1
            if len(points) > 0:
                _nsuc += 1

            if _ntry >= 1000 and _nsuc == 0:
                raise RuntimeError("sample initial interior failed")
        return misc.convert_to_dict(x, self.dim_keys)

    def __str__(self) -> str:
        """Return the name of class"""
        _str = ", ".join(
            [
                self.__class__.__name__,
                f"ndim = {self.ndim}",
                f"bbox = (time){self.timedomain.bbox} x (space){self.geometry.bbox}",
                f"diam = (time){self.timedomain.diam} x (space){self.geometry.diam}",
                f"dim_keys = {self.dim_keys}",
            ]
        )
        return _str
