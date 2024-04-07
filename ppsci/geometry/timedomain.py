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

from __future__ import annotations

import itertools
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import paddle

from ppsci.geometry import geometry
from ppsci.geometry import geometry_1d
from ppsci.geometry import geometry_2d
from ppsci.geometry import geometry_3d
from ppsci.geometry import geometry_nd
from ppsci.geometry import mesh
from ppsci.utils import misc


class TimeDomain(geometry_1d.Interval):
    """Class for timedomain, an special interval geometry.

    Args:
        t0 (float): Start of time.
        t1 (float): End of time.
        time_step (Optional[float]): Step interval of time. Defaults to None.
        timestamps (Optional[Tuple[float, ...]]): List of timestamps.
            Defaults to None.

    Examples:
        >>> import ppsci
        >>> geom = ppsci.geometry.TimeDomain(0, 1)
    """

    def __init__(
        self,
        t0: float,
        t1: float,
        time_step: Optional[float] = None,
        timestamps: Optional[Tuple[float, ...]] = None,
    ):
        super().__init__(t0, t1)
        self.t0 = t0
        self.t1 = t1
        self.time_step = time_step
        if timestamps is None:
            self.timestamps = None
        else:
            self.timestamps = np.array(
                timestamps, dtype=paddle.get_default_dtype()
            ).reshape([-1])
        if time_step is not None:
            if time_step <= 0:
                raise ValueError(f"time_step({time_step}) must be larger than 0.")
            self.num_timestamps = int(np.ceil((t1 - t0) / time_step)) + 1
        elif timestamps is not None:
            self.num_timestamps = len(timestamps)

    def on_initial(self, t: np.ndarray) -> np.ndarray:
        """Check if a specific time is on the initial time point.

        Args:
            t (np.ndarray): The time to be checked.

        Returns:
            np.ndarray: Bool numpy array of whether the specific time is on the initial time point.

        Examples:
            >>> import paddle
            >>> import ppsci
            >>> geom = ppsci.geometry.TimeDomain(0, 1)
            >>> T = [0, 0.01, 0.126, 0.2, 0.3]
            >>> check = geom.on_initial(T)
            >>> print(check)
            [ True False False False False]
        """
        return np.isclose(t, self.t0).flatten()


class TimeXGeometry(geometry.Geometry):
    """Class for combination of time and geometry.

    Args:
        timedomain (TimeDomain): TimeDomain object.
        geometry (geometry.Geometry): Geometry object.

    Examples:
        >>> import ppsci
        >>> timedomain = ppsci.geometry.TimeDomain(0, 1)
        >>> geom = ppsci.geometry.Rectangle((0, 0), (1, 1))
        >>> time_geom = ppsci.geometry.TimeXGeometry(timedomain, geom)
    """

    def __init__(self, timedomain: TimeDomain, geometry: geometry.Geometry):
        self.timedomain = timedomain
        self.geometry = geometry
        self.ndim = geometry.ndim + timedomain.ndim

    @property
    def dim_keys(self):
        return ("t",) + self.geometry.dim_keys

    def on_boundary(self, x):
        # [N, ndim(txyz)]
        return self.geometry.on_boundary(x[:, 1:])

    def on_initial(self, x):
        # [N, 1(t)]
        return self.timedomain.on_initial(x[:, :1])

    def boundary_normal(self, x):
        # x: [N, ndim(txyz)]
        normal = self.geometry.boundary_normal(x[:, 1:])
        return np.hstack((x[:, :1], normal))

    def uniform_points(self, n: int, boundary: bool = True) -> np.ndarray:
        """Uniform points on the spatial-temporal domain.
        Geometry volume ~ bbox.
        Time volume ~ diam.

        Args:
            n (int): The total number of sample points to be generated.
            boundary (bool): Indicates whether boundary points are included, default is True.

        Returns:
            np.ndarray: a set of spatial-temporal coordinate points 'tx' that represent sample points evenly distributed within the spatial-temporal domain.

        Examples:
            >>> import ppsci
            >>> timedomain = ppsci.geometry.TimeDomain(0, 1, 0.001)
            >>> geom = ppsci.geometry.Rectangle((0, 0), (1, 1))
            >>> time_geom = ppsci.geometry.TimeXGeometry(timedomain, geom)
            >>> ts = time_geom.uniform_points(1000)
            >>> print(ts.shape)
            (1000, 3)
        """
        if self.timedomain.time_step is not None:
            # exclude start time t0
            nt = int(np.ceil(self.timedomain.diam / self.timedomain.time_step))
            nx = int(np.ceil(n / nt))
        elif self.timedomain.timestamps is not None:
            # exclude start time t0
            nt = self.timedomain.num_timestamps - 1
            nx = int(np.ceil(n / nt))
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
            if self.timedomain.time_step is not None:
                t = np.linspace(
                    self.timedomain.t1,
                    self.timedomain.t0,
                    num=nt,
                    endpoint=boundary,
                    dtype=paddle.get_default_dtype(),
                )[:, None][::-1]
            else:
                t = self.timedomain.timestamps[1:]
        tx = []
        for ti in t:
            tx.append(
                np.hstack((np.full([nx, 1], ti, dtype=paddle.get_default_dtype()), x))
            )
        tx = np.vstack(tx)
        if len(tx) > n:
            tx = tx[:n]
        return tx

    def random_points(
        self, n: int, random: str = "pseudo", criteria: Optional[Callable] = None
    ) -> np.ndarray:
        """Generate random points on the spatial-temporal domain.

        Args:
            n (int): The total number of random points to generate.
            random (str): Specifies the way to generate random points, default is "pseudo" , which means that a pseudo-random number generator is used.
            criteria (Optional[Callable]): A method that filters on the generated random points, defualt is None.

        Returns:
            np.ndarray: A set of random spatial-temporal points.

        Examples:
            >>> import ppsci
            >>> timedomain = ppsci.geometry.TimeDomain(0, 1, 0.001)
            >>> geom = ppsci.geometry.Rectangle((0, 0), (1, 1))
            >>> time_geom = ppsci.geometry.TimeXGeometry(timedomain, geom)
            >>> ts = time_geom.random_points(1000)
            >>> print(ts.shape)
            (1000, 3)
        """
        if self.timedomain.time_step is None and self.timedomain.timestamps is None:
            raise ValueError("Either time_step or timestamps must be provided.")
        # time evenly and geometry random, if time_step if specified
        if self.timedomain.time_step is not None:
            nt = int(np.ceil(self.timedomain.diam / self.timedomain.time_step))
            t = np.linspace(
                self.timedomain.t1,
                self.timedomain.t0,
                num=nt,
                endpoint=False,
                dtype=paddle.get_default_dtype(),
            )[:, None][
                ::-1
            ]  # [nt, 1]
            # 1. sample nx points in static geometry with criteria
            nx = int(np.ceil(n / nt))
            _size, _ntry, _nsuc = 0, 0, 0
            x = np.empty(
                shape=(nx, self.geometry.ndim), dtype=paddle.get_default_dtype()
            )
            while _size < nx:
                _x = self.geometry.random_points(nx, random)
                if criteria is not None:
                    # fix arg 't' to None in criteria there
                    criteria_mask = criteria(
                        None, *np.split(_x, self.geometry.ndim, axis=1)
                    ).flatten()
                    _x = _x[criteria_mask]
                if len(_x) > nx - _size:
                    _x = _x[: nx - _size]
                x[_size : _size + len(_x)] = _x

                _size += len(_x)
                _ntry += 1
                if len(_x) > 0:
                    _nsuc += 1

                if _ntry >= 1000 and _nsuc == 0:
                    raise ValueError(
                        "Sample points failed, "
                        "please check correctness of geometry and given criteria."
                    )

            # 2. repeat spatial points along time
            tx = []
            for ti in t:
                tx.append(
                    np.hstack(
                        (np.full([nx, 1], ti, dtype=paddle.get_default_dtype()), x)
                    )
                )
            tx = np.vstack(tx)
            if len(tx) > n:
                tx = tx[:n]
            return tx
        elif self.timedomain.timestamps is not None:
            nt = self.timedomain.num_timestamps - 1
            t = self.timedomain.timestamps[1:]
            nx = int(np.ceil(n / nt))

            _size, _ntry, _nsuc = 0, 0, 0
            x = np.empty(
                shape=(nx, self.geometry.ndim), dtype=paddle.get_default_dtype()
            )
            while _size < nx:
                _x = self.geometry.random_points(nx, random)
                if criteria is not None:
                    # fix arg 't' to None in criteria there
                    criteria_mask = criteria(
                        None, *np.split(_x, self.geometry.ndim, axis=1)
                    ).flatten()
                    _x = _x[criteria_mask]
                if len(_x) > nx - _size:
                    _x = _x[: nx - _size]
                x[_size : _size + len(_x)] = _x

                _size += len(_x)
                _ntry += 1
                if len(_x) > 0:
                    _nsuc += 1

                if _ntry >= 1000 and _nsuc == 0:
                    raise ValueError(
                        "Sample interior points failed, "
                        "please check correctness of geometry and given criteria."
                    )

            tx = []
            for ti in t:
                tx.append(
                    np.hstack(
                        (np.full([nx, 1], ti, dtype=paddle.get_default_dtype()), x)
                    )
                )
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

    def uniform_boundary_points(
        self, n: int, criteria: Optional[Callable] = None
    ) -> np.ndarray:
        """Uniform boundary points on the spatial-temporal domain.
        Geometry surface area ~ bbox.
        Time surface area ~ diam.

        Args:
            n (int): The total number of boundary points on the spatial-temporal domain to be generated that are evenly distributed across geometry boundaries.
            criteria (Optional[Callable]): Used to filter the generated boundary points, only points that meet certain conditions are retained. Default is None.

        Returns:
            np.ndarray: A set of  point coordinates evenly distributed across geometry boundaries on the spatial-temporal domain.

        Examples:
            >>> import ppsci
            >>> timedomain = ppsci.geometry.TimeDomain(0, 1)
            >>> geom = ppsci.geometry.Rectangle((0, 0), (1, 1))
            >>> time_geom = ppsci.geometry.TimeXGeometry(timedomain, geom)
            >>> ts = time_geom.uniform_boundary_points(1000)
            >>> print(ts.shape)
            (1000, 3)
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

        _size, _ntry, _nsuc = 0, 0, 0
        x = np.empty(shape=(nx, self.geometry.ndim), dtype=paddle.get_default_dtype())
        while _size < nx:
            _x = self.geometry.uniform_boundary_points(nx)
            if criteria is not None:
                # fix arg 't' to None in criteria there
                criteria_mask = criteria(
                    None, *np.split(_x, self.geometry.ndim, axis=1)
                ).flatten()
                _x = _x[criteria_mask]
            if len(_x) > nx - _size:
                _x = _x[: nx - _size]
            x[_size : _size + len(_x)] = _x

            _size += len(_x)
            _ntry += 1
            if len(_x) > 0:
                _nsuc += 1

            if _ntry >= 1000 and _nsuc == 0:
                raise ValueError(
                    "Sample boundary points failed, "
                    "please check correctness of geometry and given criteria."
                )

        nx = len(x)
        t = np.linspace(
            self.timedomain.t1,
            self.timedomain.t0,
            num=nt,
            endpoint=False,
            dtype=paddle.get_default_dtype(),
        )[:, None][::-1]
        tx = []
        for ti in t:
            tx.append(
                np.hstack((np.full([nx, 1], ti, dtype=paddle.get_default_dtype()), x))
            )
        tx = np.vstack(tx)
        if len(tx) > n:
            tx = tx[:n]
        return tx

    def random_boundary_points(
        self, n: int, random: str = "pseudo", criteria: Optional[Callable] = None
    ) -> np.ndarray:
        """Random boundary points on the spatial-temporal domain.

        Args:
            n (int): The total number of spatial-temporal points generated on a given geometry boundary.
            random (str): Controls the way to generate random points. Default is "pseudo".
            criteria (Optional[Callable]): Used to filter the generated boundary points, only points that meet certain conditions are retained. Default is None.

        Returns:
            np.ndarray: A set of point coordinates randomly distributed across geometry boundaries on the spatial-temporal domain.

        Examples:
            >>> import ppsci
            >>> timedomain = ppsci.geometry.TimeDomain(0, 1, 0.001)
            >>> geom = ppsci.geometry.Rectangle((0, 0), (1, 1))
            >>> time_geom = ppsci.geometry.TimeXGeometry(timedomain, geom)
            >>> ts = time_geom.random_boundary_points(1000)
            >>> print(ts.shape)
            (1000, 3)
        """
        if self.timedomain.time_step is None and self.timedomain.timestamps is None:
            raise ValueError("Either time_step or timestamps must be provided.")
        if self.timedomain.time_step is not None:
            # exclude start time t0
            nt = int(np.ceil(self.timedomain.diam / self.timedomain.time_step))
            t = np.linspace(
                self.timedomain.t1,
                self.timedomain.t0,
                num=nt,
                endpoint=False,
                dtype=paddle.get_default_dtype(),
            )[:, None][::-1]
            nx = int(np.ceil(n / nt))

            if isinstance(self.geometry, mesh.Mesh):
                x, _n, a = self.geometry.random_boundary_points(nx, random=random)
            else:
                _size, _ntry, _nsuc = 0, 0, 0
                x = np.empty(
                    shape=(nx, self.geometry.ndim), dtype=paddle.get_default_dtype()
                )
                while _size < nx:
                    _x = self.geometry.random_boundary_points(nx, random)
                    if criteria is not None:
                        # fix arg 't' to None in criteria there
                        criteria_mask = criteria(
                            None, *np.split(_x, self.geometry.ndim, axis=1)
                        ).flatten()
                        _x = _x[criteria_mask]
                    if len(_x) > nx - _size:
                        _x = _x[: nx - _size]
                    x[_size : _size + len(_x)] = _x

                    _size += len(_x)
                    _ntry += 1
                    if len(_x) > 0:
                        _nsuc += 1

                    if _ntry >= 1000 and _nsuc == 0:
                        raise ValueError(
                            "Sample boundary points failed, "
                            "please check correctness of geometry and given criteria."
                        )

            t_x = []
            if isinstance(self.geometry, mesh.Mesh):
                t_normal = []
                t_area = []

            for ti in t:
                t_x.append(
                    np.hstack(
                        (np.full([nx, 1], ti, dtype=paddle.get_default_dtype()), x)
                    )
                )
                if isinstance(self.geometry, mesh.Mesh):
                    t_normal.append(
                        np.hstack(
                            (np.full([nx, 1], ti, dtype=paddle.get_default_dtype()), _n)
                        )
                    )
                    t_area.append(
                        np.hstack(
                            (np.full([nx, 1], ti, dtype=paddle.get_default_dtype()), a)
                        )
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
            nt = self.timedomain.num_timestamps - 1
            t = self.timedomain.timestamps[1:]
            nx = int(np.ceil(n / nt))

            if isinstance(self.geometry, mesh.Mesh):
                x, _n, a = self.geometry.random_boundary_points(nx, random=random)
            else:
                _size, _ntry, _nsuc = 0, 0, 0
                x = np.empty(
                    shape=(nx, self.geometry.ndim), dtype=paddle.get_default_dtype()
                )
                while _size < nx:
                    _x = self.geometry.random_boundary_points(nx, random)
                    if criteria is not None:
                        # fix arg 't' to None in criteria there
                        criteria_mask = criteria(
                            None, *np.split(_x, self.geometry.ndim, axis=1)
                        ).flatten()
                        _x = _x[criteria_mask]
                    if len(_x) > nx - _size:
                        _x = _x[: nx - _size]
                    x[_size : _size + len(_x)] = _x

                    _size += len(_x)
                    _ntry += 1
                    if len(_x) > 0:
                        _nsuc += 1

                    if _ntry >= 1000 and _nsuc == 0:
                        raise ValueError(
                            "Sample boundary points failed, "
                            "please check correctness of geometry and given criteria."
                        )

            t_x = []
            if isinstance(self.geometry, mesh.Mesh):
                t_normal = []
                t_area = []

            for ti in t:
                t_x.append(
                    np.hstack(
                        (np.full([nx, 1], ti, dtype=paddle.get_default_dtype()), x)
                    )
                )
                if isinstance(self.geometry, mesh.Mesh):
                    t_normal.append(
                        np.hstack(
                            (np.full([nx, 1], ti, dtype=paddle.get_default_dtype()), _n)
                        )
                    )
                    t_area.append(
                        np.hstack(
                            (np.full([nx, 1], ti, dtype=paddle.get_default_dtype()), a)
                        )
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

    def uniform_initial_points(self, n: int) -> np.ndarray:
        """Generate evenly distributed point coordinates on the spatial-temporal domain at the initial moment.

        Args:
            n (int): The total number of generated points.

        Returns:
           np.ndarray: A set of point coordinates evenly distributed on the spatial-temporal domain at the initial moment.

        Examples:
            >>> import ppsci
            >>> timedomain = ppsci.geometry.TimeDomain(0, 1)
            >>> geom = ppsci.geometry.Rectangle((0, 0), (1, 1))
            >>> time_geom = ppsci.geometry.TimeXGeometry(timedomain, geom)
            >>> ts = time_geom.uniform_initial_points(1000)
            >>> print(ts.shape)
            (1000, 3)
        """
        x = self.geometry.uniform_points(n, True)
        t = self.timedomain.t0
        if len(x) > n:
            x = x[:n]
        return np.hstack((np.full([n, 1], t, dtype=paddle.get_default_dtype()), x))

    def random_initial_points(self, n: int, random: str = "pseudo") -> np.ndarray:
        """Generate randomly distributed point coordinates on the spatial-temporal domain at the initial moment.

        Args:
            n (int): The total number of generated points.
            random (str): Controls the way to generate random points. Default is "pseudo".

        Returns:
            np.ndarray: A set of point coordinates randomly distributed on the spatial-temporal domain at the initial moment.

        Examples:
            >>> import ppsci
            >>> timedomain = ppsci.geometry.TimeDomain(0, 1)
            >>> geom = ppsci.geometry.Rectangle((0, 0), (1, 1))
            >>> time_geom = ppsci.geometry.TimeXGeometry(timedomain, geom)
            >>> ts = time_geom.random_initial_points(1000)
            >>> print(ts.shape)
            (1000, 3)
        """
        x = self.geometry.random_points(n, random=random)
        t = self.timedomain.t0
        return np.hstack((np.full([n, 1], t, dtype=paddle.get_default_dtype()), x))

    def periodic_point(
        self, x: Dict[str, np.ndarray], component: int
    ) -> Dict[str, np.ndarray]:
        """process given point coordinates to satisfy the periodic boundary conditions of the geometry.

        Args:
            x (Dict[str, np.ndarray]): Contains the coordinates and timestamps of the points. It represents the coordinates of the point to be processed.
            component (int): Specifies the components or dimensions of specific spatial coordinates that are periodically processed.

        Returns:
            Dict[str, np.ndarray] : contains the original timestamps and the coordinates of the spatial point after periodic processing.

        Examples:
        >>> import ppsci
        >>> timedomain = ppsci.geometry.TimeDomain(0, 1, 0.1)
        >>> geom = ppsci.geometry.Rectangle((0, 0), (1, 1))
        >>> time_geom = ppsci.geometry.TimeXGeometry(timedomain, geom)
        >>> ts = time_geom.sample_boundary(1000)
        >>> result = time_geom.periodic_point(ts, 0)
        >>> for k,v in result.items():
        ...     print(k, v.shape)
        t (1000, 1)
        x (1000, 1)
        y (1000, 1)
        normal_x (1000, 1)
        normal_y (1000, 1)
        """
        xp = self.geometry.periodic_point(x, component)
        txp = {"t": x["t"], **xp}
        return txp

    def sample_initial_interior(
        self,
        n: int,
        random: str = "pseudo",
        criteria: Optional[Callable] = None,
        evenly: bool = False,
        compute_sdf_derivatives: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Sample random points in the time-geometry and return those meet criteria.

        Args:
            n (int): The total number of interior points generated.
            random (str): The method used to specify the initial point of generation. Default is "pseudo".
            criteria (Optional[Callable]): Used to filter the generated interior points, only points that meet certain conditions are retained. Default is None.
            evenly (bool): Indicates whether the initial points are generated evenly. Default is False.
            compute_sdf_derivatives (bool): Indicates whether to calculate the derivative of signed distance function or not. Default is False.

        Returns:
            np.ndarray: Contains the coordinates of the initial internal point generated, as well as the potentially computed signed distance function and its derivative.

        Examples:
            >>> import ppsci
            >>> timedomain = ppsci.geometry.TimeDomain(0, 1)
            >>> geom = ppsci.geometry.Rectangle((0, 0), (1, 1))
            >>> time_geom = ppsci.geometry.TimeXGeometry(timedomain, geom)
            >>> ts = time_geom.sample_initial_interior(1000)
            >>> for k,v in ts.items():
            ...     print(k, v.shape)
            t (1000, 1)
            x (1000, 1)
            y (1000, 1)
            sdf (1000, 1)
        """
        x = np.empty(shape=(n, self.ndim), dtype=paddle.get_default_dtype())
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
                raise ValueError(
                    "Sample initial interior points failed, "
                    "please check correctness of geometry and given criteria."
                )

        # if sdf_func added, return x_dict and sdf_dict, else, only return the x_dict
        if hasattr(self.geometry, "sdf_func"):
            # compute sdf excluding time t
            sdf = -self.geometry.sdf_func(x[..., 1:])
            sdf_dict = misc.convert_to_dict(sdf, ("sdf",))
            sdf_derives_dict = {}
            if compute_sdf_derivatives:
                # compute sdf derivatives excluding time t
                sdf_derives = -self.geometry.sdf_derivatives(x[..., 1:])
                sdf_derives_dict = misc.convert_to_dict(
                    sdf_derives, tuple(f"sdf__{key}" for key in self.geometry.dim_keys)
                )
        else:
            sdf_dict = {}
            sdf_derives_dict = {}
        x_dict = misc.convert_to_dict(x, self.dim_keys)

        return {**x_dict, **sdf_dict, **sdf_derives_dict}

    def __str__(self) -> str:
        """Return the name of class"""
        return ", ".join(
            [
                self.__class__.__name__,
                f"ndim = {self.ndim}",
                f"bbox = (time){self.timedomain.bbox} x (space){self.geometry.bbox}",
                f"diam = (time){self.timedomain.diam} x (space){self.geometry.diam}",
                f"dim_keys = {self.dim_keys}",
            ]
        )
