"""
Defines base class for all geometries
"""

import copy
import numpy as np
import itertools
import sympy
from typing import Callable, Union, List

from ..utils.sympy import np_lambdify
from ..constants import diff_str
from .parameterization import Parameterization, Bounds
from .helper import (
    _concat_numpy_dict_list,
    _sympy_sdf_to_sdf,
    _sympy_criteria_to_criteria,
    _sympy_func_to_func,
)


def csg_curve_naming(index):
    return "PRIMITIVE_PARAM_" + str(index).zfill(5)


class Geometry:
    """
    Base class for all geometries
    """

    def __init__(
        self,
        curves,
        sdf,
        dims,
        bounds,
        parameterization=Parameterization(),
        interior_epsilon=1e-6,
    ):
        # store attributes
        self.curves = curves
        self.sdf = sdf
        self._dims = dims
        self.bounds = bounds
        self.parameterization = parameterization
        self.interior_epsilon = interior_epsilon  # to check if in domain or outside

    @property
    def dims(self):
        """
        Returns
        -------
        dims : List[srt]
            output can be ['x'], ['x','y'], or ['x','y','z']
        """

        return ["x", "y", "z"][: self._dims]

    def scale(
        self,
        x: Union[float, sympy.Basic],
        parameterization: Parameterization = Parameterization(),
    ):
        """
        Scales geometry.

        Parameters
        ----------
        x : Union[float, sympy.Basic]
            Scale factor. Can be a sympy expression if parameterizing.
        parameterization : Parameterization
            Parameterization if scale factor is parameterized.
        """

        # create scaled sdf function
        def _scale_sdf(sdf, dims, x):
            if isinstance(x, (float, int)):
                pass
            elif isinstance(x, sympy.Basic):
                x = _sympy_func_to_func(x)
            else:
                raise TypeError("Scaling by type " + str(type(x)) + "is not supported")

            def scale_sdf(invar, params, compute_sdf_derivatives=False):
                # compute scale if needed
                if isinstance(x, (float, int)):
                    computed_scale = x
                else:
                    computed_scale = x(params)

                # scale input to sdf function
                scaled_invar = {**invar}
                for key in dims:
                    scaled_invar[key] = scaled_invar[key] / computed_scale

                # compute sdf
                computed_sdf = sdf(scaled_invar, params, compute_sdf_derivatives)

                # scale output sdf values
                if isinstance(x, (float, int)):
                    computed_sdf["sdf"] *= x
                else:
                    computed_sdf["sdf"] *= x(params)
                return computed_sdf

            return scale_sdf

        new_sdf = _scale_sdf(self.sdf, self.dims, x)

        # add parameterization
        new_parameterization = self.parameterization.union(parameterization)

        # scale bounds
        new_bounds = self.bounds.scale(x, parameterization)

        # scale curves
        new_curves = [c.scale(x, parameterization) for c in self.curves]

        # return scaled geometry
        return Geometry(
            new_curves,
            new_sdf,
            len(self.dims),
            new_bounds,
            new_parameterization,
            interior_epsilon=self.interior_epsilon,
        )

    def translate(
        self,
        xyz: List[Union[float, sympy.Basic]],
        parameterization: Parameterization = Parameterization(),
    ):
        """
        Translates geometry.

        Parameters
        ----------
        xyz : List[Union[float, sympy.Basic]]
            Translation. Can be a sympy expression if parameterizing.
        parameterization : Parameterization
            Parameterization if translation is parameterized.
        """

        # create translated sdf function
        def _translate_sdf(sdf, dims, xyx):
            compiled_xyz = []
            for i, x in enumerate(xyz):
                if isinstance(x, (float, int)):
                    compiled_xyz.append(x)
                elif isinstance(x, sympy.Basic):
                    compiled_xyz.append(_sympy_func_to_func(x))
                else:
                    raise TypeError(
                        "Translate by type " + str(type(x)) + "is not supported"
                    )

            def translate_sdf(invar, params, compute_sdf_derivatives=False):
                # compute translation if needed
                computed_translation = []
                for x in compiled_xyz:
                    if isinstance(x, (float, int)):
                        computed_translation.append(x)
                    else:
                        computed_translation.append(x(params))

                # translate input to sdf function
                translated_invar = {**invar}
                for i, key in enumerate(dims):
                    translated_invar[key] = (
                        translated_invar[key] - computed_translation[i]
                    )

                # compute sdf
                computed_sdf = sdf(translated_invar, params, compute_sdf_derivatives)
                return computed_sdf

            return translate_sdf

        new_sdf = _translate_sdf(self.sdf, self.dims, xyz)

        # add parameterization
        new_parameterization = self.parameterization.union(parameterization)

        # translate bounds
        new_bounds = self.bounds.translate(xyz, parameterization)

        # translate curves
        new_curves = [c.translate(xyz, parameterization) for c in self.curves]

        # return translated geometry
        return Geometry(
            new_curves,
            new_sdf,
            len(self.dims),
            new_bounds,
            new_parameterization,
            interior_epsilon=self.interior_epsilon,
        )

    def rotate(
        self,
        angle: Union[float, sympy.Basic],
        axis: str = "z",
        center: Union[None, List[Union[float]]] = None,
        parameterization=Parameterization(),
    ):
        """
        Rotates geometry.

        Parameters
        ----------
        angle : Union[float, sympy.Basic]
            Angle of rotate in radians. Can be a sympy expression if parameterizing.
        axis : str
            Axis of rotation. Default is `"z"`.
        center : Union[None, List[Union[float, sympy.Basic]]] = None
            If given then center the rotation around this point.
        parameterization : Parameterization
            Parameterization if translation is parameterized.
        """

        # create rotated sdf function
        def _rotate_sdf(sdf, dims, angle, axis, center):
            if isinstance(angle, (float, int)):
                pass
            elif isinstance(angle, sympy.Basic):
                angle = _sympy_func_to_func(angle)
            else:
                raise TypeError("Scaling by type " + str(type(x)) + "is not supported")

            def rotate_sdf(invar, params, compute_sdf_derivatives=False):
                # compute translation if needed
                if isinstance(angle, (float, int)):
                    computed_angle = angle
                else:
                    computed_angle = angle(params)

                # rotate input to sdf function
                rotated_invar = {**invar}
                if center is not None:
                    for i, key in enumerate(dims):
                        rotated_invar[key] = rotated_invar[key] - center[i]
                _rotated_invar = {**rotated_invar}
                rotated_dims = [key for key in dims if key != axis]
                _rotated_invar[rotated_dims[0]] = (
                    np.cos(angle) * rotated_invar[rotated_dims[0]]
                    + np.sin(angle) * rotated_invar[rotated_dims[1]]
                )
                _rotated_invar[rotated_dims[1]] = (
                    -np.sin(angle) * rotated_invar[rotated_dims[0]]
                    + np.cos(angle) * rotated_invar[rotated_dims[1]]
                )
                if center is not None:
                    for i, key in enumerate(dims):
                        _rotated_invar[key] = _rotated_invar[key] + center[i]

                # compute sdf
                computed_sdf = sdf(_rotated_invar, params, compute_sdf_derivatives)
                return computed_sdf

            return rotate_sdf

        new_sdf = _rotate_sdf(self.sdf, self.dims, angle, axis, center)

        # add parameterization
        new_parameterization = self.parameterization.union(parameterization)

        # rotate bounds
        if center is not None:
            new_bounds = self.bounds.translate([-x for x in center])
            new_bounds = new_bounds.rotate(angle, axis, parameterization)
            new_bounds = new_bounds.translate(center)
        else:
            new_bounds = self.bounds.rotate(angle, axis, parameterization)

        # rotate curves
        new_curves = []
        for c in self.curves:
            if center is not None:
                new_c = c.translate([-x for x in center])
                new_c = new_c.rotate(angle, axis, parameterization)
                new_c = new_c.translate(center)
            else:
                new_c = c.rotate(angle, axis, parameterization)
            new_curves.append(new_c)

        # return rotated geometry
        return Geometry(
            new_curves,
            new_sdf,
            len(self.dims),
            new_bounds,
            new_parameterization,
            interior_epsilon=self.interior_epsilon,
        )

    def repeat(
        self,
        spacing: float,
        repeat_lower: List[int],
        repeat_higher: List[int],
        center: Union[None, List[Union[float]]] = None,
    ):
        """
        Finite Repetition of geometry.

        Parameters
        ----------
        spacing : float
            Spacing between each repetition.
        repeat_lower : List[int]
            How many repetitions going in negative direction.
        repeat_upper : List[int]
            How many repetitions going in positive direction.
        center : Union[None, List[Union[float, sympy.Basic]]] = None
            If given then center the rotation around this point.
        """

        # create repeated sdf function
        def _repeat_sdf(
            sdf, dims, spacing, repeat_lower, repeat_higher, center
        ):  # TODO make spacing, repeat_lower, and repeat_higher parameterizable
            def repeat_sdf(invar, params, compute_sdf_derivatives=False):
                # clamp position values
                clamped_invar = {**invar}
                if center is not None:
                    for i, key in enumerate(dims):
                        clamped_invar[key] = clamped_invar[key] - center[i]
                for d, rl, rh in zip(dims, repeat_lower, repeat_higher):
                    clamped_invar[d] = clamped_invar[d] - spacing * np.minimum(
                        np.maximum(np.around(clamped_invar[d] / spacing), rl), rh
                    )
                if center is not None:
                    for i, key in enumerate(dims):
                        clamped_invar[key] = clamped_invar[key] + center[i]

                # compute sdf
                computed_sdf = sdf(clamped_invar, params, compute_sdf_derivatives)
                return computed_sdf

            return repeat_sdf

        new_sdf = _repeat_sdf(
            self.sdf, self.dims, spacing, repeat_lower, repeat_higher, center
        )

        # repeat bounds and curves
        new_bounds = self.bounds.copy()
        new_curves = []
        for t in itertools.product(
            *[list(range(rl, rh + 1)) for rl, rh in zip(repeat_lower, repeat_higher)]
        ):
            new_bounds = new_bounds.union(
                self.bounds.translate([spacing * a for a in t])
            )
            new_curves += [c.translate([spacing * a for a in t]) for c in self.curves]

        # return repeated geometry
        return Geometry(
            new_curves,
            new_sdf,
            len(self.dims),
            new_bounds,
            self.parameterization.copy(),
            interior_epsilon=self.interior_epsilon,
        )

    def copy(self):
        return copy.deepcopy(self)

    def boundary_criteria(self, invar, criteria=None, params={}):
        # check if moving in or out of normal direction changes SDF
        invar_normal_plus = {**invar}
        invar_normal_minus = {**invar}
        for key in self.dims:
            invar_normal_plus[key] = (
                invar_normal_plus[key]
                + self.interior_epsilon * invar_normal_plus["normal_" + key]
            )
            invar_normal_minus[key] = (
                invar_normal_minus[key]
                - self.interior_epsilon * invar_normal_minus["normal_" + key]
            )
        sdf_normal_plus = self.sdf(
            invar_normal_plus, params, compute_sdf_derivatives=False
        )["sdf"]
        sdf_normal_minus = self.sdf(
            invar_normal_minus, params, compute_sdf_derivatives=False
        )["sdf"]
        on_boundary = np.greater_equal(0, sdf_normal_plus * sdf_normal_minus)

        # check if points satisfy the criteria function
        if criteria is not None:
            # convert sympy criteria if needed
            satify_criteria = criteria(invar, params)

            # update on_boundary
            on_boundary = np.logical_and(on_boundary, satify_criteria)

        return on_boundary

    def sample_boundary(
        self,
        nr_points: int,
        criteria: Union[sympy.Basic, None] = None,
        parameterization: Union[Parameterization, None] = None,
        quasirandom: bool = False,
        curve_index_filters=None
    ):
        """
        Samples the surface or perimeter of the geometry.

        Parameters
        ----------
        nr_points : int
            number of points to sample on boundary.
        criteria : Union[sympy.Basic, None]
            Only sample points that satisfy this criteria.
        parameterization : Union[Parameterization, None], optional
            If the geometry is parameterized then you can provide ranges
            for the parameters with this. By default the sampling will be
            done with the internal parameterization.
        quasirandom : bool
            If true then sample the points using the Halton sequences.
            Default is False.

        Returns
        -------
        points : Dict[str, np.ndarray]
            Dictionary contain a point cloud sampled uniformly.
            For example in 2D it would be
            ```
            points = {'x': np.ndarray (N, 1),
                      'y': np.ndarray (N, 1),
                      'normal_x': np.ndarray (N, 1),
                      'normal_y': np.ndarray (N, 1),
                      'area': np.ndarray (N, 1)}
            ```
            The `area` value can be used for Monte Carlo integration
            like the following,
            `total_area = np.sum(points['area'])`
        """

        # compile criteria from sympy if needed
        if criteria is not None:
            if isinstance(criteria, sympy.Basic):
                criteria = _sympy_criteria_to_criteria(criteria)
            elif isinstance(criteria, Callable):
                pass
            else:
                raise TypeError(
                    "criteria type is not supported: " + str(type(criteria))
                )

        # use internal parameterization if not given
        if parameterization is None:
            parameterization = self.parameterization
        elif isinstance(parameterization, dict):
            parameterization = Parameterization(parameterization)

        # create boundary criteria closure
        def _boundary_criteria(criteria):
            def boundary_criteria(invar, params):
                return self.boundary_criteria(invar, criteria=criteria, params=params)

            return boundary_criteria

        closed_boundary_criteria = _boundary_criteria(criteria)

        # compute required points on each curve
        curve_areas = np.array(
            [
                curve.approx_area(parameterization, criteria=closed_boundary_criteria)
                for curve in self.curves
            ]
        )

        assert np.sum(curve_areas) > 0, "Geometry has no surface"
        curve_probabilities = curve_areas / np.linalg.norm(curve_areas, ord=1)
        curve_index = np.arange(len(self.curves))
        points_per_curve = np.random.choice(
            curve_index, nr_points, p=curve_probabilities
        )
        points_per_curve, _ = np.histogram(
            points_per_curve, np.arange(len(self.curves) + 1) - 0.5
        )

        if curve_index_filters is not None:
            filtered_curves = []
            for c in curve_index_filters:
                filtered_curves.append(self.curves[c])

            self.curves = filtered_curves

        # continually sample each curve until reached desired number of points
        list_invar = []
        list_params = []
        for n, a, curve in zip(points_per_curve, curve_areas, self.curves):
            if n > 0:
                i, p = curve.sample(
                    n,
                    criteria=closed_boundary_criteria,
                    parameterization=parameterization,
                )
                i["area"] = np.full_like(i["area"], a / n)
                list_invar.append(i)
                list_params.append(p)
        invar = _concat_numpy_dict_list(list_invar)
        params = _concat_numpy_dict_list(list_params)
        invar.update(params)
        return (list_invar, invar)

    def sample_interior(
        self,
        nr_points: int,
        bounds: Union[Bounds, None] = None,
        criteria: Union[sympy.Basic, None] = None,
        parameterization: Union[Parameterization, None] = None,
        compute_sdf_derivatives: bool = False,
        quasirandom: bool = False,
    ):
        """
        Samples the interior of the geometry.

        Parameters
        ----------
        nr_points : int
            number of points to sample.
        bounds : Union[Bounds, None]
            Bounds to sample points from. For example,
            `bounds = Bounds({Parameter('x'): (0, 1), Parameter('y'): (0, 1)})`.
            By default the internal bounds will be used.
        criteria : Union[sympy.Basic, None]
            Only sample points that satisfy this criteria.
        parameterization: Union[Parameterization, None]
            If the geometry is parameterized then you can provide ranges
            for the parameters with this.
        compute_sdf_derivatives : bool
            Compute sdf derivatives if true.
        quasirandom : bool
            If true then sample the points using the Halton sequences.
            Default is False.

        Returns
        -------
        points : Dict[str, np.ndarray]
            Dictionary contain a point cloud sampled uniformly.
            For example in 2D it would be
            ```
            points = {'x': np.ndarray (N, 1),
                      'y': np.ndarray (N, 1),
                      'sdf': np.ndarray (N, 1),
                      'area': np.ndarray (N, 1)}
            ```
            The `area` value can be used for Monte Carlo integration
            like the following,
            `total_area = np.sum(points['area'])`
        """

        # compile criteria from sympy if needed
        if criteria is not None:
            if isinstance(criteria, sympy.Basic):
                criteria = _sympy_criteria_to_criteria(criteria)
            elif isinstance(criteria, Callable):
                pass
            else:
                raise TypeError(
                    "criteria type is not supported: " + str(type(criteria))
                )

        # use internal bounds if not given
        if bounds is None:
            bounds = self.bounds
        elif isinstance(bounds, dict):
            bounds = Bounds(bounds)

        # use internal parameterization if not given
        if parameterization is None:
            parameterization = self.parameterization
        elif isinstance(parameterization, dict):
            parameterization = Parameterization(parameterization)

        # continually sample until reached desired number of points
        invar = {}
        params = {}
        total_tried = 0
        nr_try = 0
        while True:
            # sample invar and params
            local_invar = bounds.sample(nr_points, parameterization, quasirandom)
            local_params = parameterization.sample(nr_points, quasirandom)

            # evaluate SDF function on points
            local_invar.update(
                self.sdf(
                    local_invar,
                    local_params,
                    compute_sdf_derivatives=compute_sdf_derivatives,
                )
            )

            # remove points outside of domain
            criteria_index = np.greater(local_invar["sdf"], 0)
            if criteria is not None:
                criteria_index = np.logical_and(
                    criteria_index, criteria(local_invar, local_params)
                )
            for key in local_invar.keys():
                local_invar[key] = local_invar[key][criteria_index[:, 0], :]
            for key in local_params.keys():
                local_params[key] = local_params[key][criteria_index[:, 0], :]

            # add sampled points to list
            for key in local_invar.keys():
                if key not in invar.keys():  # TODO this can be condensed
                    invar[key] = local_invar[key]
                else:
                    invar[key] = np.concatenate([invar[key], local_invar[key]], axis=0)
            for key in local_params.keys():
                if key not in params.keys():  # TODO this can be condensed
                    params[key] = local_params[key]
                else:
                    params[key] = np.concatenate(
                        [params[key], local_params[key]], axis=0
                    )

            # check if finished
            total_sampled = next(iter(invar.values())).shape[0]
            total_tried += nr_points
            nr_try += 1
            if total_sampled >= nr_points:
                for key, value in invar.items():
                    invar[key] = value[:nr_points]
                for key, value in params.items():
                    params[key] = value[:nr_points]
                break

            # report error if could not sample
            if nr_try > 100 and total_sampled < 1:
                raise RuntimeError(
                    "Could not sample interior of geometry. Check to make sure non-zero volume"
                )

        # compute area value for monte carlo integration
        volume = (total_sampled / total_tried) * bounds.volume(parameterization)
        invar["area"] = np.full_like(next(iter(invar.values())), volume / nr_points)

        # add params to invar
        invar.update(params)
        return invar

    @staticmethod
    def _convert_criteria(criteria):
        return criteria

    def __add__(self, other):
        def _add_sdf(sdf_1, sdf_2, dims):
            def add_sdf(invar, params, compute_sdf_derivatives=False):
                computed_sdf_1 = sdf_1(invar, params, compute_sdf_derivatives)
                computed_sdf_2 = sdf_2(invar, params, compute_sdf_derivatives)
                computed_sdf = {}
                computed_sdf["sdf"] = np.maximum(
                    computed_sdf_1["sdf"], computed_sdf_2["sdf"]
                )
                if compute_sdf_derivatives:
                    for d in dims:
                        computed_sdf["sdf" + diff_str + d] = np.where(
                            computed_sdf_1["sdf"] > computed_sdf_2["sdf"],
                            computed_sdf_1["sdf" + diff_str + d],
                            computed_sdf_2["sdf" + diff_str + d],
                        )
                return computed_sdf

            return add_sdf

        new_sdf = _add_sdf(self.sdf, other.sdf, self.dims)
        new_parameterization = self.parameterization.union(other.parameterization)
        new_bounds = self.bounds.union(other.bounds)

        return Geometry(
            self.curves + other.curves,
            new_sdf,
            len(self.dims),
            new_bounds,
            new_parameterization,
            interior_epsilon=self.interior_epsilon,
        )

    def __sub__(self, other):
        def _sub_sdf(sdf_1, sdf_2, dims):
            def sub_sdf(invar, params, compute_sdf_derivatives=False):
                computed_sdf_1 = sdf_1(invar, params, compute_sdf_derivatives)
                computed_sdf_2 = sdf_2(invar, params, compute_sdf_derivatives)
                computed_sdf = {}
                computed_sdf["sdf"] = np.minimum(
                    computed_sdf_1["sdf"], -computed_sdf_2["sdf"]
                )
                if compute_sdf_derivatives:
                    for d in dims:
                        computed_sdf["sdf" + diff_str + d] = np.where(
                            computed_sdf_1["sdf"] < -computed_sdf_2["sdf"],
                            computed_sdf_1["sdf" + diff_str + d],
                            -computed_sdf_2["sdf" + diff_str + d],
                        )
                return computed_sdf

            return sub_sdf

        new_sdf = _sub_sdf(self.sdf, other.sdf, self.dims)
        new_parameterization = self.parameterization.union(other.parameterization)
        new_bounds = self.bounds.union(other.bounds)
        new_curves = self.curves + [c.invert_normal() for c in other.curves]

        return Geometry(
            new_curves,
            new_sdf,
            len(self.dims),
            new_bounds,
            new_parameterization,
            interior_epsilon=self.interior_epsilon,
        )

    def __invert__(self):
        def _invert_sdf(sdf, dims):
            def invert_sdf(invar, params, compute_sdf_derivatives=False):
                computed_sdf = sdf(invar, params, compute_sdf_derivatives)
                computed_sdf["sdf"] = -computed_sdf["sdf"]
                if compute_sdf_derivatives:
                    for d in dims:
                        computed_sdf["sdf" + diff_str + d] = -computed_sdf[
                            "sdf" + diff_str + d
                        ]
                return computed_sdf

            return invert_sdf

        new_sdf = _invert_sdf(self.sdf, self.dims)
        new_parameterization = self.parameterization.copy()
        new_bounds = self.bounds.copy()
        new_curves = [c.invert_normal() for c in self.curves]

        return Geometry(
            new_curves,
            new_sdf,
            len(self.dims),
            new_bounds,
            new_parameterization,
            interior_epsilon=self.interior_epsilon,
        )

    def __and__(self, other):
        def _and_sdf(sdf_1, sdf_2, dims):
            def and_sdf(invar, params, compute_sdf_derivatives=False):
                computed_sdf_1 = sdf_1(invar, params, compute_sdf_derivatives)
                computed_sdf_2 = sdf_2(invar, params, compute_sdf_derivatives)
                computed_sdf = {}
                computed_sdf["sdf"] = np.minimum(
                    computed_sdf_1["sdf"], computed_sdf_2["sdf"]
                )
                if compute_sdf_derivatives:
                    for d in dims:
                        computed_sdf["sdf" + diff_str + d] = np.where(
                            computed_sdf_1["sdf"] < computed_sdf_2["sdf"],
                            computed_sdf_1["sdf" + diff_str + d],
                            computed_sdf_2["sdf" + diff_str + d],
                        )
                return computed_sdf

            return and_sdf

        new_sdf = _and_sdf(self.sdf, other.sdf, self.dims)
        new_parameterization = self.parameterization.union(other.parameterization)
        new_bounds = self.bounds.union(other.bounds)
        new_curves = self.curves + other.curves

        return Geometry(
            new_curves,
            new_sdf,
            len(self.dims),
            new_bounds,
            new_parameterization,
            interior_epsilon=self.interior_epsilon,
        )
