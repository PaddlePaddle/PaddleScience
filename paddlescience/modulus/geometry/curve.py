"""
Defines different Curve objects
"""

import types
import numpy as np
import sympy
import symengine
from .parameterization import Parameterization, Parameter
from .helper import _sympy_func_to_func


class Curve:
    """A Curve object that keeps track of the surface/perimeter of a geometry.
    The curve object also contains normals and area/length of curve.
    """

    def __init__(self, sample, dims, parameterization=Parameterization()):
        # store attributes
        self._sample = sample
        self._dims = dims
        self.parameterization = parameterization

    def sample(
        self, nr_points, criteria=None, parameterization=None, quasirandom=False
    ):
        # use internal parameterization if not given
        if parameterization is None:
            parameterization = self.parameterization

        # continually sample points throwing out points that don't satisfy criteria
        invar = {
            key: np.empty((0, 1))
            for key in self.dims + ["normal_" + x for x in self.dims] + ["area"]
        }
        params = {key: np.empty((0, 1)) for key in parameterization.parameters}
        total_sampled = 0
        total_tried = 0
        nr_try = 0
        while True:
            # sample curve
            local_invar, local_params = self._sample(
                nr_points, parameterization, quasirandom
            )

            # compute given criteria and remove points
            if criteria is not None:
                computed_criteria = criteria(local_invar, local_params)
                local_invar = {
                    key: value[computed_criteria[:, 0], :]
                    for key, value in local_invar.items()
                }
                local_params = {
                    key: value[computed_criteria[:, 0], :]
                    for key, value in local_params.items()
                }

            # store invar
            for key in local_invar.keys():
                invar[key] = np.concatenate([invar[key], local_invar[key]], axis=0)

            # store params
            for key in local_params.keys():
                params[key] = np.concatenate([params[key], local_params[key]], axis=0)

            # keep track of sampling
            total_sampled = next(iter(invar.values())).shape[0]
            total_tried += nr_points
            nr_try += 1

            # break when finished sampling
            if total_sampled >= nr_points:
                for key, value in invar.items():
                    invar[key] = value[:nr_points]
                for key, value in params.items():
                    params[key] = value[:nr_points]
                break

            # check if couldn't sample
            if nr_try > 1000 and total_sampled < 1:
                raise Exception("Unable to sample curve")

        return invar, params

    @property
    def dims(self):
        """
        Returns
        -------
        dims : list of strings
          output can be ['x'], ['x','y'], or ['x','y','z']
        """
        return ["x", "y", "z"][: self._dims]

    def approx_area(
        self,
        parameterization=Parameterization(),
        criteria=None,
        approx_nr=10000,
        quasirandom=False,
    ):
        """
        Parameters
        ----------
        parameterization: dict with of Parameters and their ranges
          If the curve is parameterized then you can provide ranges
          for the parameters with this.
        criteria : None, SymPy boolean exprs
          Calculate area discarding regions that don't satisfy
          this criteria.
        approx_nr : int
          Area might be difficult to compute if parameterized. In
          this case the area is approximated by sampleing `area`,
          `approx_nr` number of times. This amounts to monte carlo
          integration.

        Returns
        -------
        area : float
          area of curve
        """

        s, p = self._sample(
            nr_points=approx_nr,
            parameterization=parameterization,
            quasirandom=quasirandom,
        )
        computed_criteria = criteria(s, p)
        total_area = np.sum(s["area"][computed_criteria[:, 0], :])
        return total_area

    def scale(self, x, parameterization=Parameterization()):
        """
        scale curve

        Parameters
        ----------
        x : float, SymPy Symbol/Exprs
          scale factor.
        """

        def _sample(internal_sample, dims, x):
            if isinstance(x, (float, int)):
                pass
            elif isinstance(x, sympy.Basic):
                x = _sympy_func_to_func(x)
            else:
                raise TypeError("Scaling by type " + str(type(x)) + "is not supported")

            def sample(
                nr_points, parameterization=Parameterization(), quasirandom=False
            ):
                # sample points
                invar, params = internal_sample(
                    nr_points, parameterization, quasirandom
                )

                # compute scale if needed
                if isinstance(x, (float, int)):
                    computed_scale = x
                else:
                    computed_scale = s(params)

                # scale invar
                for d in dims:
                    invar[d] *= x
                invar["area"] *= x ** (len(dims) - 1)

                return invar, params

            return sample

        return Curve(
            _sample(self._sample, self.dims, x),
            len(self.dims),
            self.parameterization.union(parameterization),
        )

    def translate(self, xyz, parameterization=Parameterization()):
        """
        translate curve

        Parameters
        ----------
        xyz : tuple of floats, ints, SymPy Symbol/Exprs
          translate curve by these values.
        """

        def _sample(internal_sample, dims, xyz):
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

            def sample(
                nr_points, parameterization=Parameterization(), quasirandom=False
            ):
                # sample points
                invar, params = internal_sample(
                    nr_points, parameterization, quasirandom
                )

                # compute translation if needed
                computed_translation = []
                for x in compiled_xyz:
                    if isinstance(x, (float, int)):
                        computed_translation.append(x)
                    else:
                        computed_translation.append(x(params))

                # translate invar
                for d, x in zip(dims, computed_translation):
                    invar[d] += x
                return invar, params

            return sample

        return Curve(
            _sample(self._sample, self.dims, xyz),
            len(self.dims),
            self.parameterization.union(parameterization),
        )

    def rotate(self, angle, axis, parameterization=Parameterization()):
        """
        rotate curve

        Parameters
        ----------
        x : float, SymPy Symbol/Exprs
          scale factor.
        """

        def _sample(internal_sample, dims, angle, axis):
            if isinstance(angle, (float, int)):
                pass
            elif isinstance(angle, sympy.Basic):
                angle = _sympy_func_to_func(angle)
            else:
                raise TypeError(
                    "Scaling by type " + str(type(angle)) + "is not supported"
                )

            def sample(
                nr_points, parameterization=Parameterization(), quasirandom=False
            ):
                # sample points
                invar, params = internal_sample(
                    nr_points, parameterization, quasirandom
                )

                # compute translation if needed
                if isinstance(angle, (float, int)):
                    computed_angle = angle
                else:
                    computed_angle = angle(params)

                # angle invar
                rotated_invar = {**invar}
                rotated_dims = [key for key in self.dims if key != axis]
                rotated_invar[rotated_dims[0]] = (
                    np.cos(computed_angle) * invar[rotated_dims[0]]
                    - np.sin(computed_angle) * invar[rotated_dims[1]]
                )
                rotated_invar["normal_" + rotated_dims[0]] = (
                    np.cos(computed_angle) * invar["normal_" + rotated_dims[0]]
                    - np.sin(computed_angle) * invar["normal_" + rotated_dims[1]]
                )
                rotated_invar[rotated_dims[1]] = (
                    np.sin(computed_angle) * invar[rotated_dims[0]]
                    + np.cos(computed_angle) * invar[rotated_dims[1]]
                )
                rotated_invar["normal_" + rotated_dims[1]] = (
                    np.sin(computed_angle) * invar["normal_" + rotated_dims[0]]
                    + np.cos(computed_angle) * invar["normal_" + rotated_dims[1]]
                )

                return rotated_invar, params

            return sample

        return Curve(
            _sample(self._sample, self.dims, angle, axis),
            len(self.dims),
            self.parameterization.union(parameterization),
        )

    def invert_normal(self):
        def _sample(internal_sample, dims):
            def sample(
                nr_points, parameterization=Parameterization(), quasirandom=False
            ):
                s, p = internal_sample(nr_points, parameterization, quasirandom)
                for d in dims:
                    s["normal_" + d] = -s["normal_" + d]
                return s, p

            return sample

        return Curve(
            _sample(self._sample, self.dims), len(self.dims), self.parameterization
        )


class SympyCurve(Curve):
    """Curve defined by sympy functions

    Parameters
    ----------
    functions : dictionary of SymPy Exprs
        Parameterized curve in 1, 2 or 3 dimensions. For example, a
        circle might have::

            functions = {'x': cos(theta),
            \t'y': sin(theta),
            \t'normal_x': cos(theta),
            \t'normal_y': sin(theta)}

        TODO: refactor to remove normals.
    ranges : dictionary of Sympy Symbols and ranges
        This gives the ranges for the parameters in the parameterized
        curve. For example, a circle might have `ranges = {theta: (0, 2*pi)}`.
    area : float, int, SymPy Exprs
        The surface area/perimeter of the curve.
    criteria : SymPy Boolean Function
        If this boolean expression is false then we do not
        sample their on curve. This can be used to enforce
        uniform sample probability.
    """

    def __init__(self, functions, parameterization, area, criteria=None):

        # lambdify functions
        lambdify_functions = {}
        for key, func in functions.items():
            try:
                func = float(func)
            except:
                pass
            if isinstance(func, float):
                lambdify_functions[key] = float(func)
            elif isinstance(func, (sympy.Basic, symengine.Basic, Parameter)):
                lambdify_functions[key] = _sympy_func_to_func(func)
            else:
                raise TypeError("function type not supported: " + str(type(func)))

        # lambdify area function
        try:
            area = float(area)
        except:
            pass
        if isinstance(area, float):
            area_fn = float(area)
        elif isinstance(area, (sympy.Basic, symengine.Basic, Parameter)):
            area_fn = _sympy_func_to_func(area)
        else:
            raise TypeError("area type not supported: " + str(type(area)))
        lambdify_functions["area"] = area_fn

        # lambdify criteria function
        if criteria is not None:
            criteria = _sympy_func_to_func(criteria)

        # create closure for sample function
        def _sample(lambdify_functions, criteria, internal_parameterization):
            def sample(
                nr_points, parameterization=Parameterization(), quasirandom=False
            ):
                #import pdb
                #pdb.set_trace()
                # use internal parameterization if not given
                i_parameterization = internal_parameterization.copy()
                for key, value in parameterization.param_ranges.items():
                    i_parameterization.param_ranges[key] = value

                # continually sample points throwing out points that don't satisfy criteria
                invar = {
                    str(key): np.empty((0, 1)) for key in lambdify_functions.keys()
                }
                params = {
                    str(key): np.empty((0, 1))
                    for key in parameterization.param_ranges.keys()
                }
                total_sampled = 0
                total_tried = 0
                nr_try = 0
                while True:
                    # sample parameter ranges
                    local_params = i_parameterization.sample(nr_points, quasirandom)

                    # compute curve points from functions
                    local_invar = {}
                    for key, func in lambdify_functions.items():
                        if isinstance(func, (float, int)):
                            local_invar[key] = np.full_like(
                                next(iter(local_params.values())), func
                            )
                        else:
                            local_invar[key] = func(local_params)
                    local_invar["area"] /= next(iter(local_params.values())).shape[0]

                    # remove points that don't satisfy curve criteria if needed
                    if criteria is not None:
                        # compute curve criteria
                        computed_criteria = criteria(local_params).astype(bool)

                        # remove elements points based on curve criteria
                        local_invar = {
                            key: value[computed_criteria[:, 0], :]
                            for key, value in local_invar.items()
                        }
                        local_params = {
                            key: value[computed_criteria[:, 0], :]
                            for key, value in local_params.items()
                        }

                    # only store external parameters
                    for key in list(local_params.keys()):
                        if key not in parameterization.parameters:
                            local_params.pop(key)

                    # store invar
                    for key in local_invar.keys():
                        invar[key] = np.concatenate(
                            [invar[key], local_invar[key]], axis=0
                        )

                    # store params
                    for key in local_params.keys():
                        params[key] = np.concatenate(
                            [params[key], local_params[key]], axis=0
                        )

                    # keep track of sampling
                    total_sampled = next(iter(invar.values())).shape[0]
                    total_tried += next(iter(local_invar.values())).shape[0]
                    nr_try += 1

                    # break when finished sampling
                    if total_sampled >= nr_points:
                        for key, value in invar.items():
                            invar[key] = value[:nr_points]
                        for key, value in params.items():
                            params[key] = value[:nr_points]
                        break

                    # check if couldn't sample
                    if nr_try > 10000 and total_sampled < 1:
                        raise Exception("Unable to sample curve")

                return invar, params

            return sample

        # initialize curve
        Curve.__init__(
            self,
            _sample(lambdify_functions, criteria, parameterization),
            len(functions) // 2,
            parameterization=parameterization,
        )
