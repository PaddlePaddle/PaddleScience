import itertools
import numpy as np
from typing import Dict, List, Union, Tuple, Callable
import sympy
from typing import Callable
from chaospy.distributions.sampler.sequences.primes import create_primes
from chaospy.distributions.sampler.sequences.van_der_corput import (
    create_van_der_corput_samples as create_samples,
)

from paddlescience.modulus.utils.sympy import np_lambdify


class Parameter(sympy.Symbol):
    """A Symbolic object used to parameterize geometries.

    Currently this only overloads the Sympy Symbol class however
    capabilities may be expanded in the future.

    Parameters
    ----------
    name : str
        Name given to parameter.
    """

    def __new__(cls, name: str):
        obj = sympy.Symbol.__new__(cls, name)
        return obj


class Parameterization:
    """A object used to store parameterization information
    about geometries.

    Parameters
    ----------
    param_ranges : Dict[Parameter, Union[float, Tuple[float, float], np.ndarray (N, 1)]
        Dictionary of Parameters and their ranges. The ranges can be one of the following
        types,
        :obj: Float will sample the parameter equal to this value.
        :obj: Tuple of two float as the bounding range to sample parameter from.
        :obj: `np.ndarray` as a discrete list of possible values for the parameter.
    """

    def __init__(
        self,
        param_ranges: Dict[
            Parameter, Union[float, Tuple[float, float], np.ndarray]
        ] = {},
    ):
        # store param ranges
        self.param_ranges = param_ranges

    @property
    def parameters(self):
        return [str(x) for x in self.param_ranges.keys()]

    def sample(self, nr_points: int, quasirandom: bool = False):
        """Sample parameterization values.

        Parameters
        ----------
        nr_points : int
            Number of points sampled from parameterization.
        quasirandom : bool
            If true then sample the points using Halton sequences.
            Default is False.
        """

        return {
            str(key): value
            for key, value in _sample_ranges(
                nr_points, self.param_ranges, quasirandom
            ).items()
        }

    def union(self, other):
        new_param_ranges = self.param_ranges.copy()
        for key, value in other.param_ranges.items():
            new_param_ranges[key] = value
        return Parameterization(new_param_ranges)

    @classmethod
    def combine(cls, p1, p2):
        assert len(set(p1.parameters).intersection(set(p2.parameters))) == 0, (
            "Combining parameterizations when they have overlapping parameters: p1 "
            + str(p1)
            + ", p2 "
            + str(p2)
        )
        new_param_ranges = p1.param_ranges.copy()
        new_param_ranges.update(p2.param_ranges.copy())
        return cls(new_param_ranges)

    def copy(self):
        return Parameterization(self.param_ranges.copy())

    def __str__(self):
        return str(self.param_ranges)


class Bounds:
    """A object used to store bounds for geometries.

    Parameters
    ----------
    bound_ranges : Dict[Parameter, Tuple[Union[float, sympy.Basic], Union[float, sympy.Basic]]
        Dictionary of Parameters with names `"x"`, `"y"`, or `"z"`. The value given for each of these is
        a tuple of the lower and upper bound. Sympy expressions can be used to define these upper and lower
        bounds.
    parameterization : Parameterization
        A Parameterization object used when the upper and lower bounds are parameterized.
    """

    def __init__(
        self,
        bound_ranges: Dict[
            Parameter, Tuple[Union[float, sympy.Basic], Union[float, sympy.Basic]]
        ],
        parameterization: Parameterization = Parameterization(),
    ):
        # store internal parameterization
        self.parameterization = parameterization

        # store bounds
        self.bound_ranges = bound_ranges

    @property
    def dims(self):
        """
        Returns
        -------
        dims : list of strings
          output can be ['x'], ['x','y'], or ['x','y','z']
        """

        return [str(x) for x in self.bound_ranges.keys()]

    def sample(
        self,
        nr_points: int,
        parameterization: Union[None, Parameterization] = None,
        quasirandom: bool = False,
    ):
        """Sample points in Bounds.

        Parameters
        ----------
        nr_points : int
            Number of points sampled from parameterization.
        parameterization : Parameterization
            Given if sampling bounds with different parameterization then the internal one stored in Bounds. Default is to not use this.
        quasirandom : bool
            If true then sample the points using Halton sequences.
            Default is False.
        """

        if parameterization is not None:
            parameterization = self.parameterization
        computed_bound_ranges = self._compute_bounds(parameterization)
        return {
            str(key): value
            for key, value in _sample_ranges(
                nr_points, computed_bound_ranges, quasirandom
            ).items()
        }

    def volume(self, parameterization: Union[None, Parameterization] = None):
        """Compute volume of bounds.

        Parameters
        ----------
        parameterization : Parameterization
            Given if sampling bounds with different parameterization then the internal one stored in Bounds. Default is to not use this.
        """

        # compute bounds from parameterization
        computed_bound_ranges = self._compute_bounds(parameterization)
        return np.prod(
            [value[1] - value[0] for value in computed_bound_ranges.values()]
        )

    def union(self, other):
        new_parameterization = self.parameterization.union(other.parameterization)
        new_bound_ranges = {}
        for (key, (lower_1, upper_1)), (lower_2, upper_2) in zip(
            self.bound_ranges.items(), other.bound_ranges.values()
        ):
            # compute new lower bound
            if isinstance(lower_1, sympy.Basic) or isinstance(lower_2, sympy.Basic):
                new_lower = sympy.Min(lower_1, lower_2)
            elif isinstance(lower_1, (float, int)):
                new_lower = min(lower_1, lower_2)

            # compute new upper bound
            if isinstance(upper_1, sympy.Basic) or isinstance(upper_2, sympy.Basic):
                new_upper = sympy.Max(upper_1, upper_2)
            elif isinstance(upper_1, (float, int)):
                new_upper = max(upper_1, upper_2)

            # add to list of bound ranges
            new_bound_ranges[key] = (new_lower, new_upper)
        return Bounds(new_bound_ranges, new_parameterization)

    def intersection(self, other):
        new_parameterization = self.parameterization.union(other.parameterization)
        new_bound_ranges = {}
        for (key, (lower_1, upper_1)), (lower_2, upper_2) in zip(
            self.bound_ranges.items(), other.bound_ranges.values()
        ):
            # compute new lower bound
            if isinstance(lower_1, sympy.Basic) or isinstance(lower_2, sympy.Basic):
                new_lower = sympy.Max(lower_1, lower_2)
            elif isinstance(lower_1, (float, int)):
                new_lower = max(lower_1, lower_2)

            # compute new upper bound
            if isinstance(upper_1, sympy.Basic) or isinstance(upper_2, sympy.Basic):
                new_upper = sympy.Min(upper_1, upper_2)
            elif isinstance(upper_1, (float, int)):
                new_upper = min(upper_1, upper_2)

            # add to list of bound ranges
            new_bound_ranges[key] = (new_lower, new_upper)
        return Bounds(new_bound_ranges, new_parameterization)

    def scale(self, x, parameterization=Parameterization()):
        scaled_bound_ranges = {
            key: (lower * x, upper * x)
            for key, (lower, upper) in self.bound_ranges.items()
        }
        return Bounds(
            scaled_bound_ranges, self.parameterization.union(parameterization)
        )

    def translate(self, xyz, parameterization=Parameterization()):
        translated_bound_ranges = {
            key: (lower + x, upper + x)
            for (key, (lower, upper)), x in zip(self.bound_ranges.items(), xyz)
        }
        return Bounds(
            translated_bound_ranges, self.parameterization.union(parameterization)
        )

    def rotate(self, angle, axis, parameterization=Parameterization()):
        # rotate bounding box
        rotated_dims = [Parameter(key) for key in self.dims if key != axis]
        bounding_points = itertools.product(
            *[value for value in self.bound_ranges.values()]
        )
        rotated_bounding_points = []
        for p in bounding_points:
            p = {Parameter(key): value for key, value in zip(self.dims, p)}
            rotated_p = {**p}
            rotated_p[rotated_dims[0]] = (
                sympy.cos(angle) * p[rotated_dims[0]]
                - sympy.sin(angle) * p[rotated_dims[1]]
            )
            rotated_p[rotated_dims[1]] = (
                sympy.sin(angle) * p[rotated_dims[0]]
                + sympy.cos(angle) * p[rotated_dims[1]]
            )
            rotated_bounding_points.append(rotated_p)

        # find new bounds from rotated bounds
        rotated_bound_ranges = {**self.bound_ranges}
        for d in self.dims:
            # find upper and lower bound
            a = [p[Parameter(d)] for p in rotated_bounding_points]
            lower = sympy.Min(*a)
            upper = sympy.Max(*a)
            if lower.is_number:
                lower = float(lower)
            if upper.is_number:
                upper = float(upper)
            rotated_bound_ranges[Parameter(d)] = (lower, upper)
        return Bounds(
            rotated_bound_ranges, self.parameterization.union(parameterization)
        )

    def copy(self):
        return Bounds(self.bound_ranges.copy(), self.parameterization.copy())

    def _compute_bounds(self, parameterization=None, nr_sample=10000):
        # TODO this currently guesses the bounds by randomly sampling parameterization. This can be improved in the future.
        # get new parameterization if provided
        if parameterization is not None:
            parameterization = self.parameterization

        # set bound ranges
        computed_bound_ranges = {}
        for key, (lower, upper) in self.bound_ranges.items():
            # compute lower
            if isinstance(lower, (float, int)):
                computed_lower = lower
            elif isinstance(lower, sympy.Basic):
                fn_lower = np_lambdify(lower, parameterization.parameters)
                computed_lower = np.min(fn_lower(**parameterization.sample(nr_sample)))
            else:
                raise ValueError(
                    "Bound has non numeric or sympy values: " + str(self.bound_ranges)
                )

            # compute upper
            if isinstance(upper, (float, int)):
                computed_upper = upper
            elif isinstance(upper, sympy.Basic):
                fn_upper = np_lambdify(upper, parameterization.parameters)
                computed_upper = np.max(fn_upper(**parameterization.sample(nr_sample)))
            else:
                raise ValueError(
                    "Bound has non numeric or sympy values: " + str(self.bound_ranges)
                )

            # store new range
            computed_bound_ranges[key] = (computed_lower, computed_upper)

        return computed_bound_ranges

    def __str__(self):
        return (
            "bound_ranges: "
            + str(self.bound_ranges)
            + " param_ranges: "
            + str(self.parameterization)
        )


def _sample_ranges(batch_size, ranges, quasirandom=False):
    parameterization = {}
    if quasirandom:
        prime_index = 0
        primes = create_primes(1000)
    for key, value in ranges.items():
        # sample parameter
        if isinstance(value, tuple):
            if quasirandom:
                indices = [idx for idx in range(batch_size)]
                rand_param = (
                    value[0]
                    + (value[1] - value[0])
                    * create_samples(indices, number_base=primes[prime_index]).reshape(
                        -1, 1
                    )
                ).astype(float)
                prime_index += 1
            else:
                rand_param = np.random.uniform(value[0], value[1], size=(batch_size, 1))
        elif isinstance(value, (float, int)):
            rand_param = np.zeros((batch_size, 1)) + value
        elif isinstance(value, np.ndarray):
            np_index = np.random.choice(value.shape[0], batch_size)
            rand_param = value[np_index, :]
        elif isinstance(value, Callable):
            rand_param = value(batch_size)
        else:
            raise ValueError(
                "range type: "
                + str(type(value))
                + " not supported, try (tuple, or np.ndarray)"
            )

        # if dependent sample break up parameter
        if isinstance(key, tuple):
            for i, k in enumerate(key):
                parameterization[k] = rand_param[:, i : i + 1]
        else:
            parameterization[key] = rand_param
    return parameterization
