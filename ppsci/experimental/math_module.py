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

from __future__ import annotations

import functools
from typing import Any
from typing import Callable
from typing import List
from typing import Tuple

import numpy as np
import paddle
from typing_extensions import Literal


def bessel_i0(x: paddle.Tensor) -> paddle.Tensor:
    """Zero-order modified Bézier curve functions of the first kind.

    Args:
        x (paddle.Tensor): Input data of the formula.

    Examples:
        >>> import paddle
        >>> import ppsci
        >>> res = ppsci.experimental.bessel_i0(paddle.to_tensor([0, 1, 2, 3, 4], dtype="float32"))
    """
    return paddle.i0(x)


def bessel_i0e(x: paddle.Tensor) -> paddle.Tensor:
    """Exponentially scaled zero-order modified Bézier curve functions of the first kind.

    Args:
        x (paddle.Tensor): Input data of the formula.

    Examples:
        >>> import paddle
        >>> import ppsci
        >>> res = ppsci.experimental.bessel_i0e(paddle.to_tensor([0, 1, 2, 3, 4], dtype="float32"))
    """
    return paddle.i0e(x)


def bessel_i1(x: paddle.Tensor) -> paddle.Tensor:
    """First-order modified Bézier curve functions of the first kind.

    Args:
        x (paddle.Tensor): Input data of the formula.

    Examples:
        >>> import paddle
        >>> import ppsci
        >>> res = ppsci.experimental.bessel_i1(paddle.to_tensor([0, 1, 2, 3, 4], dtype="float32"))
    """
    return paddle.i1(x)


def bessel_i1e(x: paddle.Tensor) -> paddle.Tensor:
    """Exponentially scaled first-order modified Bézier curve functions of the first kind.

    Args:
        x (paddle.Tensor): Input data of the formula.

    Examples:
        >>> import paddle
        >>> import ppsci
        >>> res = ppsci.experimental.bessel_i1e(paddle.to_tensor([0, 1, 2, 3, 4], dtype="float32"))
    """
    return paddle.i1e(x)


def expand_func_values_and_squeeze_integral(f: Callable):
    """This decorator ensures that the trailing dimension of integrands is indeed the integrand dimension.
    This is pertinent in the 1d case when the sampled values are often of shape `(N,)`.  Then, to maintain backward
    consistency, we squeeze the result in the 1d case so it does not have any trailing dimensions.

    Args:
        f (Callable): the wrapped function.
    """

    @functools.wraps(f)
    def wrap(*args, **kwargs):
        # i.e we only have one dimension, or the second dimension (that of the integrand) is 1
        is_1d = len(args[0].shape) == 1 or (
            len(args[0].shape) == 2 and args[0].shape[1] == 1
        )
        if is_1d:
            return paddle.squeeze(
                f(paddle.unsqueeze(args[0], axis=1), *args[1:], **kwargs)
            )
        return f(*args, **kwargs)

    return wrap


def gaussian_integrate(
    fn: Callable[[Any], paddle.Tensor],
    dim: int,
    N: int,
    integration_domains: List[List[float]],
    dtype: Literal["float32", "float64"] = "float64",
) -> paddle.Tensor:
    """Integrate given function using gaussian quadrature.

    Args:
        fn (Callable[[Any], paddle.Tensor]): Function to be integrated.
        dim (int): Dimensionality of the integrand.
        N (int): Number of dicretization points.
        integration_domains (List[List[float]]): Intergration domains.
        dtype (Literal["float32", "float64"], optional): Dtype used during computation. Defaults to "float64".

    Returns:
        paddle.Tensor: Integral result.

    Examples:
        >>> import numpy as np
        >>> import paddle
        >>> import ppsci.experimental
        >>> func = lambda x: paddle.sin(x)
        >>> dim = 1
        >>> N = 500
        >>> integration_domains = [[0, np.pi]]
        >>> result = ppsci.experimental.gaussian_integrate(func, dim, N, integration_domains)
        >>> np.testing.assert_allclose(float(result), 2.0, 1e-6)
        >>> print(float(result))
        1.9999999999999576
    """

    def _compatible_meshgrid(*args: paddle.Tensor, **kwargs: paddle.Tensor):
        # TODO(HydrogenSulfate): paddle.meshgrid do not support single Tensor,
        # which will be fixed in paddle framework.
        if len(args) == 1:
            return args
        else:
            return paddle.meshgrid(*args, **kwargs)

    def _roots(N: int) -> np.ndarray:
        return np.polynomial.legendre.leggauss(N)[0]

    def _calculate_grid(
        N: int,
        integration_domains: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, int]:
        """Calculate grid points, widths and N per dim

        Args:
            N (int): Number of points.
            integration_domain (paddle.Tensor): Integration domain.

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor, int]: Grid points, grid widths and
                Number of grid slices per dimension.
        """
        # Create grid and assemble evaluation points
        grid_1d = []
        _dim = integration_domains.shape[0]
        n_per_dim = int(N ** (1.0 / _dim) + 1e-8)

        # Determine for each dimension grid points and mesh width
        def _resize_roots(
            integration_domain: Tuple[float, float], roots: np.ndarray
        ):  # scale from [-1,1] to [a,b]
            a = integration_domain[0]
            b = integration_domain[1]
            return ((b - a) / 2) * roots + ((a + b) / 2)

        for dim in range(_dim):
            grid_1d.append(_resize_roots(integration_domains[dim], _roots(n_per_dim)))
        h = paddle.stack([grid_1d[dim][1] - grid_1d[dim][0] for dim in range(_dim)])

        # Get grid points
        points = _compatible_meshgrid(*grid_1d)
        points = paddle.stack([mg.reshape([-1]) for mg in points], axis=1)

        return points, h, n_per_dim

    def _evaluate_integrand(fn, points, weights=None, fn_args=None) -> paddle.Tenosr:
        """Evaluate the integrand function at the passed points.

        Args:
            fn (function): Integrand function.
            points (paddle.Tensor): Integration points.
            weights (paddle.Tensor, optional): Integration weights. Defaults to None.
            fn_args (list or tuple, optional): Any arguments required by the function. Defaults to None.

        Returns:
            paddle.Tenosr: Integral result.
        """
        if fn_args is None:
            fn_args = ()

        result = fn(points, *fn_args)
        if not str(result.dtype).endswith(dtype):
            result = result.astype(dtype)

        if result.shape[0] != points.shape[0]:
            raise ValueError(
                f"The passed function was given {points.shape[0]} points but only returned {result.shape[0]} value(s)."
                f"Please ensure that your function is vectorized, i.e. can be called with multiple evaluation points at once. It should return a tensor "
                f"where first dimension matches length of passed elements. "
            )

        if weights is not None:
            if (
                len(result.shape) > 1
            ):  # if the the integrand is multi-dimensional, we need to reshape/repeat weights so they can be broadcast in the *=
                integrand_shape = result.shape[1:]
                weights = paddle.repeat_interleave(
                    paddle.unsqueeze(weights, axis=1), np.prod(integrand_shape)
                ).reshape((weights.shape[0], *(integrand_shape)))
            result *= weights

        return result

    def _weights(N, dim):
        """return the weights, broadcast across the dimensions, generated from the polynomial of choice.

        Args:
            N (int): number of nodes.
            dim (int): number of dimensions.

        Returns:
            paddle.Tensor: Integration weights.
        """
        weights = paddle.to_tensor(np.polynomial.legendre.leggauss(N)[1], dtype=dtype)
        return paddle.prod(
            paddle.stack(_compatible_meshgrid(*([weights] * dim)), axis=0),
            axis=0,
        ).reshape([-1])

    def _apply_composite_rule(cur_dim_areas, dim, hs, domain):
        """Apply "composite" rule for gaussian integrals

        cur_dim_areas will contain the areas per dimension
        """
        # We collapse dimension by dimension
        for cur_dim in range(dim):
            cur_dim_areas = (
                0.5
                * (domain[cur_dim][1] - domain[cur_dim][0])
                * paddle.sum(
                    cur_dim_areas, axis=len(cur_dim_areas.shape) - 1, dtype=dtype
                )
            )
        return cur_dim_areas

    @expand_func_values_and_squeeze_integral
    def _calculate_result(
        function_values: paddle.Tensor,
        dim: int,
        n_per_dim: int,
        hs: paddle.Tensor,
        integration_domains: paddle.Tensor,
    ) -> paddle.Tensor:
        """Apply the "composite rule" to calculate a result from the evaluated integrand.

        Args:
            function_values (paddle.Tensor): Output of the integrand.
            dim (int): Dimensionality.
            n_per_dim (int): Number of grid slices per dimension.
            hs (paddle.Tensor): Distances between grid slices for each dimension.

        Returns:
            paddle.Tensor: Quadrature result.
        """
        # Reshape the output to be [integrand_dim,N,N,...] points instead of [integrand_dim,dim*N] points
        integrand_shape = function_values.shape[1:]
        dim_shape = [n_per_dim] * dim
        new_shape = [*integrand_shape, *dim_shape]

        perm = list(range(len(function_values.shape)))
        if len(perm) >= 2:
            perm.append(perm.pop(0))
        reshaped_function_values = paddle.transpose(function_values, perm)
        reshaped_function_values = reshaped_function_values.reshape(new_shape)

        assert new_shape == list(
            reshaped_function_values.shape
        ), f"reshaping produced shape {reshaped_function_values.shape}, expected shape was {new_shape}"

        result = _apply_composite_rule(
            reshaped_function_values, dim, hs, integration_domains
        )
        return result

    assert dtype in [
        "float32",
        "float64",
    ], f"dtype must be either 'float32' or 'float64', but got {dtype}"

    neg = False
    for i, (a, b) in enumerate(integration_domains):
        if a > b:
            neg = not neg
            integration_domains[i] = [b, a]

    integration_domains = paddle.to_tensor(
        integration_domains,
        dtype=dtype,
    )

    if integration_domains.shape[0] != dim:
        raise ValueError(
            f"The number of integration domain({integration_domains.shape[0]}) "
            f"must be equal to the given 'dim'({dim})."
        )
    if integration_domains.shape[1] != 2:
        raise ValueError(
            f"integration_domain should be in format of [[a_1, b_1], [a_2, b_2], ..., "
            f"[a_dim, b_dim]], but got each range of integration is {integration_domains[0]}"
        )
    grid_points, hs, n_per_dim = _calculate_grid(N, integration_domains)

    function_values = _evaluate_integrand(
        fn, grid_points, weights=_weights(n_per_dim, dim)
    )

    result = _calculate_result(function_values, dim, n_per_dim, hs, integration_domains)
    return result if (not neg) else -result


def fractional_diff(
    func: Callable, alpha: float, a: float, t: float, h: float, dtype="float64"
) -> paddle.Tensor:
    r"""Compute fractional derivative of given function at point t with fractional order
    alpha using [Caputo derivative of fractional](https://en.wikipedia.org/wiki/Fractional_calculus#Caputo_fractional_derivative).

    $$
    D_t^\alpha f(t)=\frac{1}{\Gamma(n-\alpha)} \int_0^t \frac{f^{(n)}(s)}{(t-s)^{\alpha+1-n}} d s .
    $$

    $$
    s.t. 0 \lt \alpha \lt 1 .
    $$

    Args:
        func (Callable): Function to compute the fractional derivative of.
        alpha (float): Fractional order.
        t (float): Point to compute the fractional derivative at.
        a (float): Start point of the fractional integral.
        h (float): Step size for finite difference.
        dtype (str, optional): Data dtype during computation. Defaults to "float64".

    Returns:
        paddle.Tensor: Fractional derivative result of the function at t.

    Examples:
        >>> from ppsci.experimental import fractional_diff
        >>> import numpy as np
        >>> # define f(x) = x^2
        >>> def f(x):
        ...     return x * x
        >>> # compute 0.5-order fractional derivative of f(x) at t=1.0 with step size h=1e-6
        >>> res = fractional_diff(f, alpha=0.5, a=0, t=1.0, h=1e-6, dtype="float64")
        >>> np.testing.assert_allclose(float(res), 1.503547, 1e-6)
    """

    if not (0 < alpha < 1):
        raise NotImplementedError(
            f"Given alpha should be in range (0, 1), but got {alpha}"
        )

    def _finite_derivative(
        func: Callable, x: paddle.Tensor, dx: float
    ) -> paddle.Tensor:
        """Compute the finite difference of a function at x using centered difference.

        Args:
            func (Callable): Function to compute the finite difference of.
            x (paddle.Tensor): Point to compute the finite difference at.
            dx (float): Delta to use for the finite difference.

        Returns:
            paddle.Tensor: First-order Finite difference of the function at x.
        """
        return (func(x + dx) - func(x - dx)) / (2 * dx)

    def int_func(s):
        return _finite_derivative(func, s, dx=h) / (t - s) ** (alpha)

    result = (
        1.0 / paddle.exp(paddle.lgamma(paddle.to_tensor(1.0 - alpha, dtype=dtype)))
    ) * gaussian_integrate(
        int_func, dim=1, N=2**10 + 1, integration_domains=[[a, t]], dtype=dtype
    )
    return result


def trapezoid_integrate(
    y: paddle.Tensor,
    x: paddle.Tensor = None,
    dx: float = None,
    axis: int = -1,
    mode: Literal["sum", "cumsum"] = "sum",
) -> paddle.Tensor:
    """
    Integrate along the given axis using the composite trapezoidal rule. Use the sum method.

    Args:
        y (paddle.Tensor): Input to be integrated.
        x (paddle.Tensor, optional): The sample points corresponding to the input samples. its shape should be,
        (1) input.shape; (2) the input.shape[axis] if axis is not default. Defaults to None.
        dx (float, optional): The sample points are assumed to be evenly spaced and it is the spacing between sample points.
        If 'x' and 'dx' are both default, 'dx' is set to 1 by default. Defaults to None.
        axis (int, optional): The axis along which to integrate. Defaults to -1.
        mode (Literal["sum", "cumsum"], optional): Which type cumulative sum function used. Defaults to "sum".

    Returns:
        paddle.Tensor: Integral result. If dim of input is N, return is N-1 dim.

    Examples:
        >>> import paddle
        >>> import ppsci
        >>> y = paddle.to_tensor([[0, 1, 2], [3, 4, 5]], dtype="float32")
        >>> res = ppsci.experimental.trapezoid_integrate(y)
        >>> print(res)
        Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            [2., 8.])
        >>> res = ppsci.experimental.trapezoid_integrate(y, mode="cumsum")
        >>> print(res)
        Tensor(shape=[2, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            [[0.50000000, 2.        ],
            [3.50000000, 8.        ]])
        >>> res = ppsci.experimental.trapezoid_integrate(
            y, x=paddle.to_tensor([[0, 1, 2], [3, 4, 5]], dtype="float32")
        )
        >>> print(res)
        Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            [2., 8.])
        >>> res = ppsci.experimental.trapezoid_integrate(
            y, x=paddle.to_tensor([0, 1], dtype="float32"), axis=0
        )
        >>> print(res)
        Tensor(shape=[3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            [1.50000000, 2.50000000, 3.50000000])
        >>> res = ppsci.experimental.trapezoid_integrate(
            y, x=paddle.to_tensor([0, 1, 2], dtype="float32"), axis=1
        )
        >>> print(res)
        Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            [2., 8.])
        >>> res = ppsci.experimental.trapezoid_integrate(y, dx=2)
        >>> print(res)
        Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            [4. , 16.])
    """
    if mode == "sum":
        return paddle.trapezoid(y, x, dx, axis)
    elif mode == "cumsum":
        return paddle.cumulative_trapezoid(y, x, dx, axis)
    else:
        raise ValueError(f'mode should be "sum" or "cumsum", but got {mode}')
