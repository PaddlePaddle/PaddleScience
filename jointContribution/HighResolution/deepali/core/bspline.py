from itertools import combinations_with_replacement
from itertools import permutations
from itertools import product
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import overload

import paddle

from .enum import PaddingMode
from .enum import SpatialDim
from .enum import SpatialDimArg
from .grid import Grid
from .image import conv
from .image import conv1d
from .itertools import is_even_permutation
from .kernels import cubic_bspline1d
from .tensor import move_dim
from .types import ScalarOrTuple


@overload
def cubic_bspline_control_point_grid_size(size: int, stride: int) -> int:
    ...


@overload
def cubic_bspline_control_point_grid_size(
    size: Sequence[int], stride: int
) -> Tuple[int, ...]:
    ...


@overload
def cubic_bspline_control_point_grid_size(
    size: int, stride: Sequence[int]
) -> Tuple[int, ...]:
    ...


@overload
def cubic_bspline_control_point_grid_size(
    size: Sequence[int], stride: Sequence[int]
) -> Tuple[int, ...]:
    ...


def cubic_bspline_control_point_grid_size(
    size: ScalarOrTuple[int], stride: ScalarOrTuple[int]
) -> ScalarOrTuple[int]:
    """Calculate required number of cubic B-spline coefficients for given output size."""
    device = str("cpu").replace("cuda", "gpu")
    m: paddle.Tensor = paddle.atleast_1d(
        paddle.to_tensor(data=size, dtype="int32", place=device)
    )
    s: paddle.Tensor = paddle.atleast_1d(
        paddle.to_tensor(data=stride, dtype="int32", place=device)
    )
    if m.ndim != 1:
        raise ValueError(
            "cubic_bspline_control_point_grid_size() 'size' must be scalar or sequence"
        )
    if m.less_equal(y=paddle.to_tensor(0, dtype=m.dtype)).astype("bool").any():
        raise ValueError(
            "cubic_bspline_control_point_grid_size() 'size' must be positive"
        )
    if s.less_equal(y=paddle.to_tensor(0, dtype=s.dtype)).astype("bool").any():
        raise ValueError(
            "cubic_bspline_control_point_grid_size() 'stride' must be positive"
        )
    ndim = tuple(m.shape)[0]
    if ndim == 1 and tuple(s.shape)[0] > 1:
        ndim = tuple(s.shape)[0]
    for arg, name in zip([m, s], ["size", "stride"]):
        if arg.ndim != 1 or arg.shape[0] not in (1, ndim):
            raise ValueError(
                f"cubic_bspline_control_point_grid_size() {name!r} must be scalar or sequence of length {ndim}"
            )
    m = m.expand(shape=ndim)
    s = s.expand(shape=ndim)
    n = m.div(s, rounding_mode="floor").add_(y=paddle.to_tensor(3))
    n = (m % s == 0).where(x=n, y=n.add(1))
    if isinstance(size, int) and isinstance(stride, int):
        return n[0].item()
    return tuple(n.tolist())


def cubic_bspline_control_point_grid(grid: Grid, stride: ScalarOrTuple[int]) -> Grid:
    """Get control point grid for given image grid and control point stride."""
    size = cubic_bspline_control_point_grid_size(tuple(grid.shape), stride)
    s: paddle.Tensor = paddle.atleast_1d(
        paddle.to_tensor(data=stride, dtype="int32", place=grid.place)
    )
    s = s.expand(shape=grid.ndim)
    return Grid(
        size=size,
        origin=grid.index_to_world(-s),
        spacing=grid.spacing(),
        direction=grid.direction(),
        device=grid.place,
        align_corners=True,
    )


@overload
def bspline_interpolation_weights(
    degree: int,
    stride: int,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[str] = None,
) -> paddle.Tensor:
    ...


@overload
def bspline_interpolation_weights(
    degree: int,
    stride: Sequence[int],
    dtype: Optional[paddle.dtype] = None,
    device: Optional[str] = None,
) -> Tuple[paddle.Tensor, ...]:
    ...


def bspline_interpolation_weights(
    degree: int,
    stride: ScalarOrTuple[int],
    dtype: Optional[paddle.dtype] = None,
    device: Optional[str] = None,
) -> Union[paddle.Tensor, Tuple[paddle.Tensor, ...]]:
    """Compute B-spline interpolation weights."""
    if degree == 3:
        return cubic_bspline_interpolation_weights(stride, dtype=dtype, device=device)
    kernels = {}
    return_single_tensor = False
    if isinstance(stride, int):
        stride = [stride]
        return_single_tensor = True
    for s in stride:
        if s in kernels:
            continue
        kernel = paddle.empty(shape=(s, degree + 1), dtype=dtype)
        offset = paddle.arange(start=0, end=1, step=1 / s, dtype=kernel.dtype)
        if degree % 2 == 0:
            offset = offset.subtract_(y=paddle.to_tensor(offset.round()))
        if degree == 2:
            kernel[:, (1)] = 0.75 - offset.square()
            kernel[:, (2)] = (
                offset.sub(kernel[:, (1)])
                .add_(y=paddle.to_tensor(1))
                .multiply_(y=paddle.to_tensor(0.5))
            )
            kernel[:, (0)] = -kernel[:, ([1, 2])].sum(axis=1).sub(1)
        elif degree == 4:
            a = offset.square()
            t = a.mul(1 / 6)
            t0 = t.sub(11 / 24).mul(offset)
            t1 = (
                t.sub(0.25)
                .multiply_(y=paddle.to_tensor(-a))
                .add_(y=paddle.to_tensor(19 / 96))
            )
            kernel[:, (0)] = (
                paddle.to_tensor(data=0.5, dtype=dtype, place=device)
                .sub(offset)
                .square()
            )
            kernel[:, (0)] = kernel[:, (0)].mul(kernel[:, (0)].mul(1 / 24))
            kernel[:, (1)] = t1.add(t0)
            kernel[:, (3)] = t1.sub(t0)
            kernel[:, (4)] = (
                offset.mul(0.5)
                .add_(y=paddle.to_tensor(kernel[:, (0)]))
                .add_(y=paddle.to_tensor(t0))
            )
            kernel[:, (2)] = -kernel[:, ([0, 1, 3, 4])].sum(axis=1).sub(1)
        elif degree == 5:
            a = offset.square()
            kernel[:, (5)] = offset.mul(a.square()).multiply_(
                y=paddle.to_tensor(1 / 120)
            )
            a = a.subtract_(y=paddle.to_tensor(offset))
            b = a.square()
            offset = offset.subtract_(y=paddle.to_tensor(0.5))
            t = a.sub(3).multiply_(y=paddle.to_tensor(a))
            kernel[:, (0)] = (
                a.add(b)
                .add_(y=paddle.to_tensor(1 / 5))
                .multiply_(y=paddle.to_tensor(1 / 24))
                .subtract_(y=paddle.to_tensor(kernel[:, (5)]))
            )
            t0 = (
                a.sub(5)
                .multiply_(y=paddle.to_tensor(a))
                .add_(y=paddle.to_tensor(46 / 5))
                .multiply_(y=paddle.to_tensor(1 / 24))
            )
            t1 = (
                t.add(4)
                .multiply_(y=paddle.to_tensor(offset))
                .multiply_(y=paddle.to_tensor(-1 / 12))
            )
            kernel[:, (2)] = t0.add(t1)
            kernel[:, (3)] = t0.sub(t1)
            t0 = t.sub(9 / 5).multiply_(y=paddle.to_tensor(1.0 / 16.0))
            t1 = (
                b.sub(a)
                .subtract_(y=paddle.to_tensor(5))
                .multiply_(y=paddle.to_tensor(offset))
                .multiply_(y=paddle.to_tensor(1.0 / 24.0))
            )
            kernel[:, (1)] = t0.add(t1)
            kernel[:, (4)] = t0.sub(t1)
        else:
            raise NotImplementedError(f"B-spline interpolation for degree={degree}")
        kernels[s] = kernel
    kernels = tuple(kernels[s] for s in stride)
    if return_single_tensor:
        assert len(kernels) == 1
        return kernels[0]
    return kernels


@overload
def cubic_bspline_interpolation_weights(
    stride: int,
    derivative: int = 0,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[str] = None,
) -> paddle.Tensor:
    ...


@overload
def cubic_bspline_interpolation_weights(
    stride: int,
    derivative: Sequence[int],
    dtype: Optional[paddle.dtype] = None,
    device: Optional[str] = None,
) -> Tuple[paddle.Tensor, ...]:
    ...


@overload
def cubic_bspline_interpolation_weights(
    stride: Sequence[int],
    derivative: ScalarOrTuple[int] = 0,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[str] = None,
) -> Tuple[paddle.Tensor, ...]:
    ...


def cubic_bspline_interpolation_weights(
    stride: ScalarOrTuple[int],
    derivative: ScalarOrTuple[int] = 0,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[str] = None,
) -> Union[paddle.Tensor, Tuple[paddle.Tensor, ...]]:
    """Compute cubic B-spline interpolation weights."""
    kernels = {}
    return_single_tensor = isinstance(stride, int) and isinstance(derivative, int)
    if isinstance(stride, int):
        stride = [stride] * (len(derivative) if isinstance(derivative, Sequence) else 1)
    if isinstance(derivative, int):
        derivative = [derivative] * len(stride)
    elif not isinstance(derivative, Sequence):
        raise TypeError(
            "cubic_bspline_interpolation_weights() 'derivative' must be int or Sequence[int]"
        )
    elif len(derivative) != len(stride):
        raise ValueError(
            "cubic_bspline_interpolation_weights() length of 'derivative' sequence does not match 'stride'"
        )
    for s, d in zip(stride, derivative):
        if (s, d) in kernels:
            continue
        kernel = paddle.empty(shape=(s, 4), dtype=dtype)
        offset = paddle.arange(start=0, end=1, step=1 / s, dtype=kernel.dtype)
        if d == 0:
            kernel[:, (3)] = offset.pow(y=3).multiply_(y=paddle.to_tensor(1 / 6))
            kernel[:, (0)] = (
                offset.mul(offset.sub(1))
                .multiply_(y=paddle.to_tensor(0.5))
                .add_(y=paddle.to_tensor(1 / 6))
                .subtract_(y=paddle.to_tensor(kernel[:, (3)]))
            )
            kernel[:, (2)] = offset.add(kernel[:, (0)]).subtract_(
                y=paddle.to_tensor(kernel[:, (3)].mul(2))
            )
            kernel[:, (1)] = -kernel[:, ([0, 2, 3])].sum(axis=1).sub(1)
        elif d == 1:
            kernel[:, (3)] = offset.pow(y=2).multiply_(y=paddle.to_tensor(0.5))
            kernel[:, (0)] = offset.sub(kernel[:, (3)]).subtract_(
                y=paddle.to_tensor(0.5)
            )
            kernel[:, (2)] = (
                kernel[:, (0)].sub(kernel[:, (3)].mul(2)).add_(y=paddle.to_tensor(1))
            )
            kernel[:, (1)] = -kernel[:, ([0, 2, 3])].sum(axis=1)
        elif d == 2:
            kernel[:, (3)] = offset
            kernel[:, (0)] = -offset.sub(1)
            kernel[:, (2)] = -offset.mul(3).subtract_(y=paddle.to_tensor(1))
            kernel[:, (1)] = offset.mul(3).subtract_(y=paddle.to_tensor(2))
        elif d == 3:
            kernel[:, (3)] = 1
            kernel[:, (0)] = -1
            kernel[:, (2)] = -3
            kernel[:, (1)] = 3
        else:
            kernel.fill_(value=0)
        kernels[s, d] = kernel
    kernels = tuple(kernels[s, d] for s, d in zip(stride, derivative))
    if return_single_tensor:
        assert len(kernels) == 1
        return kernels[0]
    return kernels


def evaluate_cubic_bspline(
    data: paddle.Tensor,
    stride: ScalarOrTuple[int],
    size: Optional[list] = None,
    shape: Optional[list] = None,
    kernel: Optional[Union[paddle.Tensor, Sequence[paddle.Tensor]]] = None,
    derivative: ScalarOrTuple[int] = 0,
    transpose: bool = False,
) -> paddle.Tensor:
    """Evaluate cubic B-spline function.

    Args:
        data: Cubic B-spline interpolation coefficients as tensor of shape ``(N, C, ..., X)``.
        stride: Number of output grid points between control points plus one. This is the stride of the
            transposed convolution used to upsample the control point displacements to the output size.
            If a sequence of values is given, these must be the strides for the different spatial
            dimensions in the order ``(sx, ...)``.
        size: Spatial size of output tensor in the order ``(nx, ...)``.
        shape: Spatial size of output tensor in the order ``(..., nx)``.
        kernel: Precomputed cubic B-spline interpolation kernel. When multiple 1D kernels are given,
            these must be in the order ``(kx, ...)``.
        transpose: Whether to use separable transposed convolution as implemented in AIRLab.
            When ``False``, a more efficient implementation using multi-channel convolution followed
            by a reshuffling of the output is performed. This more efficient and also more accurate
            implementation is adapted from the C++ code of MIRTK (``mirtk::BSplineInterpolateImageFunction``).

    Returns:
        Cubic B-spline function values as tensor of shape ``(N, C, ..., X')``, where ``X' = sx * X``
        when neither output ``size`` nor ``shape`` is specified. Otherwise, the output tensor is cropped
        to the requested spatial output size.

    """
    if not isinstance(data, paddle.Tensor):
        raise TypeError("evaluate_cubic_bspline() 'data' must be paddle.Tensor")
    if not paddle.is_floating_point(x=data):
        raise TypeError(
            "evaluate_cubic_bspline() 'data' must have floating point dtype"
        )
    if data.ndim < 3:
        raise ValueError(
            "evaluate_cubic_bspline() 'data' must have shape (N, C, ..., X)"
        )
    if size is not None:
        if shape is not None:
            raise ValueError(
                "evaluate_cubic_bspline() 'size' and 'shape' are mutually exclusive"
            )
        shape = tuple(reversed(size))
    D = data.ndim - 2
    N = tuple(data.shape)[0]
    C = tuple(data.shape)[1]
    if isinstance(stride, int):
        stride = [stride] * D
    if transpose:
        if kernel is None:
            if derivative != 0:
                raise NotImplementedError(
                    "evaluate_cubic_bspline() 'derivative' must be 0 when kernel=None and transpose=True"
                )
            kernels = {}
            for s in stride:
                if s not in kernels:
                    kernels[s] = cubic_bspline1d(s)
            kernel = [kernels[s] for s in stride]
        stride = tuple(reversed(stride))
        if isinstance(kernel, Sequence):
            kernel = tuple(reversed(kernel))
        output = conv(
            data,
            kernel=kernel,
            stride=stride,
            padding=PaddingMode.ZEROS,
            transpose=True,
        )
        if shape is not None:
            output = output[
                (slice(0, N), slice(0, C))
                + tuple(slice(s, s + n) for s, n in zip(stride, shape))
            ]
    else:
        if kernel is None:
            kernel = cubic_bspline_interpolation_weights(
                stride=stride,
                derivative=derivative,
                dtype=data.dtype,
                device=data.place,
            )
        elif isinstance(kernel, paddle.Tensor):
            kernel = [kernel] * D
        elif not isinstance(kernel, Sequence):
            raise TypeError(
                "evaluate_cubic_bspline() 'kernel' must be paddle.Tensor or Sequence of tensors"
            )
        output = data
        dims = tuple(SpatialDim(dim).tensor_dim(data.ndim) for dim in range(D))
        conv_fn: Callable[..., paddle.Tensor] = [
            paddle.nn.functional.conv1d,
            paddle.nn.functional.conv2d,
            paddle.nn.functional.conv3d,
        ][D - 1]
        for dim, w in zip(dims, kernel):
            weight = w.reshape(
                (tuple(w.shape)[0], 1, tuple(w.shape)[1]) + (1,) * (D - 1)
            )
            weight = weight.tile(repeat_times=(C,) + (1,) * (weight.ndim - 1))
            output = move_dim(output, dim, 2)
            output = conv_fn(output, weight, groups=C)
            output = output.reshape((N, C, tuple(w.shape)[0]) + tuple(output.shape)[2:])
            x = output
            perm_10 = list(range(x.ndim))
            perm_10[2] = 3
            perm_10[3] = 2
            output = x.transpose(perm=perm_10).flatten(start_axis=2, stop_axis=3)
            output = move_dim(output, 2, dim)
        if shape is not None:
            output = output[
                (slice(0, N), slice(0, C)) + tuple(slice(0, n) for n in shape)
            ]
    return output


def cubic_bspline_jacobian_det(
    data: paddle.Tensor, stride: ScalarOrTuple[int]
) -> paddle.Tensor:
    """Evaluate Jacobian determinant of cubic B-spline free-form deformation."""
    if not isinstance(data, paddle.Tensor):
        raise TypeError("cubic_bspline_jacobian_det() 'data' must be paddle.Tensor")
    if not paddle.is_floating_point(x=data):
        raise TypeError(
            "cubic_bspline_jacobian_det() 'data' must have floating point dtype"
        )
    if data.ndim < 3:
        raise ValueError(
            "cubic_bspline_jacobian_det() 'data' must have shape (N, C, ..., X)"
        )
    D = data.ndim - 2
    C = tuple(data.shape)[1]
    if C != D:
        raise ValueError(
            f"cubic_bspline_jacobian_det() 'data' mismatch between number of channels ({C}) and spatial dimensions ({D})"
        )
    jac: Optional[paddle.Tensor] = None
    for perm in permutations(range(D)):
        term: Optional[paddle.Tensor] = None
        for i, j in zip(range(D), perm):
            derivative = [(1 if dim == j else 0) for dim in range(D)]
            start_18 = data.shape[1] + i if i < 0 else i
            du = evaluate_cubic_bspline(
                paddle.slice(data, [1], [start_18], [start_18 + 1]),
                stride=stride,
                derivative=derivative,
            )
            if i == j:
                du = du.add_(y=paddle.to_tensor(1))
            term = du if term is None else term.multiply_(y=paddle.to_tensor(du))
        assert term is not None
        if jac is None:
            jac = term
        elif is_even_permutation(perm):
            jac = jac.add_(y=paddle.to_tensor(term))
        else:
            jac = jac.subtract_(y=paddle.to_tensor(term))
    assert jac is not None
    return jac


def cubic_bspline_jacobian_dict(
    data: paddle.Tensor,
    stride: ScalarOrTuple[int],
    size: Optional[list] = None,
    shape: Optional[list] = None,
    add_identity: bool = False,
) -> Dict[Tuple[int, int], paddle.Tensor]:
    """Evaluate Jacobian of cubic B-spline free-form deformation.

    Args:
        data: Cubic B-spline interpolation coefficients as tensor of shape ``(N, D, ..., X)``,
            where ``D`` is the number of spatial dimensions.
        stride: Number of output grid points between control points plus one. If a sequence of
            values is given, these must be the strides for the different spatial dimensions in
            the order ``(sx, ...)``.
        size: Spatial size of output tensor in the order ``(nx, ...)``.
        shape: Spatial size of output tensor in the order ``(..., nx)``.
        add_identity: Whether to calculate derivatives of :math:`u(x)` (False) or the free-form
            deformation given by :math:`x + u(x)` (True), where :math:`u` is the cubic B-spline
            function, by adding the identity matrix to the Jacobian of :math:`u`.

    Returns:
        Dictionary of spatial derivatives with keys corresponding to (row, col) indices.

    """
    if not isinstance(data, paddle.Tensor):
        raise TypeError("cubic_bspline_jacobian_dict() 'data' must be paddle.Tensor")
    if not paddle.is_floating_point(x=data):
        raise TypeError(
            "cubic_bspline_jacobian_dict() 'data' must have floating point dtype"
        )
    if data.ndim < 3:
        raise ValueError(
            "cubic_bspline_jacobian_dict() 'data' must have shape (N, C, ..., X)"
        )
    if size is not None:
        if shape is not None:
            raise ValueError(
                "cubic_bspline_jacobian_dict() 'size' and 'shape' are mutually exclusive"
            )
        shape = tuple(reversed(size))
    D = data.ndim - 2
    C = tuple(data.shape)[1]
    if C != D:
        raise ValueError(
            f"cubic_bspline_jacobian_dict() 'data' mismatch between number of channels ({C}) and spatial dimensions ({D})"
        )
    jac = {}
    for i, j in combinations_with_replacement(range(D), 2):
        derivative = [(1 if dim == j else 0) for dim in range(D)]
        start_19 = data.shape[1] + i if i < 0 else i
        coeff = paddle.slice(data, [1], [start_19], [start_19 + 1])
        deriv = evaluate_cubic_bspline(
            coeff, shape=shape, stride=stride, derivative=derivative
        )
        if add_identity and i == j:
            deriv = deriv.add_(y=paddle.to_tensor(1))
        jac[i, j] = deriv
    return {
        (i, j): jac[(i, j) if i < j else (j, i)] for i, j in product(range(D), repeat=2)
    }


def cubic_bspline_jacobian_matrix(
    data: paddle.Tensor,
    stride: ScalarOrTuple[int],
    size: Optional[list] = None,
    shape: Optional[list] = None,
    add_identity: bool = False,
) -> paddle.Tensor:
    """Evaluate Jacobian of cubic B-spline free-form deformation.

    Args:
        data: Cubic B-spline interpolation coefficients as tensor of shape ``(N, D, ..., X)``,
            where ``D`` is the number of spatial dimensions.
        stride: Number of output grid points between control points plus one. If a sequence of
            values is given, these must be the strides for the different spatial dimensions in
            the order ``(sx, ...)``.
        size: Spatial size of output tensor in the order ``(nx, ...)``.
        shape: Spatial size of output tensor in the order ``(..., nx)``.
        add_identity: Whether to calculate derivatives of :math:`u(x)` (False) or the free-form
            deformation given by :math:`x + u(x)` (True), where :math:`u` is the cubic B-spline
            function, by adding the identity matrix to the Jacobian of :math:`u`.

    Returns:
        Full Jacobian matrices as tensors of shape ``(N, ..., X, D, D)``.

    """
    N = tuple(data.shape)[0]
    D = data.ndim - 2
    jac = cubic_bspline_jacobian_dict(
        data, stride=stride, shape=shape, size=size, add_identity=add_identity
    )
    mat = paddle.concat(x=[jac[i, j] for i, j in product(range(D), repeat=2)], axis=1)
    mat = move_dim(mat, 1, -1)
    mat = mat.reshape((N,) + tuple(jac[0, 0].shape)[2:] + (D, D))
    return mat


def cubic_bspline_jacobian_triu(
    data: paddle.Tensor,
    stride: ScalarOrTuple[int],
    size: Optional[list] = None,
    shape: Optional[list] = None,
    add_identity: bool = False,
) -> paddle.Tensor:
    """Evaluate Jacobian of cubic B-spline free-form deformation.

    Args:
        data: Cubic B-spline interpolation coefficients as tensor of shape ``(N, D, ..., X)``,
            where ``D`` is the number of spatial dimensions.
        stride: Number of output grid points between control points plus one. If a sequence of
            values is given, these must be the strides for the different spatial dimensions in
            the order ``(sx, ...)``.
        size: Spatial size of output tensor in the order ``(nx, ...)``.
        shape: Spatial size of output tensor in the order ``(..., nx)``.
        add_identity: Whether to calculate derivatives of :math:`u(x)` (False) or the free-form
            deformation given by :math:`x + u(x)` (True), where :math:`u` is the cubic B-spline
            function, by adding the identity matrix to the Jacobian of :math:`u`.

    Returns:
        Flattened upper triangular Jacobian matrices as tensors of shape ``(N, D * (D + 1) / 2, ..., X)``.

    """
    D = data.ndim - 2
    jac = cubic_bspline_jacobian_dict(
        data, stride=stride, shape=shape, size=size, add_identity=add_identity
    )
    return paddle.concat(
        x=[jac[i, j] for i, j in combinations_with_replacement(range(D), 2)], axis=1
    )


def subdivide_cubic_bspline(
    data: paddle.Tensor,
    dims: Optional[Union[SpatialDimArg, Sequence[SpatialDimArg]]] = None,
) -> paddle.Tensor:
    """Compute cubic B-spline coefficients for subdivided control point grid.

    Args:
        data: Input control point coefficients as tensor of shape ``(N, C, ..., X)``.
        dims: Spatial dimensions along which to subdivide.

    Returns:
        Coefficients of subdivided cubic B-spline function.

    """
    if not isinstance(data, paddle.Tensor):
        raise TypeError("subdivide_cubic_bspline() 'data' must be paddle.Tensor")
    if not paddle.is_floating_point(x=data):
        raise TypeError(
            "subdivide_cubic_bspline() 'data' must have floating point dtype"
        )
    if data.ndim < 4:
        raise ValueError(
            "subdivide_cubic_bspline() 'data' must have shape (N, C, ..., X)"
        )
    if dims is None:
        dims = tuple(range(data.ndim - 2))
    elif isinstance(dims, (int, str)):
        dims = [dims]
    elif not isinstance(dims, Sequence):
        raise TypeError(
            "subdivide_cubic_bspline() 'dims' must be int, str, or Sequence thereof"
        )
    dims = sorted(SpatialDim.from_arg(dim).tensor_dim(data.ndim) for dim in dims)
    output = data
    kernel_1 = paddle.to_tensor(
        data=[0.125, 0.75, 0.125], dtype=data.dtype, place=data.place
    )
    kernel_2 = paddle.to_tensor(data=[0.5, 0.5], dtype=data.dtype, place=data.place)
    for dim in dims:
        shape = (
            tuple(output.shape)[:dim]
            + (2 * tuple(output.shape)[dim] - 1,)
            + tuple(output.shape)[dim + 1 :]
        )
        temp = paddle.empty(shape=shape, dtype=data.dtype)
        indices = [slice(0, n) for n in shape]
        indices[dim] = slice(0, shape[dim], 2)
        temp[tuple(indices)] = conv1d(output, kernel_1, dim=dim, padding=1)
        indices = [slice(0, n) for n in shape]
        indices[dim] = slice(1, shape[dim], 2)
        temp[tuple(indices)] = conv1d(output, kernel_2, dim=dim, padding=0)
        output = temp
    return output
