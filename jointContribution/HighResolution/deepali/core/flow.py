from typing import Optional
from typing import Union

import paddle

from . import affine as A
from .enum import PaddingMode
from .enum import Sampling
from .grid import ALIGN_CORNERS
from .grid import Grid
from .image import _image_size
from .image import check_sample_grid
from .image import grid_reshape
from .image import grid_sample
from .image import spatial_derivatives
from .image import zeros_image
from .tensor import move_dim
from .types import Array
from .types import Device
from .types import Scalar
from .types import Shape
from .types import Size


def affine_flow(
    matrix: paddle.Tensor, grid: Union[Grid, paddle.Tensor], channels_last: bool = False
) -> paddle.Tensor:
    """Compute dense flow field from homogeneous transformation.

    Args:
        matrix: Homogeneous coordinate transformation matrices of shape ``(N, D, 1)`` (translation),
            ``(N, D, D)`` (affine), or ``(N, D, D + 1)`` (homogeneous), respectively.
        grid: Image sampling ``Grid`` or tensor of shape ``(N, ..., X, D)`` of points at
            which to sample flow fields. If an object of type ``Grid`` is given, the value
            of ``grid.align_corners()`` determines if output flow vectors are with respect to
            ``Axes.CUBE`` (False) or ``Axes.CUBE_CORNERS`` (True), respectively.
        channels_last: If ``True``, flow vector components are stored in the last dimension
            of the output tensor, and first dimension otherwise.

    Returns:
        paddle.Tensor of shape ``(N, C, ..., X)`` if ``channels_last=False`` and ``(N, ..., X, C)``, otherwise.

    """
    if matrix.ndim != 3:
        raise ValueError(
            f"affine_flow() 'matrix' must be tensor of shape (N, D, 1|D|D+1), not {tuple(matrix.shape)}"
        )
    device = matrix.place
    if isinstance(grid, Grid):
        grid = grid.coords(device=device)
        grid = grid.unsqueeze(axis=0)
    elif grid.ndim < 3:
        raise ValueError(
            f"affine_flow() 'grid' must be tensor of shape (N, ...X, D), not {tuple(grid.shape)}"
        )
    assert grid.place == device
    flow = A.transform_points(matrix, grid) - grid
    if not channels_last:
        flow = move_dim(flow, -1, 1)
    assert flow.place == device
    return flow


def compose_flows(
    a: paddle.Tensor, b: paddle.Tensor, align_corners: bool = True
) -> paddle.Tensor:
    """Compute composite flow field ``c = b o a``."""
    a = move_dim(b, 1, -1)
    c = paddle.nn.functional.grid_sample(
        x=b, grid=a, mode="bilinear", padding_mode="border", align_corners=align_corners
    )
    return c


def curl(
    flow: paddle.Tensor,
    spacing: Optional[Union[Scalar, Array]] = None,
    mode: str = "central",
) -> paddle.Tensor:
    """Calculate curl of vector field.

    TODO: Implement curl for 2D vector field.

    Args:
        flow: Vector field as tensor of shape ``(N, 3, Z, Y, X)``.
        spacing: Physical size of image voxels used to compute ``spatial_derivatives()``.
        mode: Mode of ``spatial_derivatives()`` approximation.

    Returns:
        In case of a 3D input vector field, output is another 3D vector field of rotation vectors,
        where axis of rotation corresponds to the unit vector and rotation angle to the magnitude
        of the rotation vector, as tensor of shape ``(N, 3, Z, Y, X)``.

    """
    if flow.ndim == 4:
        if tuple(flow.shape)[1] != 2:
            raise ValueError("curl() 'flow' must have shape (N, 2, Y, X)")
        raise NotImplementedError("curl() of 2-dimensional vector field")
    if flow.ndim == 5:
        if tuple(flow.shape)[1] != 3:
            raise ValueError("curl() 'flow' must have shape (N, 3, Z, Y, X)")
        start_7 = flow.shape[1] + 0 if 0 < 0 else 0
        dx = spatial_derivatives(
            paddle.slice(flow, [1], [start_7], [start_7 + 1]),
            mode=mode,
            which=("y", "z"),
            spacing=spacing,
        )
        start_8 = flow.shape[1] + 1 if 1 < 0 else 1
        dy = spatial_derivatives(
            paddle.slice(flow, [1], [start_8], [start_8 + 1]),
            mode=mode,
            which=("x", "z"),
            spacing=spacing,
        )
        start_9 = flow.shape[1] + 2 if 2 < 0 else 2
        dz = spatial_derivatives(
            paddle.slice(flow, [1], [start_9], [start_9 + 1]),
            mode=mode,
            which=("x", "y"),
            spacing=spacing,
        )
        rotvec = paddle.concat(
            x=[dz["y"] - dy["z"], dx["z"] - dz["x"], dy["x"] - dx["y"]], axis=1
        )
        return rotvec
    raise ValueError("curl() 'flow' must be 2- or 3-dimensional vector field")


def expv(
    flow: paddle.Tensor,
    scale: Optional[float] = None,
    steps: Optional[int] = None,
    sampling: Union[Sampling, str] = Sampling.LINEAR,
    padding: Union[PaddingMode, str] = PaddingMode.BORDER,
    align_corners: bool = ALIGN_CORNERS,
) -> paddle.Tensor:
    """Group exponential maps of flow fields computed using scaling and squaring.

    Args:
        flow: Batch of flow fields as tensor of shape ``(N, D, ..., X)``.
        scale: Constant flow field scaling factor.
        steps: Number of scaling and squaring steps.
        sampling: Flow field interpolation mode.
        padding: Flow field extrapolation mode.
        align_corners: Whether ``flow`` vectors are defined with respect to
            ``Axes.CUBE`` (False) or ``Axes.CUBE_CORNERS`` (True).

    Returns:
        Exponential map of input flow field. If ``steps=0``, a reference to ``flow`` is returned.

    """
    if scale is None:
        scale = 1
    if steps is None:
        steps = 5
    if not isinstance(steps, int):
        raise TypeError("expv() 'steps' must be of type int")
    if steps < 0:
        raise ValueError("expv() 'steps' must be positive value")
    if steps == 0:
        return flow
    device = flow.place
    grid = Grid(shape=tuple(flow.shape)[2:], align_corners=align_corners)
    grid = grid.coords(dtype=flow.dtype, device=device)
    assert grid.place == device
    disp = flow * (scale / 2**steps)
    assert disp.place == device
    for _ in range(steps):
        disp = disp + warp_image(
            disp,
            grid,
            flow=move_dim(disp, 1, -1),
            mode=sampling,
            padding=padding,
            align_corners=align_corners,
        )
        assert disp.place == device
    return disp


def jacobian_det(
    u: paddle.Tensor, mode: str = "central", channels_last: bool = False
) -> paddle.Tensor:
    """Evaluate Jacobian determinant of given flow field using finite difference approximations.

    Note that for differentiable parametric spatial transformation models, an accurate Jacobian could
    be calculated instead from the given transformation parameters. See for example ``cubic_bspline_jacobian_det()``
    which is specialized for a free-form deformation (FFD) determined by a continuous cubic B-spline function.

    Args:
        u: Input vector field as tensor of shape ``(N, D, ..., X)`` when ``channels_last=False`` and
            shape ``(N, ..., X, D)`` when ``channels_last=True``.
        mode: Mode of ``spatial_derivatives()`` to use for approximating spatial partial derivatives.
        channels_last: Whether input vector field has vector (channels) dimension at second or last index.

    Returns:
        Scalar field of approximate Jacobian determinant values as tensor of shape ``(N, 1, ..., X)`` when
        ``channels_last=False`` and ``(N, ..., X, 1)`` when ``channels_last=True``.

    """
    if u.ndim < 4:
        shape_str = "(N, ..., X, D)" if channels_last else "(N, D, ..., X)"
        raise ValueError(
            f"jacobian_det() 'u' must be dense vector field of shape {shape_str}"
        )
    shape = tuple(u.shape)[1:-1] if channels_last else tuple(u.shape)[2:]
    mat = paddle.empty(shape=(tuple(u.shape)[0],) + shape + (3, 3), dtype=u.dtype)
    for i, which in enumerate(("x", "y", "z")):
        deriv = spatial_derivatives(u, mode=mode, which=which)[which]
        if not channels_last:
            deriv = move_dim(deriv, 1, -1)
        mat[..., i] = deriv
    for i in range(tuple(mat.shape)[-1]):
        mat[..., i, i].add_(y=paddle.to_tensor(1))
    jac = mat.det().unsqueeze_(axis=-1 if channels_last else 1)
    return jac


def normalize_flow(
    data: paddle.Tensor,
    size: Optional[Union[paddle.Tensor, list]] = None,
    side_length: float = 2,
    align_corners: bool = ALIGN_CORNERS,
    channels_last: bool = False,
) -> paddle.Tensor:
    """Map vectors with respect to unnormalized grid to vectors with respect to normalized grid."""
    if not isinstance(data, paddle.Tensor):
        raise TypeError("normalize_flow() 'data' must be tensor")
    if not data.is_floating_point():
        data = data.astype(dtype="float32")
    if size is None:
        if data.ndim < 4 or tuple(data.shape)[1] != data.ndim - 2:
            raise ValueError(
                "normalize_flow() 'data' must have shape (N, D, ..., X) when 'size' not given"
            )
        size = tuple(reversed(tuple(data.shape)[2:]))
    zero = paddle.to_tensor(data=0, dtype=data.dtype, place=data.place)
    size = paddle.to_tensor(data=size, dtype=data.dtype, place=data.place)
    size_ = size.sub(1) if align_corners else size
    if not channels_last:
        data = move_dim(data, 1, -1)
    if side_length != 1:
        data = data.mul(side_length)
    data = paddle.where(condition=size > 1, x=data.div(size_), y=zero)
    if not channels_last:
        data = move_dim(data, -1, 1)
    return data


def denormalize_flow(
    data: paddle.Tensor,
    size: Optional[Union[paddle.Tensor, list]] = None,
    side_length: float = 2,
    align_corners: bool = ALIGN_CORNERS,
    channels_last: bool = False,
) -> paddle.Tensor:
    """Map vectors with respect to normalized grid to vectors with respect to unnormalized grid."""
    if not isinstance(data, paddle.Tensor):
        raise TypeError("denormalize_flow() 'data' must be tensors")
    if size is None:
        if data.ndim < 4 or tuple(data.shape)[1] != data.ndim - 2:
            raise ValueError(
                "denormalize_flow() 'data' must have shape (N, D, ..., X) when 'size' not given"
            )
        size = tuple(reversed(tuple(data.shape)[2:]))
    zero = paddle.to_tensor(data=0, dtype=data.dtype, place=data.place)
    size = paddle.to_tensor(data=size, dtype=data.dtype, place=data.place)
    size_ = size.sub(1) if align_corners else size
    if not channels_last:
        data = move_dim(data, 1, -1)
    data = paddle.where(condition=size > 1, x=data.mul(size_), y=zero)
    if side_length != 1:
        data = data.div(side_length)
    if not channels_last:
        data = move_dim(data, -1, 1)
    return data


def sample_flow(
    flow: paddle.Tensor, coords: paddle.Tensor, align_corners: bool = ALIGN_CORNERS
) -> paddle.Tensor:
    """Sample non-rigid flow fields at given points.

    This function samples a vector field at spatial points. The ``coords`` tensor can be of any shape,
    including ``(N, M, D)``, i.e., a batch of N point sets with cardianality M. It can also be applied to
    a tensor of grid points of shape ``(N, ..., X, D)`` regardless if the grid points are located at the
    undeformed grid positions or an already deformed grid. The given non-rigid flow field is interpolated
    at the input points ``x`` using linear interpolation. These flow vectors ``u(x)`` are returned.

    Args:
        flow: Flow fields of non-rigid transformations given as tensor of shape ``(N, D, ..., X)``
            or ``(1, D, ..., X)``. If batch size is one, but the batch size of ``coords`` is greater
            than one, this single flow fields is sampled at the different sets of points.
        coords: Normalized coordinates of points given as tensor of shape ``(N, ..., D)``
            or ``(1, ..., D)``. If batch size is one, all flow fields are sampled at the same points.
        align_corners: Whether point coordinates are with respect to ``Axes.CUBE`` (False)
            or ``Axes.CUBE_CORNERS`` (True). This option is in particular passed on to the
            ``grid_sample()`` function used to sample the flow vectors at the input points.

    Returns:
        paddle.Tensor of shape ``(N, ..., D)``.

    """
    if not isinstance(flow, paddle.Tensor):
        raise TypeError("sample_flow() 'flow' must be of type paddle.Tensor")
    if flow.ndim < 4:
        raise ValueError("sample_flow() 'flow' must be at least 4-dimensional tensor")
    if not isinstance(coords, paddle.Tensor):
        raise TypeError("sample_flow() 'coords' must be of type paddle.Tensor")
    if coords.ndim < 2:
        raise ValueError("sample_flow() 'coords' must be at least 2-dimensional tensor")
    G = tuple(flow.shape)[0]
    N = tuple(coords.shape)[0] if G == 1 else G
    D = tuple(flow.shape)[1]
    if tuple(coords.shape)[0] not in (1, N):
        raise ValueError(f"sample_flow() 'coords' must be batch of length 1 or {N}")
    if tuple(coords.shape)[-1] != D:
        raise ValueError(
            f"sample_flow() 'coords' must be tensor of {D}-dimensional points"
        )
    x = coords.expand(shape=(N,) + tuple(coords.shape)[1:])
    t = flow.expand(shape=(N,) + tuple(flow.shape)[1:])
    g = x.reshape((N,) + (1,) * (t.ndim - 3) + (-1, D))
    u = grid_sample(t, g, padding=PaddingMode.BORDER, align_corners=align_corners)
    u = move_dim(u, 1, -1)
    u = u.reshape(tuple(x.shape))
    return u


def warp_grid(
    flow: paddle.Tensor, grid: paddle.Tensor, align_corners: bool = ALIGN_CORNERS
) -> paddle.Tensor:
    """Transform undeformed grid by a tensor of non-rigid flow fields.

    This function applies a non-rigid transformation to map a tensor of undeformed grid points to a
    tensor of deformed grid points with the same shape as the input tensor. The input points must be
    the positions of undeformed spatial grid points, because this function uses interpolation to
    resize the vector fields to the size of the input ``grid``. This assumes that input points ``x``
    are the coordinates of points located on a regularly spaced undeformed grid which is aligned with
    the borders of the grid domain on which the vector fields of the non-rigid transformations are
    sampled, i.e., ``y = x + u``.

    If in doubt whether the input points will be sampled regularly at grid points of the domain of
    the spatial transformation, use ``warp_points()`` instead.

    Args:
        flow: Flow fields of non-rigid transformations given as tensor of shape ``(N, D, ..., X)``
            or ``(1, D, ..., X)``. If batch size is one, but the batch size of ``points`` is greater
            than one, all point sets are transformed by the same non-rigid transformation.
        grid: Coordinates of points given as tensor of shape ``(N, ..., D)`` or ``(1, ..., D)``.
            If batch size is one, but multiple flow fields are given, this single point set is
            transformed by each non-rigid transformation to produce ``N`` output point sets.
        align_corners: Whether grid points and flow vectors are with respect to ``Axes.CUBE``
            (False) or ``Axes.CUBE_CORNERS`` (True). This option is in particular passed on to
            the ``grid_reshape()`` function used to resize the flow fields to the ``grid`` shape.

    Returns:
        paddle.Tensor of shape ``(N, ..., D)`` with coordinates of spatially transformed points.

    """
    if not isinstance(flow, paddle.Tensor):
        raise TypeError("warp_grid() 'flow' must be of type paddle.Tensor")
    if flow.ndim < 4:
        raise ValueError("warp_grid() 'flow' must be at least 4-dimensional tensor")
    if not isinstance(grid, paddle.Tensor):
        raise TypeError("warp_grid() 'grid' must be of type paddle.Tensor")
    if grid.ndim < 4:
        raise ValueError("warp_grid() 'grid' must be at least 4-dimensional tensor")
    G = tuple(flow.shape)[0]
    N = tuple(grid.shape)[0] if G == 1 else G
    D = tuple(flow.shape)[1]
    if tuple(grid.shape)[0] not in (1, N):
        raise ValueError(f"warp_grid() 'grid' must be batch of length 1 or {N}")
    if tuple(grid.shape)[-1] != D:
        raise ValueError(f"warp_grid() 'grid' must be tensor of {D}-dimensional points")
    x = grid.expand(shape=(N,) + tuple(grid.shape)[1:])
    t = flow.expand(shape=(N,) + tuple(flow.shape)[1:])
    u = grid_reshape(t, tuple(grid.shape)[1:-1], align_corners=align_corners)
    u = move_dim(u, 1, -1).reshape(x.shape)
    y = x + u
    return y


def warp_points(
    flow: paddle.Tensor, coords: paddle.Tensor, align_corners: bool = ALIGN_CORNERS
) -> paddle.Tensor:
    """Transform set of points by a tensor of non-rigid flow fields.

    This function applies a non-rigid transformation to map a tensor of spatial points to another tensor
    of spatial points of the same shape as the input tensor. Unlike ``warp_grid()``, it can be used
    to spatially transform any set of points which are defined with respect to the grid domain of the
    non-rigid transformation, including a tensor of shape ``(N, M, D)``, i.e., a batch of N point sets with
    cardianality M. It can also be applied to a tensor of grid points of shape ``(N, ..., X, D)`` regardless
    if the grid points are located at the undeformed grid positions or an already deformed grid. The given
    non-rigid flow field is interpolated at the input points ``x`` using linear interpolation. These flow
    vectors ``u(x)`` are then added to the input points, i.e., ``y = x + u(x)``.

    Args:
        flow: Flow fields of non-rigid transformations given as tensor of shape ``(N, D, ..., X)``
            or ``(1, D, ..., X)``. If batch size is one, but the batch size of ``points`` is greater
            than one, all point sets are transformed by the same non-rigid transformation.
        coords: Normalized coordinates of points given as tensor of shape ``(N, ..., D)``
            or ``(1, ..., D)``. If batch size is one, this single point set is deformed by each
            flow field to produce ``N`` output point sets.
        align_corners: Whether points and flow vectors are with respect to ``Axes.CUBE`` (False)
            or ``Axes.CUBE_CORNERS`` (True). This option is in particular passed on to the
            ``grid_sample()`` function used to sample the flow vectors at the input points.

    Returns:
        paddle.Tensor of shape ``(N, ..., D)`` with coordinates of spatially transformed points.

    """
    x = coords
    u = sample_flow(flow, coords, align_corners=align_corners)
    y = x + u
    return y


def warp_image(
    data: paddle.Tensor,
    grid: paddle.Tensor,
    flow: Optional[paddle.Tensor] = None,
    mode: Optional[Union[Sampling, str]] = None,
    padding: Optional[Union[PaddingMode, str, Scalar]] = None,
    align_corners: bool = ALIGN_CORNERS,
) -> paddle.Tensor:
    """Sample data at optionally displaced grid points.

    Args:
        data: Image batch tensor of shape ``(1, C, ..., X)`` or ``(N, C, ..., X)``.
        grid: Grid points tensor of shape  ``(..., X, D)``, ``(1, ..., X, D)``, or``(N, ..., X, D)``.
            Coordinates of points at which to sample ``data`` must be with respect to ``Axes.CUBE``.
        flow: Batch of flow fields of shape  ``(..., X, D)``, ``(1, ..., X, D)``, or``(N, ..., X, D)``.
            If specified, the flow field(s) are added to ``grid`` in order to displace the grid points.
        mode: Image interpolate mode.
        padding: Image extrapolation mode or constant by which to pad input ``data``.
        align_corners: Whether ``grid`` extrema ``(-1, 1)`` refer to the grid boundary
            edges (``align_corners=False``) or corner points (``align_corners=True``).

    Returns:
        Image batch tensor of sampled data with shape determined by ``grid``.

    """
    if data.ndim < 4:
        raise ValueError("warp_image() expected tensor 'data' of shape (N, C, ..., X)")
    grid = check_sample_grid("warp", data, grid)
    N = tuple(grid.shape)[0]
    D = tuple(grid.shape)[-1]
    if flow is not None:
        if flow.ndim == data.ndim - 1:
            flow = flow.unsqueeze(axis=0)
        elif flow.ndim != data.ndim:
            raise ValueError(
                f"warp_image() expected 'flow' tensor with {data.ndim - 1} or {data.ndim} dimensions"
            )
        if tuple(flow.shape)[0] != N:
            flow = flow.expand(shape=[N, *tuple(flow.shape)[1:]])
        if tuple(flow.shape)[0] != N or tuple(flow.shape)[-1] != D:
            msg = f"warp_image() expected tensor 'flow' of shape (..., X, {D})"
            msg += f" or (1, ..., X, {D})" if N == 1 else f" or (1|{N}, ..., X, {D})"
            raise ValueError(msg)
        grid = grid + flow
    assert data.place == grid.place
    return grid_sample(
        data, grid, mode=mode, padding=padding, align_corners=align_corners
    )


def zeros_flow(
    size: Optional[Union[int, Size, Grid]] = None,
    shape: Optional[Shape] = None,
    num: int = 1,
    named: bool = False,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[Device] = None,
) -> paddle.Tensor:
    """Create batch of flow fields filled with zeros for given image batch size or grid."""
    size = _image_size("zeros_flow", size, shape)
    return zeros_image(
        size, num=num, channels=len(size), named=named, dtype=dtype, device=device
    )
