r"""Basic linear algebra functions, e.g., to work with homogeneous coordinate transformations."""

from enum import Enum
from functools import reduce
from operator import mul
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import paddle
from deepali.utils import paddle_aux  # noqa
from paddle import Tensor

from ._kornia import angle_axis_to_quaternion
from ._kornia import angle_axis_to_rotation_matrix
from ._kornia import normalize_quaternion
from ._kornia import quaternion_exp_to_log
from ._kornia import quaternion_log_to_exp
from ._kornia import quaternion_to_angle_axis
from ._kornia import quaternion_to_rotation_matrix
from ._kornia import rotation_matrix_to_angle_axis
from ._kornia import rotation_matrix_to_quaternion
from .tensor import as_tensor
from .typing import DeviceStr
from .typing import DType

__all__ = (
    "as_homogeneous_matrix",
    "as_homogeneous_tensor",
    "hmm",
    "homogeneous_matmul",
    "homogeneous_matrix",
    "homogeneous_transform",
    "tensordot",
    "vectordot",
    "vector_rotation",
    # adapted from kornia.geometry
    "angle_axis_to_rotation_matrix",
    "angle_axis_to_quaternion",
    "rotation_matrix_to_angle_axis",
    "rotation_matrix_to_quaternion",
    "quaternion_to_angle_axis",
    "quaternion_to_rotation_matrix",
    "quaternion_log_to_exp",
    "quaternion_exp_to_log",
    "normalize_quaternion",
)


class HomogeneousTensorType(Enum):
    r"""Type of homogeneous transformation tensor."""

    AFFINE = "affine"  # Square affine transformation matrix.
    HOMOGENEOUS = "homogeneous"  # Full homogeneous transformation matrix.
    TRANSLATION = "translation"  # Translation vector.


def as_homogeneous_tensor(
    tensor: paddle.Tensor, dtype: Optional[DType] = None, device: Optional[DeviceStr] = None
) -> Tuple[paddle.Tensor, HomogeneousTensorType]:
    r"""Convert tensor to homogeneous coordinate transformation."""
    tensor_ = as_tensor(tensor, dtype=dtype, device=device)
    if tensor_.ndim == 0:
        raise ValueError("Expected at least 1-dimensional 'tensor'")
    if tensor_.ndim == 1:
        tensor_ = tensor_.unsqueeze(axis=1)
    if tuple(tensor_.shape)[-1] == 1:
        type_ = HomogeneousTensorType.TRANSLATION
    elif tuple(tensor_.shape)[-1] == tuple(tensor_.shape)[-2]:
        type_ = HomogeneousTensorType.AFFINE
    elif tuple(tensor_.shape)[-1] == tuple(tensor_.shape)[-2] + 1:
        type_ = HomogeneousTensorType.HOMOGENEOUS
    else:
        raise ValueError(f"Invalid homogeneous 'tensor' shape {tuple(tensor_.shape)}")
    return tensor_, type_


def as_homogeneous_matrix(
    tensor: paddle.Tensor, dtype: Optional[DType] = None, device: Optional[DeviceStr] = None
) -> paddle.Tensor:
    r"""Convert tensor to homogeneous coordinate transformation matrix.

    Args:
        tensor: Tensor of translations of shape ``(D,)`` or ``(..., D, 1)``, tensor of square affine
            matrices of shape ``(..., D, D)``, or tensor of homogeneous transformation matrices of
            shape ``(..., D, D + 1)``.
        dtype: Data type of output matrix. If ``None``, use ``tensor.dtype`` or default.
        device: Device on which to create matrix. If ``None``, use ``tensor.device`` or default.

    Returns:
        Homogeneous coordinate transformation matrices of shape ``(..., D, D + 1)``. If ``tensor`` has already
        shape ``(..., D, D + 1)``, a reference to this tensor is returned without making a copy, unless requested
        ``dtype`` and ``device`` differ from ``tensor`` (cf. ``as_tensor()``). Use ``homogeneous_matrix()``
        if a copy of the input ``tensor`` should always be made.


    """
    tensor_, type_ = as_homogeneous_tensor(tensor, dtype=dtype, device=device)
    if type_ == HomogeneousTensorType.TRANSLATION:
        A = paddle.eye(num_rows=tuple(tensor_.shape)[-2], dtype=tensor_.dtype)
        tensor_ = paddle.concat(x=[A, tensor_], axis=-1)
    elif type_ == HomogeneousTensorType.AFFINE:
        t = paddle.to_tensor(data=0, dtype=tensor_.dtype, place=tensor_.place).expand(
            shape=[*tuple(tensor_.shape)[:-1], 1]
        )
        tensor_ = paddle.concat(x=[tensor_, t], axis=-1)
    elif type_ != HomogeneousTensorType.HOMOGENEOUS:
        raise ValueError(
            "Expected 'tensor' to have shape (D,), (..., D, 1), (..., D, D) or (..., D, D + 1)"
        )
    return tensor_


def homogeneous_transform(
    transform: paddle.Tensor, points: paddle.Tensor, vectors: bool = False
) -> paddle.Tensor:
    r"""Transform points or vectors by given homogeneous transformations.

    The data type used for matrix-vector products, as well as the data type of
    the resulting tensor, is by default set to ``points.dtype``. If ``points.dtype``
    is not a floating point data type, ``transforms.dtype`` is used instead.

    Args:
        transform: Tensor of translations of shape ``(D,)``, ``(D, 1)`` or ``(N, D, 1)``, tensor of
            affine transformation matrices of shape ``(D, D)`` or ``(N, D, D)``, or tensor of homogeneous
            matrices of shape ``(D, D + 1)`` or ``(N, D, D + 1)``, respectively. When 3-dimensional
            batch of transformation matrices is given, the size of leading dimension N must be 1
            for applying the same transformation to all points, or be equal to the leading dimension
            of ``points``, otherwise. All points within a given batch dimension are transformed by
            the matrix of matching leading index. If size of ``points`` batch dimension is one,
            the size of the leading output batch dimension is equal to the number of transforms,
            each applied to the same set of input points.
        points: Either 1-dimensional tensor of single point coordinates, or multi-dimensional tensor
            of shape ``(N, ..., D)``, where last dimension contains the spatial coordinates in the
            order ``(x, y)`` (2D) or ``(x, y, z)`` (3D), respectively.
        vectors: Whether ``points`` is tensor of vectors. If ``True``, only the affine
            component of the given ``transforms`` is applied without any translation offset.
            If ``transforms`` is a 2-dimensional tensor of translation offsets, a tensor sharing
            the data memory of the input ``points`` is returned.

    Returns:
        Tensor of transformed points/vectors with the same shape as the input ``points``, except
        for the size of the leading batch dimension if the size of the input ``points`` batch dimension
        is one, but the ``transform`` batch contains multiple transformations.

    """
    if transform.ndim == 0:
        raise TypeError("homogeneous_transform() 'transform' must be non-scalar tensor")
    if transform.ndim == 1:
        transform = transform.unsqueeze(axis=1)
    if transform.ndim == 2:
        transform = transform.unsqueeze(axis=0)
    N = tuple(transform.shape)[0]
    D = tuple(transform.shape)[1]
    if N < 1:
        raise ValueError(
            "homogeneous_transform() 'transform' size of leading dimension must not be zero"
        )
    if (
        transform.ndim != 3
        or tuple(transform.shape)[2] != 1
        and (1 < tuple(transform.shape)[2] < D or tuple(transform.shape)[2] > D + 1)
    ):
        raise ValueError(
            "homogeneous_transform() 'transform' must be tensor of shape"
            + " (D,), (D, 1), (D, D), (D, D + 1) or (N, D, 1), (N, D, D), (N, D, D + 1)"
        )
    if points.ndim == 0:
        raise TypeError("'points' must be non-scalar tensor")
    if points.ndim == 1 and len(points) != D or points.ndim > 1 and tuple(points.shape)[-1] != D:
        raise ValueError(
            "homogeneous_transform() 'points' number of spatial dimensions does not match 'transform'"
        )
    if points.ndim == 1:
        output_shape = (N,) + tuple(points.shape) if N > 1 else tuple(points.shape)
        points = points.expand(shape=(N,) + tuple(points.shape))
    elif N == 1:
        output_shape = tuple(points.shape)
    elif tuple(points.shape)[0] == 1 or tuple(points.shape)[0] == N:
        output_shape = (N,) + tuple(points.shape)[1:]
        points = points.expand(shape=(N,) + tuple(points.shape)[1:])
    else:
        raise ValueError(
            "homogeneous_transform() expected size of leading dimension of 'transform' and 'points' to be either 1 or equal"
        )
    points = points.reshape(N, -1, D)
    if paddle.is_floating_point(x=points):
        transform = transform.astype(points.dtype)
    else:
        points = points.astype(transform.dtype)
    if tuple(transform.shape)[2] == 1:
        if not vectors:
            points = points + transform[..., 0].unsqueeze(axis=1)
    else:
        points = paddle.bmm(
            x=points,
            y=transform[:, :D, :D].transpose(
                perm=paddle_aux.transpose_aux_func(transform[:, :D, :D].ndim, 1, 2)
            ),
        )
        if not vectors and tuple(transform.shape)[2] == D + 1:
            points += transform[..., D].unsqueeze(axis=1)
    return points.reshape(output_shape)


def hmm(a: paddle.Tensor, b: paddle.Tensor) -> paddle.Tensor:
    r"""Compose two homogeneous coordinate transformations.

    Args:
        a: Tensor of second homogeneous transformation.
        b: Tensor of first homogeneous transformation.

    Returns:
        Composite homogeneous transformation given by a tensor of shape ``(..., D, D + 1)``.

    See also:
        ``homogeneous_matmul()``

    """
    c = homogeneous_matmul(a, b)
    return as_homogeneous_matrix(c)


def homogeneous_matmul(*args: paddle.Tensor) -> paddle.Tensor:
    r"""Compose homogeneous coordinate transformations.

    This function performs the equivalent of a matrix-matrix product for homogeneous coordinate transformations
    given as either a translation vector (tensor of shape ``(D,)`` or ``(..., D, 1)``), a tensor of square affine
    matrices of shape ``(..., D, D)``, or a tensor of homogeneous coordinate transformation matrices of shape
    ``(..., D, D + 1)``. The size of leading dimensions must either match, or be all 1 for one of the input tensors.
    In the latter case, the same homogeneous transformation is composed with each individual trannsformation of the
    tensor with leading dimension size greater than 1.

    For example, if the shape of tensor ``a`` is either ``(D,)``, ``(D, 1)``, or ``(1, D, 1)``, and the shape of tensor
    ``b`` is ``(N, D, D)``, the translation given by ``a`` is applied after each affine transformation given by each
    matrix in grouped batch tensor ``b``, and the shape of the composite transformation tensor is ``(N, D, D + 1)``.

    Args:
        args: Tensors of homogeneous coordinate transformations, where the transformation corresponding to the first
            argument is applied last, and the transformation corresponding to the last argument is applied first.

    Returns:
        Composite homogeneous transformation given by tensor of shape ``(..., D, 1)``, ``(..., D, D)``, or  ``(..., D, D + 1)``,
        respectively, where the shape of leading dimensions is determined by input tensors.

    """
    if not args:
        raise ValueError("homogeneous_matmul() at least one argument is required")
    # Convert first input to homogeneous transformation tensor
    a = args[0]
    dtype = a.dtype
    device = a.place
    if not paddle_aux.is_floating_point(dtype):  # type: ignore
        for b in args[1:]:
            if b.is_floating_point():
                dtype = b.dtype
                break
        if not paddle_aux.is_floating_point(dtype):  # type: ignore
            dtype = "float32"
    a, a_type = as_homogeneous_tensor(a, dtype=dtype)
    # Successively compose transformation matrices
    D = tuple(a.shape)[-2]
    for b in args[1:]:
        # Convert input to homogeneous transformation tensor
        b, b_type = as_homogeneous_tensor(b, dtype=dtype, device=device)
        b.to(device)
        if not paddle_aux.is_eq_place(b.place, device):
            raise RuntimeError("homogeneous_matmul() tensors must be on the same 'device'")
        if tuple(b.shape)[-2] != D:
            raise ValueError(
                "homogeneous_matmul() tensors have mismatching number of spatial dimensions"
                + f" ({tuple(a.shape)[-2]} != {tuple(b.shape)[-2]})"
            )
        # Unify shape of leading dimensions
        leading_shape = None
        a_numel = len(tuple(a.shape)[:-2])
        b_numel = len(tuple(b.shape)[:-2])
        if a_numel > 1:
            if b_numel > 1 and tuple(a.shape)[:-2] != tuple(b.shape)[:-2]:
                raise ValueError(
                    "Expected homogeneous tensors to have matching leading dimensions:"
                    + f" {tuple(a.shape)[:-2]} != {tuple(b.shape)[:-2]}"
                )
            if b.ndim > a.ndim:
                raise ValueError("Homogeneous tensors have different number of leading dimensions")
            leading_shape = tuple(a.shape)[:-2]
            b = b.expand(shape=leading_shape + tuple(b.shape)[-2:])
        elif b_numel > 1:
            if a.ndim > b.ndim:
                raise ValueError("Homogeneous tensors have different number of leading dimensions")
            leading_shape = tuple(b.shape)[:-2]
            a = a.expand(shape=leading_shape + tuple(a.shape)[-2:])
        elif a.ndim > b.ndim:
            leading_shape = tuple(a.shape)[:-2]
            b = b.expand(shape=tuple(a.shape)[:-2] + tuple(b.shape)[-2:])
        else:
            leading_shape = tuple(b.shape)[:-2]
            a = a.expand(shape=tuple(b.shape)[:-2] + tuple(a.shape)[-2:])
        assert leading_shape is not None
        # Compose homogeneous transformations
        a = a.reshape(-1, *tuple(a.shape)[-2:])
        b = b.reshape(-1, *tuple(b.shape)[-2:])
        c, c_type = None, None
        if a_type == HomogeneousTensorType.TRANSLATION:
            if b_type == HomogeneousTensorType.TRANSLATION:
                c = a + b
                c_type = HomogeneousTensorType.TRANSLATION
            elif b_type == HomogeneousTensorType.AFFINE:
                c = paddle.concat(x=[b, a], axis=-1)
                c_type = HomogeneousTensorType.HOMOGENEOUS
            elif b_type == HomogeneousTensorType.HOMOGENEOUS:
                c = b.clone()
                c[..., D] += a[..., :, 0]
                c_type = HomogeneousTensorType.HOMOGENEOUS
        elif a_type == HomogeneousTensorType.AFFINE:
            if b_type == HomogeneousTensorType.TRANSLATION:
                t = paddle.bmm(x=a, y=b)
                c = paddle.concat(x=[a, t], axis=-1)
                c_type = HomogeneousTensorType.HOMOGENEOUS
            elif b_type == HomogeneousTensorType.AFFINE:
                c = paddle.bmm(x=a, y=b)
                c_type = HomogeneousTensorType.AFFINE
            elif b_type == HomogeneousTensorType.HOMOGENEOUS:
                A = paddle.bmm(x=a, y=b[..., :D])
                t = paddle.bmm(x=a[..., :D], y=b[..., D:])
                c = paddle.concat(x=[A, t], axis=-1)
                c_type = HomogeneousTensorType.HOMOGENEOUS
        elif a_type == HomogeneousTensorType.HOMOGENEOUS:
            if b_type == HomogeneousTensorType.TRANSLATION:
                t = paddle.bmm(x=a[..., :D], y=b)
                c = a.clone()
                c[..., D] += t[..., :, 0]
            elif b_type == HomogeneousTensorType.AFFINE:
                A = paddle.bmm(x=a[..., :D], y=b)
                t = a[..., D:]
                c = paddle.concat(x=[A, t], axis=-1)
            elif b_type == HomogeneousTensorType.HOMOGENEOUS:
                A = paddle.bmm(x=a[..., :D], y=b[..., :D])
                t = a[..., D:] + paddle.bmm(x=a[..., :D], y=b[..., D:])
                c = paddle.concat(x=[A, t], axis=-1)
            c_type = HomogeneousTensorType.HOMOGENEOUS
        assert c is not None, "as_homogeneous_tensor() returned invalid 'type' enumeration value"
        assert c_type is not None
        c = c.reshape(leading_shape + tuple(c.shape)[-2:])
        assert str(c.place) == str(device)
        a, a_type = c, c_type
    return a


def homogeneous_matrix(
    tensor: paddle.Tensor,
    offset: Optional[paddle.Tensor] = None,
    dtype: Optional[DType] = None,
    device: Optional[DeviceStr] = None,
) -> paddle.Tensor:
    r"""Convert square matrix or vector to homogeneous coordinate transformation matrix.

    Args:
        tensor: Tensor of translations of shape ``(D,)`` or ``(..., D, 1)``, tensor of square affine
            matrices of shape ``(..., D, D)``, or tensor of homogeneous transformation matrices of
            shape ``(..., D, D + 1)``.
        offset: Translation offset to add to homogeneous transformations of shape ``(..., D)``.
            If a scalar is given, this offset is used as translation along each spatial dimension.
        dtype: Data type of output matrix. If ``None``, use ``offset.dtype``.
        device: Device on which to create matrix. If ``None``, use ``offset.device``.

    Returns:
        Homogeneous coordinate transformation matrices of shape (..., D, D + 1). Always makes a copy
        of ``tensor`` even if it has already the shape of homogeneous coordinate transformation matrices.

    """
    matrix = as_homogeneous_matrix(tensor, dtype=dtype, device=device)
    if matrix is tensor:
        matrix = tensor.clone()
    if offset is not None:
        D = tuple(matrix.shape)[-2]
        if offset.ndim == 0:
            offset = offset.tile(repeat_times=D)
        if tuple(offset.shape)[-1] != D:
            raise ValueError(
                f"Expected homogeneous_matrix() 'offset' to be scalar or have last dimension of size {D}"
            )
        matrix[..., D] += offset
    return matrix


def tensordot(
    a: paddle.Tensor,
    b: paddle.Tensor,
    dims: Union[int, Sequence[int], Tuple[Sequence[int], Sequence[int]]] = 2,
) -> paddle.Tensor:
    r"""Implements ``numpy.tensordot()`` for ``Tensor``.

    Based on https://gist.github.com/deanmark/9aec75b7dc9fa71c93c4bc85c5438777.

    """
    if isinstance(dims, int):
        axes_a = list(range(-dims, 0))
        axes_b = list(range(0, dims))
    else:
        axes_a, axes_b = dims

    if isinstance(axes_a, int):
        axes_a = [axes_a]
        na = 1
    else:
        na = len(axes_a)
        axes_a = list(axes_a)

    if isinstance(axes_b, int):
        axes_b = [axes_b]
        nb = 1
    else:
        nb = len(axes_b)
        axes_b = list(axes_b)

    a = as_tensor(a)
    b = as_tensor(b)
    as_ = tuple(a.shape)
    nda = a.ndim
    bs = tuple(b.shape)
    ndb = b.ndim
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    # Move the axes to sum over to the end of "a" and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    newshape_a = int(reduce(mul, [as_[ax] for ax in notin])), N2
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    newshape_b = N2, int(reduce(mul, [bs[ax] for ax in notin]))
    oldb = [bs[axis] for axis in notin]

    at = a.transpose(perm=newaxes_a).reshape(newshape_a)
    bt = b.transpose(perm=newaxes_b).reshape(newshape_b)

    res = at.matmul(y=bt)
    return res.reshape(olda + oldb)


def vectordot(
    a: paddle.Tensor, b: paddle.Tensor, w: Optional[paddle.Tensor] = None, dim: int = -1
) -> paddle.Tensor:
    r"""Inner product of vectors over specified input tensor dimension."""
    c = a.mul(b)
    if w is not None:
        c.mul(w)
    return c.sum(axis=dim)


def vector_rotation(a: paddle.Tensor, b: paddle.Tensor) -> paddle.Tensor:
    r"""Calculate rotation matrix which aligns two 3D vectors."""
    if not isinstance(a, paddle.Tensor) or not isinstance(b, paddle.Tensor):
        raise TypeError("vector_rotation() 'a' and 'b' must be of type Tensor")
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError("vector_rotation() 'a' and 'b' must have identical shape")
    a = paddle.nn.functional.normalize(x=a, p=2, axis=-1)
    b = paddle.nn.functional.normalize(x=b, p=2, axis=-1)
    axis = a.cross(y=b, axis=-1)
    norm: Tensor = axis.norm(p=2, axis=-1, keepdim=True)
    angle_axis = axis.div(norm).mul(norm.asin())
    rotation_matrix = angle_axis_to_rotation_matrix(angle_axis)
    return rotation_matrix
