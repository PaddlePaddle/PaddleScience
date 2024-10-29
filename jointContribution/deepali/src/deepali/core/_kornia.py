r"""Conversion functions between different representations of 3D rotations."""

# Copyright (C) 2017-2019, Arraiy, Inc., all rights reserved.
# Copyright (C) 2019-    , Kornia authors, all rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The conversion functions of this module have been copied and adapted from kornia
# https://github.com/kornia/kornia/blob/8b016621105fcbf1fb1ae13be8361471887bab16/kornia/geometry/conversions.py.
#
# Temporary support for QuaternionCoeffOrder.XYZW, which will be deprecated in kornia >0.6,
# has been removed from the functions here (cf. https://github.com/kornia/kornia/issues/903).


import paddle
from deepali.utils import paddle_aux

__all__ = (
    "angle_axis_to_rotation_matrix",
    "angle_axis_to_quaternion",
    "normalize_quaternion",
    "rotation_matrix_to_angle_axis",
    "rotation_matrix_to_quaternion",
    "quaternion_to_angle_axis",
    "quaternion_to_rotation_matrix",
    "quaternion_log_to_exp",
    "quaternion_exp_to_log",
)


def angle_axis_to_rotation_matrix(angle_axis: paddle.Tensor) -> paddle.Tensor:
    r"""Convert 3d vector of axis-angle rotation to 3x3 rotation matrix

    Args:
        angle_axis (paddle.Tensor): tensor of 3d vector of axis-angle rotations.

    Returns:
        paddle.Tensor: tensor of 3x3 rotation matrices.

    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 3, 3)`

    Example:
        >>> input = paddle.rand(1, 3)  # Nx3
        >>> output = angle_axis_to_rotation_matrix(input)  # Nx3x3
    """
    if not isinstance(angle_axis, paddle.Tensor):
        raise TypeError("Input type is not a paddle.Tensor. Got {}".format(type(angle_axis)))

    if not tuple(angle_axis.shape)[-1] == 3:
        raise ValueError(
            "Input size must be a (*, 3) tensor. Got {}".format(tuple(angle_axis.shape))
        )

    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = paddle.sqrt(x=theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = paddle.chunk(x=wxyz, chunks=3, axis=1)
        cos_theta = paddle.cos(x=theta)
        sin_theta = paddle.sin(x=theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = paddle.concat(x=[r00, r01, r02, r10, r11, r12, r20, r21, r22], axis=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = paddle.chunk(x=angle_axis, chunks=3, axis=1)
        k_one = paddle.ones_like(x=rx)
        rotation_matrix = paddle.concat(x=[k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], axis=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = paddle.unsqueeze(x=angle_axis, axis=1)
    theta2 = paddle.matmul(
        x=_angle_axis,
        y=_angle_axis.transpose(perm=paddle_aux.transpose_aux_func(_angle_axis.ndim, 1, 2)),
    )
    theta2 = paddle.squeeze(x=theta2, axis=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-06
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.place)
    mask_pos = mask.astype(dtype=theta2.dtype)
    mask_neg = (mask is False).astype(dtype=theta2.dtype)

    # create output pose matrix
    batch_size = tuple(angle_axis.shape)[0]
    rotation_matrix = paddle.eye(num_rows=3).to(angle_axis.place).astype(dtype=angle_axis.dtype)
    rotation_matrix = rotation_matrix.view(1, 3, 3).tile(repeat_times=[batch_size, 1, 1])
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = (
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    )
    return rotation_matrix  # Nx3x3


def rotation_matrix_to_angle_axis(rotation_matrix: paddle.Tensor) -> paddle.Tensor:
    r"""Convert 3x3 rotation matrix to Rodrigues vector.

    Args:
        rotation_matrix (paddle.Tensor): rotation matrix.

    Returns:
        paddle.Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 3)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = paddle.rand(2, 3, 3)  # Nx3x3
        >>> output = rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if not isinstance(rotation_matrix, paddle.Tensor):
        raise TypeError(f"Input type is not a paddle.Tensor. Got {type(rotation_matrix)}")

    if not tuple(rotation_matrix.shape)[-2:] == (3, 3):
        raise ValueError(
            f"Input size must be a (*, 3, 3) tensor. Got {tuple(rotation_matrix.shape)}"
        )
    quaternion: paddle.Tensor = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)


def rotation_matrix_to_quaternion(
    rotation_matrix: paddle.Tensor, eps: float = 1e-08
) -> paddle.Tensor:
    r"""Convert 3x3 rotation matrix to 4d quaternion vector.

    The quaternion vector has components in (w, x, y, z) or (x, y, z, w) format.

    .. note::
        The (x, y, z, w) order is going to be deprecated in favor of efficiency.

    Args:
        rotation_matrix (paddle.Tensor): the rotation matrix to convert.
        eps (float): small value to avoid zero division. Default: 1e-8.
        order (QuaternionCoeffOrder): quaternion coefficient order. Default: 'xyzw'.
          Note: 'xyzw' will be deprecated in favor of 'wxyz'.

    Return:
        paddle.Tensor: the rotation in quaternion.

    Shape:
        - Input: :math:`(*, 3, 3)`
        - Output: :math:`(*, 4)`

    Example:
        >>> input = paddle.rand(4, 3, 3)  # Nx3x3
        >>> output = rotation_matrix_to_quaternion(input, eps=paddle.finfo(input.dtype).eps,
        ...                                        order=QuaternionCoeffOrder.WXYZ)  # Nx4
    """
    if not isinstance(rotation_matrix, paddle.Tensor):
        raise TypeError(f"Input type is not a paddle.Tensor. Got {type(rotation_matrix)}")

    if not tuple(rotation_matrix.shape)[-2:] == (3, 3):
        raise ValueError(
            f"Input size must be a (*, 3, 3) tensor. Got {tuple(rotation_matrix.shape)}"
        )

    def safe_zero_division(numerator: paddle.Tensor, denominator: paddle.Tensor) -> paddle.Tensor:
        eps: float = paddle.finfo(dtype=numerator.dtype).tiny
        return numerator / paddle.clip(x=denominator, min=eps)

    rotation_matrix_vec: paddle.Tensor = rotation_matrix.view(*tuple(rotation_matrix.shape)[:-2], 9)

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = paddle.chunk(
        x=rotation_matrix_vec, chunks=9, axis=-1
    )

    trace: paddle.Tensor = m00 + m11 + m22

    def trace_positive_cond():
        sq = paddle.sqrt(x=trace + 1.0) * 2.0
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        return paddle.concat(x=(qw, qx, qy, qz), axis=-1)

    def cond_1():
        sq = paddle.sqrt(x=1.0 + m00 - m11 - m22 + eps) * 2.0
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        return paddle.concat(x=(qw, qx, qy, qz), axis=-1)

    def cond_2():
        sq = paddle.sqrt(x=1.0 + m11 - m00 - m22 + eps) * 2.0
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        return paddle.concat(x=(qw, qx, qy, qz), axis=-1)

    def cond_3():
        sq = paddle.sqrt(x=1.0 + m22 - m00 - m11 + eps) * 2.0
        qw = safe_zero_division(m10 - m01, sq)
        qx = safe_zero_division(m02 + m20, sq)
        qy = safe_zero_division(m12 + m21, sq)
        qz = 0.25 * sq
        return paddle.concat(x=(qw, qx, qy, qz), axis=-1)

    where_2 = paddle.where(condition=m11 > m22, x=cond_2(), y=cond_3())
    where_1 = paddle.where(condition=(m00 > m11) & (m00 > m22), x=cond_1(), y=where_2)

    quaternion: paddle.Tensor = paddle.where(
        condition=trace > 0.0, x=trace_positive_cond(), y=where_1
    )
    return quaternion


def normalize_quaternion(quaternion: paddle.Tensor, eps: float = 1e-12) -> paddle.Tensor:
    r"""Normalizes a quaternion.

    The quaternion should be in (x, y, z, w) format.

    Args:
        quaternion (paddle.Tensor): a tensor containing a quaternion to be
          normalized. The tensor can be of shape :math:`(*, 4)`.
        eps (Optional[bool]): small value to avoid division by zero.
          Default: 1e-12.

    Return:
        paddle.Tensor: the normalized quaternion of shape :math:`(*, 4)`.

    Example:
        >>> quaternion = paddle.to_tensor((1., 0., 1., 0.))
        >>> normalize_quaternion(quaternion)
        tensor([0.7071, 0.0000, 0.7071, 0.0000])
    """
    if not isinstance(quaternion, paddle.Tensor):
        raise TypeError("Input type is not a paddle.Tensor. Got {}".format(type(quaternion)))
    if not tuple(quaternion.shape)[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(tuple(quaternion.shape))
        )
    return paddle.nn.functional.normalize(x=quaternion, p=2.0, axis=-1, epsilon=eps)


# based on:
# https://github.com/matthew-brett/transforms3d/blob/8965c48401d9e8e66b6a8c37c65f2fc200a076fa/transforms3d/quaternions.py#L101
# https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/rotation_matrix_3d.py#L247


def quaternion_to_rotation_matrix(quaternion: paddle.Tensor) -> paddle.Tensor:
    r"""Converts a quaternion to a rotation matrix.

    The quaternion should be in (x, y, z, w) or (w, x, y, z) format.

    Args:
        quaternion (paddle.Tensor): a tensor containing a quaternion to be
          converted. The tensor can be of shape :math:`(*, 4)`.
        order (QuaternionCoeffOrder): quaternion coefficient order. Default: 'xyzw'.
          Note: 'xyzw' will be deprecated in favor of 'wxyz'.

    Return:
        paddle.Tensor: the rotation matrix of shape :math:`(*, 3, 3)`.

    Example:
        >>> quaternion = paddle.to_tensor((0., 0., 0., 1.))
        >>> quaternion_to_rotation_matrix(quaternion, order=QuaternionCoeffOrder.WXYZ)
        tensor([[-1.,  0.,  0.],
                [ 0., -1.,  0.],
                [ 0.,  0.,  1.]])
    """
    if not isinstance(quaternion, paddle.Tensor):
        raise TypeError(f"Input type is not a paddle.Tensor. Got {type(quaternion)}")

    if not tuple(quaternion.shape)[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape (*, 4). Got {tuple(quaternion.shape)}")

    # normalize the input quaternion
    quaternion_norm: paddle.Tensor = normalize_quaternion(quaternion)

    # unpack the normalized quaternion components
    w, x, y, z = paddle.chunk(x=quaternion_norm, chunks=4, axis=-1)

    # compute the actual conversion
    tx: paddle.Tensor = 2.0 * x
    ty: paddle.Tensor = 2.0 * y
    tz: paddle.Tensor = 2.0 * z
    twx: paddle.Tensor = tx * w
    twy: paddle.Tensor = ty * w
    twz: paddle.Tensor = tz * w
    txx: paddle.Tensor = tx * x
    txy: paddle.Tensor = ty * x
    txz: paddle.Tensor = tz * x
    tyy: paddle.Tensor = ty * y
    tyz: paddle.Tensor = tz * y
    tzz: paddle.Tensor = tz * z
    one: paddle.Tensor = paddle.to_tensor(data=1.0)
    matrix: paddle.Tensor = paddle.stack(
        x=(
            one - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            one - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            one - (txx + tyy),
        ),
        axis=-1,
    ).view(-1, 3, 3)
    if len(tuple(quaternion.shape)) == 1:
        matrix = paddle.squeeze(x=matrix, axis=0)
    return matrix


def quaternion_to_angle_axis(quaternion: paddle.Tensor) -> paddle.Tensor:
    r"""Convert quaternion vector to angle axis of rotation.

    The quaternion should be in (x, y, z, w) or (w, x, y, z) format.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (paddle.Tensor): tensor with quaternions.
        order (QuaternionCoeffOrder): quaternion coefficient order. Default: 'xyzw'.
          Note: 'xyzw' will be deprecated in favor of 'wxyz'.

    Return:
        paddle.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = paddle.rand(2, 4)  # Nx4
        >>> angle_axis = quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not paddle.is_tensor(x=quaternion):
        raise TypeError(f"Input type is not a paddle.Tensor. Got {type(quaternion)}")

    if not tuple(quaternion.shape)[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape Nx4 or 4. Got {tuple(quaternion.shape)}")

    # unpack input and compute conversion
    q1: paddle.Tensor = paddle.to_tensor(data=[])
    q2: paddle.Tensor = paddle.to_tensor(data=[])
    q3: paddle.Tensor = paddle.to_tensor(data=[])
    cos_theta: paddle.Tensor = paddle.to_tensor(data=[])

    cos_theta = quaternion[..., 0]
    q1 = quaternion[..., 1]
    q2 = quaternion[..., 2]
    q3 = quaternion[..., 3]

    sin_squared_theta: paddle.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: paddle.Tensor = paddle.sqrt(x=sin_squared_theta)
    two_theta: paddle.Tensor = 2.0 * paddle.where(
        condition=cos_theta < 0.0,
        x=paddle.atan2(x=-sin_theta, y=-cos_theta),
        y=paddle.atan2(x=sin_theta, y=cos_theta),
    )

    k_pos: paddle.Tensor = two_theta / sin_theta
    k_neg: paddle.Tensor = 2.0 * paddle.ones_like(x=sin_theta)
    k: paddle.Tensor = paddle.where(condition=sin_squared_theta > 0.0, x=k_pos, y=k_neg)

    angle_axis: paddle.Tensor = paddle.zeros_like(x=quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def quaternion_log_to_exp(quaternion: paddle.Tensor, eps: float = 1e-08) -> paddle.Tensor:
    r"""Applies exponential map to log quaternion.

    The quaternion should be in (x, y, z, w) or (w, x, y, z) format.

    Args:
        quaternion (paddle.Tensor): a tensor containing a quaternion to be
          converted. The tensor can be of shape :math:`(*, 3)`.
        order (QuaternionCoeffOrder): quaternion coefficient order. Default: 'xyzw'.
          Note: 'xyzw' will be deprecated in favor of 'wxyz'.

    Return:
        paddle.Tensor: the quaternion exponential map of shape :math:`(*, 4)`.

    Example:
        >>> quaternion = paddle.to_tensor((0., 0., 0.))
        >>> quaternion_log_to_exp(quaternion, eps=paddle.finfo(quaternion.dtype).eps,
        ...                       order=QuaternionCoeffOrder.WXYZ)
        tensor([1., 0., 0., 0.])
    """
    if not isinstance(quaternion, paddle.Tensor):
        raise TypeError(f"Input type is not a paddle.Tensor. Got {type(quaternion)}")

    if not tuple(quaternion.shape)[-1] == 3:
        raise ValueError(f"Input must be a tensor of shape (*, 3). Got {tuple(quaternion.shape)}")

    # compute quaternion norm
    norm_q: paddle.Tensor = paddle.linalg.norm(x=quaternion, p=2, axis=-1, keepdim=True).clip(
        min=eps
    )

    # compute scalar and vector
    quaternion_vector: paddle.Tensor = quaternion * paddle.sin(x=norm_q) / norm_q
    quaternion_scalar: paddle.Tensor = paddle.cos(x=norm_q)

    # compose quaternion and return
    quaternion_exp: paddle.Tensor = paddle.to_tensor(data=[])
    quaternion_exp = paddle.concat(x=(quaternion_scalar, quaternion_vector), axis=-1)

    return quaternion_exp


def quaternion_exp_to_log(quaternion: paddle.Tensor, eps: float = 1e-08) -> paddle.Tensor:
    r"""Applies the log map to a quaternion.

    The quaternion should be in (x, y, z, w) format.

    Args:
        quaternion (paddle.Tensor): a tensor containing a quaternion to be
          converted. The tensor can be of shape :math:`(*, 4)`.
        eps (float): A small number for clamping.
        order (QuaternionCoeffOrder): quaternion coefficient order. Default: 'xyzw'.
          Note: 'xyzw' will be deprecated in favor of 'wxyz'.

    Return:
        paddle.Tensor: the quaternion log map of shape :math:`(*, 3)`.

    Example:
        >>> quaternion = paddle.to_tensor((1., 0., 0., 0.))
        >>> quaternion_exp_to_log(quaternion, eps=paddle.finfo(quaternion.dtype).eps,
        ...                       order=QuaternionCoeffOrder.WXYZ)
        tensor([0., 0., 0.])
    """
    if not isinstance(quaternion, paddle.Tensor):
        raise TypeError(f"Input type is not a paddle.Tensor. Got {type(quaternion)}")

    if not tuple(quaternion.shape)[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape (*, 4). Got {tuple(quaternion.shape)}")

    # unpack quaternion vector and scalar
    quaternion_vector: paddle.Tensor = paddle.to_tensor(data=[])
    quaternion_scalar: paddle.Tensor = paddle.to_tensor(data=[])

    quaternion_scalar = quaternion[..., 0:1]
    quaternion_vector = quaternion[..., 1:4]

    # compute quaternion norm
    norm_q: paddle.Tensor = paddle.linalg.norm(
        x=quaternion_vector, p=2, axis=-1, keepdim=True
    ).clip(min=eps)

    # apply log map
    quaternion_log: paddle.Tensor = (
        quaternion_vector
        * paddle.acos(x=paddle.clip(x=quaternion_scalar, min=-1.0, max=1.0))
        / norm_q
    )
    return quaternion_log


# based on:
# https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py#L138


def angle_axis_to_quaternion(angle_axis: paddle.Tensor) -> paddle.Tensor:
    r"""Convert an angle axis to a quaternion.

    The quaternion vector has components in (x, y, z, w) or (w, x, y, z) format.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        angle_axis (paddle.Tensor): tensor with angle axis.
        order (QuaternionCoeffOrder): quaternion coefficient order. Default: 'xyzw'.
          Note: 'xyzw' will be deprecated in favor of 'wxyz'.

    Return:
        paddle.Tensor: tensor with quaternion.

    Shape:
        - Input: :math:`(*, 3)` where `*` means, any number of dimensions
        - Output: :math:`(*, 4)`

    Example:
        >>> angle_axis = paddle.rand(2, 3)  # Nx3
        >>> quaternion = angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)  # Nx4
    """
    if not paddle.is_tensor(x=angle_axis):
        raise TypeError(f"Input type is not a paddle.Tensor. Got {type(angle_axis)}")

    if not tuple(angle_axis.shape)[-1] == 3:
        raise ValueError(f"Input must be a tensor of shape Nx3 or 3. Got {tuple(angle_axis.shape)}")

    # unpack input and compute conversion
    a0: paddle.Tensor = angle_axis[..., 0:1]
    a1: paddle.Tensor = angle_axis[..., 1:2]
    a2: paddle.Tensor = angle_axis[..., 2:3]
    theta_squared: paddle.Tensor = a0 * a0 + a1 * a1 + a2 * a2

    theta: paddle.Tensor = paddle.sqrt(x=theta_squared)
    half_theta: paddle.Tensor = theta * 0.5

    mask: paddle.Tensor = theta_squared > 0.0
    ones: paddle.Tensor = paddle.ones_like(x=half_theta)

    k_neg: paddle.Tensor = 0.5 * ones
    k_pos: paddle.Tensor = paddle.sin(x=half_theta) / theta
    k: paddle.Tensor = paddle.where(condition=mask, x=k_pos, y=k_neg)
    w: paddle.Tensor = paddle.where(condition=mask, x=paddle.cos(x=half_theta), y=ones)

    quaternion: paddle.Tensor = paddle.zeros(
        shape=(*tuple(angle_axis.shape)[:-1], 4), dtype=angle_axis.dtype
    )
    quaternion[..., 1:2] = a0 * k
    quaternion[..., 2:3] = a1 * k
    quaternion[..., 3:4] = a2 * k
    quaternion[..., 0:1] = w
    return quaternion
