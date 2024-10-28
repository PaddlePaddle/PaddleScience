r"""Linear transformation models."""
from __future__ import annotations  # noqa

import math
from collections import OrderedDict
from typing import Callable
from typing import Optional
from typing import Union

import paddle
from deepali.core import affine as U
from deepali.core.grid import Grid
from deepali.core.linalg import normalize_quaternion
from deepali.core.linalg import quaternion_to_rotation_matrix
from deepali.core.linalg import rotation_matrix_to_quaternion
from deepali.core.tensor import as_float_tensor
from deepali.utils import paddle_aux  # noqa

from .base import LinearTransform
from .composite import SequentialTransform
from .parametric import InvertibleParametricTransform


class HomogeneousTransform(InvertibleParametricTransform, LinearTransform):
    r"""Arbitrary homogeneous coordinate transformation."""

    def __init__(
        self: HomogeneousTransform,
        grid: Grid,
        groups: Optional[int] = None,
        params: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
    ) -> None:
        r"""Initialize transformation parameters.

        Args:
            grid: Grid domain on which transformation is defined.
            groups: Number of transformations. Must be either 1 or equal to batch size.
            params: Homogeneous transform as tensor of shape ``(N, D, D + 1)``.

        """
        super().__init__(grid, groups=groups, params=params)

    @property
    def data_shape(self: HomogeneousTransform) -> list:
        r"""Get shape of transformation parameters tensor."""
        return tuple((self.ndim, self.ndim + 1))

    def matrix_(self: HomogeneousTransform, arg: paddle.Tensor) -> HomogeneousTransform:
        r"""Set transformation matrix."""
        if not isinstance(arg, paddle.Tensor):
            raise TypeError("HomogeneousTransform.matrix() 'arg' must be tensor")
        if arg.ndim != 3:
            raise ValueError("HomogeneousTransform.matrix() 'arg' must be 3-dimensional tensor")
        return self.data_(arg)

    def tensor(self: HomogeneousTransform) -> paddle.Tensor:
        r"""Get tensor representation of this transformation

        Returns:
            Batch of homogeneous transformation matrices as tensor of shape ``(N, D, D + 1)``.

        """
        matrix = self.data()
        if self.invert:
            N = tuple(matrix.shape)[0]
            D = tuple(matrix.shape)[1]
            row = paddle.zeros(shape=(1, 1, D + 1), dtype=matrix.dtype)
            row[..., -1] = 1
            matrix = paddle.concat(x=(matrix, row.expand(shape=[N, 1, D + 1])), axis=1)
            matrix = paddle.linalg.inv(x=matrix)
            start_47 = matrix.shape[1] + 0 if 0 < 0 else 0
            matrix = paddle.slice(matrix, [1], [start_47], [start_47 + D])
            matrix = matrix
        return matrix

    def extra_repr(self: HomogeneousTransform) -> str:
        r"""Print current transformation."""
        s = super().extra_repr() + ", matrix="
        if self.params is None:
            s += "undef"
        else:
            s += f"{self.matrix().tolist()!r}"
        return s


class Translation(InvertibleParametricTransform, LinearTransform):
    r"""Translation."""

    def __init__(
        self: Translation,
        grid: Grid,
        groups: Optional[int] = None,
        params: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
    ) -> None:
        r"""Initialize transformation parameters.

        Args:
            grid: Grid domain on which transformation is defined.
            groups: Number of transformations. Must be either 1 or equal to batch size.
            params: Translation offsets as tensor of shape ``(N, D)`` or ``(N, D, 1)``.

        """
        super().__init__(grid, groups=groups, params=params)

    @property
    def data_shape(self: Translation) -> list:
        r"""Get shape of transformation parameters tensor."""
        return tuple((self.ndim,))

    def offset(self: Translation) -> paddle.Tensor:
        r"""Get current translation offset in cube units."""
        return self.data()

    @paddle.no_grad()
    def offset_(self: Translation, arg: paddle.Tensor) -> Translation:
        r"""Reset parameters to given translation in cube units."""
        if not isinstance(arg, paddle.Tensor):
            raise TypeError("Translation.offset() 'arg' must be tensor")
        params = as_float_tensor(arg)
        if params.isnan().astype("bool").any() or params.isinf().astype("bool").any():
            raise ValueError("Translation.offset() 'arg' must not be nan or inf")
        self.data_(params)
        return self

    def tensor(self: Translation) -> paddle.Tensor:
        r"""Get tensor representation of this transformation

        Returns:
            Batch of homogeneous transformation matrices as tensor of shape ``(N, D, 1)``.

        """
        offset = self.offset()
        if self.invert:
            offset = -offset
        return U.translation(offset)

    def extra_repr(self: Translation) -> str:
        r"""Print current transformation."""
        s = super().extra_repr() + ", offset="
        if self.params is None:
            s += "undef"
        else:
            s += f"{self.offset().tolist()!r}"
        return s


class EulerRotation(InvertibleParametricTransform, LinearTransform):
    r"""Euler rotation."""

    def __init__(
        self: EulerRotation,
        grid: Grid,
        groups: Optional[int] = None,
        params: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
        order: Optional[str] = None,
    ) -> None:
        r"""Initialize transformation parameters.

        Args:
            grid: Grid domain on which transformation is defined.
            groups: Number of transformations. Must be 1 or equal to batch size.
            params: Rotation angles in degrees. This parameterization is adopted from MIRTK
                and ensures that rotation angles and scaling factors (percentage) are within
                a similar range of magnitude which is useful for direct optimization of these
                parameters. If parameters are predicted by a callable ``torch.nn.Module``,
                different output activations may be chosen before converting these to degrees.
            order: Order in which to compose elementary rotations. For example in 3D, "zxz" means
                that the first rotation occurs about z, the second about x, and the third rotation
                about z again. In 2D, this argument is ignored and a single rotation about z
                (plane normal) is applied.

        """
        if grid.ndim < 2 or grid.ndim > 3:
            raise ValueError("EulerRotation() 'grid' must be 2- or 3-dimensional")
        super().__init__(grid, groups=groups, params=params)
        self.order = order

    @property
    def data_shape(self: EulerRotation) -> list:
        r"""Get shape of transformation parameters tensor."""
        return tuple((self.nangles,))

    @property
    def nangles(self: EulerRotation) -> int:
        r"""Number of Euler angles."""
        return 1 if self.ndim == 2 else self.ndim

    def angles(self: EulerRotation) -> paddle.Tensor:
        r"""Get Euler angles in radians."""
        params = self.data()
        if self.has_parameters():
            params = params.tanh().mul(math.pi)
        return params

    @paddle.no_grad()
    def angles_(self: EulerRotation, arg: paddle.Tensor) -> EulerRotation:
        r"""Reset parameters to given Euler angles in radians."""
        if not isinstance(arg, paddle.Tensor):
            raise TypeError("EulerRotation.angles() 'arg' must be tensor")
        shape = self.data_shape
        if arg.ndim != len(shape) + 1:
            raise ValueError(
                f"EulerRotation.angles() 'arg' must be {len(shape) + 1}-dimensional tensor"
            )
        shape = (arg.shape[0],) + self.data_shape
        if arg.shape != shape:
            raise ValueError(f"EulerRotation.angles() 'arg' must have shape {shape!r}")
        params = as_float_tensor(arg)
        if self.has_parameters():
            params = params.div(math.pi).atanh()
            if params.isnan().astype("bool").any():
                raise ValueError("EulerRotation.angles() 'arg' must be in range [-pi, pi]")
        self.data_(params)
        return self

    def matrix_(self: EulerRotation, arg: paddle.Tensor) -> EulerRotation:
        r"""Set rotation angles from rotation matrix."""
        if not isinstance(arg, paddle.Tensor):
            raise TypeError("EulerRotation.matrix() 'arg' must be tensor")
        if arg.ndim != 3:
            raise ValueError("EulerRotation.matrix() 'arg' must be 3-dimensional tensor")
        shape = arg.shape[0], 3, 3
        if arg.shape != shape:
            raise ValueError(f"Rotation matrix must have shape {shape!r}")
        angles = U.euler_rotation_angles(arg, order=self.order)
        return self.angles_(angles)

    def tensor(self: EulerRotation) -> paddle.Tensor:
        r"""Get tensor representation of this transformation

        Returns:
            Batch of homogeneous transformation matrices as tensor of shape ``(N, D, D)``.

        """
        mat = U.euler_rotation_matrix(self.angles(), order=self.order)
        if self.invert:
            mat = mat.transpose(perm=paddle_aux.transpose_aux_func(mat.ndim, 1, 2))
        return mat

    def extra_repr(self: EulerRotation) -> str:
        r"""Print current transformation."""
        s = super().extra_repr() + ", angles="
        if self.params is None:
            s += "undef"
        else:
            s += f"{self.angles().tolist()!r}"
        return s


class QuaternionRotation(InvertibleParametricTransform, LinearTransform):
    r"""Quaternion based rotation in 3D."""

    def __init__(
        self: QuaternionRotation,
        grid: Grid,
        groups: Optional[int] = None,
        params: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
    ) -> None:
        r"""Initialize transformation parameters.

        Args:
            grid: Grid domain on which transformation is defined.
            groups: Number of transformations. Must be either 1 or equal to batch size.
            params: (normalized quaternion as 2-dimensional tensor of ``(N, 4)``.

        """
        if grid.ndim != 3:
            raise ValueError("QuaternionRotation() 'grid' must be 3-dimensional")
        super().__init__(grid, groups=groups, params=params)

    @property
    def data_shape(self: QuaternionRotation) -> list:
        r"""Get shape of transformation parameters tensor."""
        return tuple((4,))

    @paddle.no_grad()
    def reset_parameters(self: QuaternionRotation) -> None:
        r"""Reset transformation parameters."""
        params = self.params
        if isinstance(params, paddle.Tensor):
            paddle.assign(
                paddle.to_tensor(data=[0, 0, 0, 1], dtype=params.dtype, place=params.place),
                output=params,
            )

    def quaternion(self: QuaternionRotation) -> paddle.Tensor:
        r"""Get rotation quaternion."""
        params = self.data()
        return normalize_quaternion(params)

    @paddle.no_grad()
    def quaternion_(self: QuaternionRotation, arg: paddle.Tensor) -> QuaternionRotation:
        r"""Set rotation quaternion."""
        if not isinstance(arg, paddle.Tensor):
            raise TypeError("QuaternionRotation.quaternion() 'arg' must be tensor")
        shape = self.data_shape
        if arg.ndim != len(shape) + 1:
            raise ValueError(
                f"QuaternionRotation.quaternion() 'arg' must be {len(shape) + 1}-dimensional tensor"
            )
        shape = (arg.shape[0],) + self.data_shape
        if arg.shape != shape:
            raise ValueError(f"QuaternionRotation.quaternion() 'arg' must have shape {shape!r}")
        params = as_float_tensor(arg)
        params = normalize_quaternion(params)
        self.data_(params)
        return self

    def matrix_(self: QuaternionRotation, arg: paddle.Tensor) -> QuaternionRotation:
        r"""Set rotation quaternion from rotation matrix."""
        if not isinstance(arg, paddle.Tensor):
            raise TypeError("QuaternionRotation.matrix() 'arg' must be tensor")
        if arg.ndim != 3:
            raise ValueError("QuaternionRotation.matrix() 'arg' must be 3-dimensional tensor")
        shape = arg.shape[0], 3, 3
        if arg.shape != shape:
            raise ValueError(f"Rotation matrix must have shape {shape!r}")
        arg = rotation_matrix_to_quaternion(arg)
        return self.quaternion_(arg)

    def tensor(self: QuaternionRotation) -> paddle.Tensor:
        r"""Get tensor representation of this transformation

        Returns:
            Batch of homogeneous transformation matrices as tensor of shape ``(N, D, D)``.

        """
        q = self.data()
        mat = quaternion_to_rotation_matrix(q)
        if self.invert:
            mat = mat.transpose(perm=paddle_aux.transpose_aux_func(mat.ndim, 1, 2))
        return mat

    def extra_repr(self: QuaternionRotation) -> str:
        r"""Print current transformation."""
        s = super().extra_repr() + ", q="
        if self.params is None:
            s += "undef"
        else:
            s += f"{self.quaternion().tolist()!r}"
        return s


class IsotropicScaling(InvertibleParametricTransform, LinearTransform):
    r"""Isotropic scaling."""

    def __init__(
        self: IsotropicScaling,
        grid: Grid,
        groups: Optional[int] = None,
        params: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
    ) -> None:
        r"""Initialize transformation parameters.

        Args:
            grid: Grid domain on which transformation is defined.
            groups: Number of transformations. Must be either 1 or equal to batch size.
            params: Isotropic scaling factor as a percentage. This parameterization is
                adopted from MIRTK and ensures that rotation angles in degrees and scaling
                factors are within a similar range of magnitude which is useful for direct
                optimization of these parameters. If parameters are predicted by a callable
                ``torch.nn.Module``, different output activations may be chosen before
                converting these to percentages (i.e., multiplied by 100).

        """
        super().__init__(grid, groups=groups, params=params)

    @property
    def data_shape(self: IsotropicScaling) -> list:
        r"""Get shape of transformation parameters tensor."""
        return tuple((1,))

    @paddle.no_grad()
    def reset_parameters(self: IsotropicScaling) -> None:
        r"""Reset transformation parameters."""
        params = self.params
        if isinstance(params, paddle.Tensor):
            init_Constant = paddle.nn.initializer.Constant(value=1)
            init_Constant(params)

    def scales(self: IsotropicScaling) -> paddle.Tensor:
        r"""Get scaling factors."""
        params = self.data()
        if self.has_parameters():
            params = params.sub(1).tanh().exp()
        return params

    @paddle.no_grad()
    def scales_(self: IsotropicScaling, arg: paddle.Tensor) -> IsotropicScaling:
        r"""Set transformation parameters from scaling factors."""
        if not isinstance(arg, paddle.Tensor):
            raise TypeError("IsotropicScaling.scales() 'arg' must be tensor")
        shape = self.data_shape
        if arg.ndim != len(shape) + 1:
            raise ValueError(
                f"IsotropicScaling.scales() 'arg' must be {len(shape) + 1}-dimensional tensor"
            )
        shape = (arg.shape[0],) + shape
        if arg.shape != shape:
            raise ValueError(f"IsotropicScaling.scales() 'arg' must have shape {shape!r}")
        params = as_float_tensor(arg)
        if self.has_parameters():
            params = params.log().atanh().add(1)
            if params.isnan().astype("bool").any():
                raise ValueError("IsotropicScaling.scales() 'arg' must be positive")
        self.data_(params)
        return self

    def tensor(self: IsotropicScaling) -> paddle.Tensor:
        r"""Get tensor representation of this transformation

        Returns:
            Batch of homogeneous transformation matrices as tensor of shape ``(N, D, D)``.

        """
        scales = self.scales()
        if self.invert:
            scales = 1 / scales
        return U.scaling_transform(scales.expand(shape=[tuple(scales.shape)[0], self.ndim]))

    def extra_repr(self: IsotropicScaling) -> str:
        r"""Print current transformation."""
        s = super().extra_repr() + ", scales="
        if self.params is None:
            s += "undef"
        else:
            s += f"{self.scales().tolist()!r}"
        return s


class AnisotropicScaling(InvertibleParametricTransform, LinearTransform):
    r"""Anisotropic scaling."""

    def __init__(
        self: AnisotropicScaling,
        grid: Grid,
        groups: Optional[int] = None,
        params: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
    ) -> None:
        r"""Initialize transformation parameters.

        Args:
            grid: Grid domain on which transformation is defined.
            groups: Number of transformations. Must be either 1 or equal to batch size.
            params: Anisotropic scaling factors as percentages. This parameterization is
                adopted from MIRTK and ensures that rotation angles in degrees and scaling
                factors are within a similar range of magnitude which is useful for direct
                optimization of these parameters. If parameters are predicted by a callable
                ``torch.nn.Module``, different output activations may be chosen before
                converting these to percentages (i.e., multiplied by 100).

        """
        super().__init__(grid, groups=groups, params=params)

    @property
    def data_shape(self: AnisotropicScaling) -> list:
        r"""Get shape of transformation parameters tensor."""
        return tuple((self.ndim,))

    @paddle.no_grad()
    def reset_parameters(self: AnisotropicScaling) -> None:
        r"""Reset transformation parameters."""
        params = self.params
        if isinstance(params, paddle.Tensor):
            init_Constant = paddle.nn.initializer.Constant(value=1)
            init_Constant(params)

    def scales(self: AnisotropicScaling) -> paddle.Tensor:
        r"""Get scaling factors."""
        params = self.data()
        if self.has_parameters():
            params = params.sub(1).tanh().exp()
        return params

    @paddle.no_grad()
    def scales_(self: AnisotropicScaling, arg: paddle.Tensor) -> AnisotropicScaling:
        r"""Set transformation parameters from scaling factors."""
        if not isinstance(arg, paddle.Tensor):
            raise TypeError("AnisotropicScaling.scales() 'arg' must be tensor")
        shape = self.data_shape
        if arg.ndim != len(shape) + 1:
            raise ValueError(
                f"AnisotropicScaling.scales() 'arg' must be {len(shape) + 1}-dimensional tensor"
            )
        shape = (arg.shape[0],) + shape
        if arg.shape != shape:
            raise ValueError(f"AnisotropicScaling.scales() 'arg' must have shape {shape!r}")
        params = as_float_tensor(arg)
        if self.has_parameters():
            params = params.log().atanh().add(1)
            if params.isnan().astype("bool").any():
                raise ValueError("AnisotropicScaling.scales() 'arg' must be positive")
        self.data_(params)
        return self

    def tensor(self: AnisotropicScaling) -> paddle.Tensor:
        r"""Get tensor representation of this transformation

        Returns:
            Batch of homogeneous transformation matrices as tensor of shape ``(N, D, D)``.

        """
        scales = self.scales()
        if self.invert:
            scales = 1 / scales
        return U.scaling_transform(scales)

    def extra_repr(self: AnisotropicScaling) -> str:
        r"""Print current transformation."""
        s = super().extra_repr() + ", scales="
        if self.params is None:
            s += "undef"
        else:
            s += f"{self.scales().tolist()!r}"
        return s


class Shearing(InvertibleParametricTransform, LinearTransform):
    r"""Shear transformation."""

    def __init__(
        self: Shearing,
        grid: Grid,
        groups: Optional[int] = None,
        params: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
    ) -> None:
        r"""Initialize transformation parameters.

        Args:
            grid: Grid domain on which transformation is defined.
            groups: Number of transformations. Must be either 1 or equal to batch size.
            params: Shearing angles in degrees. This parameterization is adopted from MIRTK
                and ensures that rotation angles and scaling factors (percentage) are within
                a similar range of magnitude which is useful for direct optimization of these
                parameters. If parameters are predicted by a callable ``torch.nn.Module``,
                different output activations may be chosen before converting these to degrees.

        """
        if grid.ndim < 2 or grid.ndim > 3:
            raise ValueError("Shearing() 'grid' must be 2- or 3-dimensional'")
        super().__init__(grid, groups=groups, params=params)

    @property
    def data_shape(self: Shearing) -> list:
        r"""Get shape of transformation parameters tensor."""
        return tuple((self.nangles,))

    @property
    def nangles(self: Shearing) -> int:
        r"""Number of shear angles."""
        return 1 if self.ndim == 2 else self.ndim

    def angles(self: Shearing) -> paddle.Tensor:
        r"""Get shear angles in radians."""
        params = self.data()
        if self.has_parameters():
            params = params.tanh().mul(math.pi / 4)
        return params

    @paddle.no_grad()
    def angles_(self: Shearing, arg: paddle.Tensor) -> Shearing:
        r"""Set transformation parameters from shear angles in radians."""
        if not isinstance(arg, paddle.Tensor):
            raise TypeError("Shearing.angles_() 'arg' must be tensor")
        shape = self.data_shape
        if arg.ndim != len(shape) + 1:
            raise ValueError(
                f"Shearing.angles_() 'arg' must be {len(shape) + 1}-dimensional tensor"
            )
        shape = (arg.shape[0],) + shape
        if arg.shape != shape:
            raise ValueError(f"Shearing.angles_() 'arg' must have shape {shape!r}")
        params = as_float_tensor(arg)
        if self.has_parameters():
            params = params.mul(4 / math.pi).atanh()
            if params.isnan().astype("bool").any():
                raise ValueError("Shear 'angles' must be in range [-pi/4, pi/4]")
        self.data_(params)
        return self

    def tensor(self: Shearing) -> paddle.Tensor:
        r"""Get tensor representation of this transformation

        Returns:
            Batch of homogeneous transformation matrices as tensor of shape ``(N, D, D)``.

        """
        mat = U.shear_matrix(self.angles())
        if self.invert:
            mat = paddle.linalg.inv(x=mat)
        return mat

    def extra_repr(self: Shearing) -> str:
        r"""Print current transformation."""
        s = super().extra_repr() + "angles="
        if self.params is None:
            s += "undef"
        else:
            s += f"{self.angles().tolist()!r}"
        return s


class RigidTransform(SequentialTransform):
    r"""Rigid transformation."""

    def __init__(
        self: RigidTransform,
        grid: Grid,
        groups: Optional[int] = None,
        rotation: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
        translation: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
    ) -> None:
        r"""Initialize transformation parameters.

        Args:
            grid: Domain with respect to which transformation is defined.
            groups: Number of transformations ``N``.
            rotation: Parameters of ``EulerRotation``.
            translation: Parameters of ``Translation``.

        """
        transforms = OrderedDict()
        transforms["rotation"] = EulerRotation(grid, groups=groups, params=rotation)
        transforms["translation"] = Translation(grid, groups=groups, params=translation)
        super().__init__(transforms)

    @property
    def rotation(self) -> EulerRotation:
        return self._transforms["rotation"]

    @property
    def translation(self) -> Translation:
        return self._transforms["translation"]


class RigidQuaternionTransform(SequentialTransform):
    r"""Rigid transformation parameterized by rotation quaternion."""

    def __init__(
        self: RigidQuaternionTransform,
        grid: Grid,
        groups: Optional[int] = None,
        rotation: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
        translation: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
    ) -> None:
        r"""Initialize transformation parameters.

        Args:
            grid: Domain with respect to which transformation is defined.
            groups: Number of transformations ``N``.
            rotation: Parameters of ``QuaternionRotation``.
            translation: Parameters of ``Translation``.

        """
        transforms = OrderedDict()
        transforms["rotation"] = QuaternionRotation(grid, groups=groups, params=rotation)
        transforms["translation"] = Translation(grid, groups=groups, params=translation)
        super().__init__(transforms)

    @property
    def rotation(self) -> QuaternionRotation:
        return self._transforms["rotation"]

    @property
    def translation(self) -> Translation:
        return self._transforms["translation"]


class SimilarityTransform(SequentialTransform):
    r"""Similarity transformation with isotropic scaling."""

    def __init__(
        self: SimilarityTransform,
        grid: Grid,
        groups: Optional[int] = None,
        scaling: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
        rotation: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
        translation: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
    ) -> None:
        r"""Initialize transformation parameters.

        Args:
            grid: Domain with respect to which transformation is defined.
            groups: Number of transformations ``N``.
            scaling: Parameters of ``IsotropicScaling``.
            rotation: Parameters of ``EulerRotation``.
            translation: Parameters of ``Translation``.

        """
        transforms = OrderedDict()
        transforms["scaling"] = IsotropicScaling(grid, groups=groups, params=scaling)
        transforms["rotation"] = EulerRotation(grid, groups=groups, params=rotation)
        transforms["translation"] = Translation(grid, groups=groups, params=translation)
        super().__init__(transforms)

    @property
    def rotation(self) -> EulerRotation:
        return self._transforms["rotation"]

    @property
    def scaling(self) -> IsotropicScaling:
        return self._transforms["scaling"]

    @property
    def translation(self) -> Translation:
        return self._transforms["translation"]


class AffineTransform(SequentialTransform):
    r"""Affine transformation without shearing."""

    def __init__(
        self: AffineTransform,
        grid: Grid,
        groups: Optional[int] = None,
        scaling: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
        rotation: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
        translation: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
    ) -> None:
        r"""Initialize transformation parameters.

        Args:
            grid: Domain with respect to which transformation is defined.
            groups: Number of transformations ``N``.
            scaling: Parameters of ``AnisotropicScaling``.
            rotation: Parameters of ``EulerRotation``.
            translation: Parameters of ``Translation``.

        """
        transforms = OrderedDict()
        transforms["scaling"] = AnisotropicScaling(grid, groups=groups, params=scaling)
        transforms["rotation"] = EulerRotation(grid, groups=groups, params=rotation)
        transforms["translation"] = Translation(grid, groups=groups, params=translation)
        super().__init__(transforms)

    @property
    def rotation(self) -> EulerRotation:
        return self._transforms["rotation"]

    @property
    def scaling(self) -> AnisotropicScaling:
        return self._transforms["scaling"]

    @property
    def translation(self) -> Translation:
        return self._transforms["translation"]


class FullAffineTransform(SequentialTransform):
    r"""Affine transformation including shearing."""

    def __init__(
        self: FullAffineTransform,
        grid: Grid,
        groups: Optional[int] = None,
        scaling: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
        shearing: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
        rotation: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
        translation: Optional[Union[bool, paddle.Tensor, Callable[..., paddle.Tensor]]] = True,
    ) -> None:
        r"""Initialize transformation parameters.

        Args:
            grid: Domain with respect to which transformation is defined.
            groups: Number of transformations ``N``.
            scaling: Parameters of ``AnisotropicScaling``.
            shearing: Parameters of ``Shearing``.
            rotation: Parameters of ``EulerRotation``.
            translation: Parameters of ``Translation``.

        """
        transforms = OrderedDict()
        transforms["scaling"] = AnisotropicScaling(grid, groups=groups, params=scaling)
        transforms["shearing"] = Shearing(grid, groups=groups, params=shearing)
        transforms["rotation"] = EulerRotation(grid, groups=groups, params=rotation)
        transforms["translation"] = Translation(grid, groups=groups, params=translation)
        super().__init__(transforms)

    @property
    def rotation(self) -> EulerRotation:
        return self._transforms["rotation"]

    @property
    def scaling(self) -> AnisotropicScaling:
        return self._transforms["scaling"]

    @property
    def shearing(self) -> Shearing:
        return self._transforms["shearing"]

    @property
    def translation(self) -> Translation:
        return self._transforms["translation"]
