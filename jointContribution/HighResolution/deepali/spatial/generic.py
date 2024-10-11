from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Union

import paddle

from ..core.affine import euler_rotation_angles
from ..core.affine import euler_rotation_matrix
from ..core.config import DataclassConfig
from ..core.grid import Grid
from ..core.linalg import quaternion_to_rotation_matrix
from ..core.linalg import rotation_matrix_to_quaternion
from ..core.types import ScalarOrTuple
from .bspline import FreeFormDeformation
from .bspline import StationaryVelocityFreeFormDeformation
from .composite import SequentialTransform
from .linear import AnisotropicScaling
from .linear import EulerRotation
from .linear import HomogeneousTransform
from .linear import QuaternionRotation
from .linear import Shearing
from .linear import Translation
from .nonrigid import DisplacementFieldTransform
from .nonrigid import StationaryVelocityFieldTransform

ParamsDict = Mapping[str, paddle.Tensor]
AFFINE_NAMES = {
    "A": "affine",
    "K": "shearing",
    "T": "translation",
    "R": "rotation",
    "S": "scaling",
    "Q": "quaternion",
}
"""Names of elementary affine transformation child modules.

The dictionary key is the letter used in :attr:`TransformConfig.affine_model`, i.e.,

- ``A``: ``"affine"``
- ``K``: ``"shearing"``
- ``T``: ``"translation"``
- ``R``: ``"rotation"``
- ``S``: ``"scaling"``
- ``Q``: ``"quaternion"``

"""
AFFINE_TRANSFORMS = {
    "A": HomogeneousTransform,
    "K": Shearing,
    "T": Translation,
    "R": EulerRotation,
    "S": AnisotropicScaling,
    "Q": QuaternionRotation,
}
"""Types of elementary affine transformations.

The dictionary key is the letter used in :attr:`TransformConfig.affine_model`, i.e.,

- ``A``: :class:`.HomogeneousTransform`
- ``K``: :class:`.Shearing`
- ``T``: :class:`.Translation`
- ``R``: :class:`.EulerRotation`
- ``S``: :class:`.AnisotropicScaling`
- ``Q``: :class:`.QuaternionRotation`

"""
NONRIGID_TRANSFORMS = {
    "DDF": DisplacementFieldTransform,
    "FFD": FreeFormDeformation,
    "SVF": StationaryVelocityFieldTransform,
    "SVFFD": StationaryVelocityFreeFormDeformation,
}
"""Types of non-rigid transformations.

The dictionary key is the string used in :attr:`TransformConfig.transform`, i.e.,

- ``DDF``: :class:`.DisplacementFieldTransform`
- ``FFD``: :class:`.FreeFormDeformation`
- ``SVF``: :class:`.StationaryVelocityFieldTransform`
- ``SVFFD``: :class:`.StationaryVelocityFreeFormDeformation`

"""
VALID_COMPONENTS = ("Affine",) + tuple(NONRIGID_TRANSFORMS.keys())
"""Valid transformation names in :attr:`TransformConfig.transform` string value.

This includes "Affine" and all keys of the :data:`NONRIGID_TRANSFORMS` dictionary.

"""


def transform_components(model: str) -> List[str]:
    """Non-rigid component of transformation or ``None`` if it is a linear transformation."""
    return model.split(" o ")


def valid_transform_model(
    model: str, max_affine: Optional[int] = None, max_nonrigid: Optional[int] = None
) -> bool:
    """Whether given string denotes a valid transformation model."""
    components = transform_components(model)
    num_affine = 0
    num_nonrigid = 0
    for component in components:
        if component not in VALID_COMPONENTS:
            return False
        if component == "Affine":
            num_affine += 1
        else:
            num_nonrigid += 1
    if len(components) < 1:
        return False
    if max_affine is not None and num_affine > max_affine:
        return False
    if max_nonrigid is not None and num_nonrigid > max_nonrigid:
        return False
    return True


def has_affine_component(model: str) -> bool:
    """Whether transformation model includes an affine component."""
    return "Affine" in transform_components(model)


def has_nonrigid_component(model: str) -> bool:
    """Whether transformation model includes a non-rigid component."""
    return nonrigid_components(model)


def nonrigid_components(model: str) -> List[str]:
    """Non-rigid components of transformation model."""
    return [comp for comp in transform_components(model) if comp in NONRIGID_TRANSFORMS]


def affine_first(model: str) -> bool:
    """Whether transformation applies affine component first."""
    components = transform_components(model)
    assert components, "must contain at least one transformation component"
    return components[-1] == "Affine"


@dataclass
class TransformConfig(DataclassConfig):
    """Configuration of generic spatial transformation model."""

    transform: str = "Affine o SVF"
    """String encoding of spatial transformation model to use.

    The linear transforms making up the ``Affine`` component are defined by :attr:`affine_model`.

    The non-rigid component can be one of the following:

    - ``DDF``: :class:`.DisplacementFieldTransform`
    - ``FFD``: :class:`.FreeFormDeformation`
    - ``SVF``: :class:`.StationaryVelocityFieldTransform`
    - ``SVFFD``: :class:`.StationaryVelocityFreeFormDeformation`

    """
    affine_model: str = "TRS"
    """String encoding of composition of elementary linear transformations.

    The string value of this configuration entry can be in one of two forms:

    - Matrix notation: Each letter is a factor in the sequence of matrix-matrix products.
    - Function composition: Use deliminator " o " between transformations to denote composition.

    Valid elementary linear transform identifiers are:

    - ``A``: :class:`.HomogeneousTransform`
    - ``K``: :class:`.Shearing`
    - ``T``: :class:`.Translation`
    - ``R``: :class:`.EulerRotation`
    - ``S``: :class:`.AnisotropicScaling`
    - ``Q``: :class:`.QuaternionRotation`

    """
    rotation_model: str = "ZXZ"
    """Order of elementary Euler rotations.

    This configuration value is only used when :attr:`affine_model` contains an :class:`EulerRotation`
    denoted by letter "R". Valid values are "ZXZ", "XZX", ... (cf. :func:`.core.affine.euler_rotation_matrix`).

    """
    control_point_spacing: ScalarOrTuple[int] = 1
    """Control point spacing of non-rigid transformations.

    The spacing must be given in voxel units of the grid domain with respect to
    which the transformations are defined.

    """
    scaling_and_squaring_steps: int = 6
    """Number of scaling and squaring steps in case of a stationary velocity field transform."""
    flip_grid_coords: bool = False
    """Whether predicted transformation parameters are with respect to a grid
        with point coordinates in the order (..., x) instead of (x, ...)."""

    def _finalize(self, parent: Path) -> None:
        """Finalize parameters after loading these from input file."""
        super()._finalize(parent)


class GenericSpatialTransform(SequentialTransform):
    """Configurable generic spatial transformation."""

    def __init__(
        self,
        grid: Grid,
        params: Optional[Union[bool, Callable[..., ParamsDict], ParamsDict]] = True,
        config: Optional[TransformConfig] = None,
    ) -> None:
        """Initialize spatial transformation."""
        if (
            params not in (None, False, True)
            and not callable(params)
            and not isinstance(params, Mapping)
        ):
            raise TypeError(
                f"{type(self).__name__}() 'params' must be bool, callable, dict, or None"
            )
        if config is None:
            config = getattr(params, "config", None)
            if config is None:
                raise AssertionError(
                    f"{type(self).__name__}() 'config' or 'params.config' required"
                )
            if not isinstance(config, TransformConfig):
                raise TypeError(
                    f"{type(self).__name__}() 'params.config' must be TransformConfig"
                )
        elif not isinstance(config, TransformConfig):
            raise TypeError(f"{type(self).__name__}() 'config' must be TransformConfig")
        if not valid_transform_model(config.transform, max_affine=1, max_nonrigid=1):
            raise ValueError(
                f"{type(self).__name__}() 'config.transform' invalid or not supported: {config.transform}"
            )
        modules = paddle.nn.LayerDict()
        if has_affine_component(config.transform):
            for key in reversed(config.affine_model.replace(" o ", "")):
                key = key.upper()
                if key not in AFFINE_TRANSFORMS:
                    raise ValueError(
                        f"{type(self).__name__}() invalid character '{key}' in 'config.affine_model'"
                    )
                name = AFFINE_NAMES[key]
                if name in modules:
                    raise NotImplementedError(
                        f"{type(self).__name__}() 'config.affine_model' must contain each elementary transform at most once, but encountered key '{key}' more than once."
                    )
                kwargs = dict(
                    grid=grid, params=params if isinstance(params, bool) else None
                )
                if key == "R":
                    kwargs["order"] = config.rotation_model
                modules[name] = AFFINE_TRANSFORMS[key](**kwargs)
        nonrigid_models = nonrigid_components(config.transform)
        if len(nonrigid_models) > 1:
            raise ValueError(
                f"{type(self).__name__}() 'config.transform' must contain at most one non-rigid component"
            )
        if nonrigid_models:
            nonrigid_model = nonrigid_models[0]
            nonrigid_params = params if isinstance(params, bool) else None
            nonrigid_kwargs = dict(grid=grid, params=nonrigid_params)
            NonRigidTransform = NONRIGID_TRANSFORMS[nonrigid_model]
            if nonrigid_model in ("DDF", "SVF") and config.control_point_spacing > 1:
                size = grid.size_tensor()
                stride = paddle.to_tensor(data=config.control_point_spacing).to(size)
                size = size.div(stride).ceil().astype(dtype="int64")
                nonrigid_kwargs["grid"] = grid.resize(size)
            if nonrigid_model == "SVF":
                nonrigid_kwargs["steps"] = config.scaling_and_squaring_steps
            if nonrigid_model in ("FFD", "SVFFD"):
                nonrigid_kwargs["stride"] = config.control_point_spacing
            _modules = paddle.nn.LayerDict(
                sublayers={"nonrigid": NonRigidTransform(**nonrigid_kwargs)}
            )
            if affine_first(config.transform):
                modules.update(_modules)
            else:
                _modules.update(modules)
                modules = _modules
        if isinstance(params, Mapping):
            for name, transform in self.named_transforms():
                transform.data_(params[name])
        super().__init__(grid, modules)
        self.config = config
        self.params = params if callable(params) else None

    def _data(self) -> Dict[str, paddle.Tensor]:
        """Get most recent transformation parameters."""
        if not self._transforms:
            return {}
        params = self.params
        if params is None:
            params = {}
            for name, transform in self.named_transforms():
                params[name] = transform.data()
            return params
        if isinstance(params, GenericSpatialTransform):
            return {}
        if callable(params):
            args, kwargs = self.condition()
            pred = params(*args, **kwargs)
            if not isinstance(pred, Mapping):
                raise TypeError(
                    f"{type(self).__name__} 'params' callable return value must be a Mapping"
                )
        elif isinstance(params, Mapping):
            pred = params
        else:
            raise TypeError(
                f"{type(self).__name__} 'params' attribute must be a callable, Mapping, linked GenericSpatialTransform, or None"
            )
        data = {}
        flip_grid_coords = self.config.flip_grid_coords
        if "affine" in self._transforms:
            matrix = pred["affine"]
            assert isinstance(matrix, paddle.Tensor)
            assert matrix.ndim >= 2
            D = tuple(matrix.shape)[-2]
            assert tuple(matrix.shape)[-1] == D + 1
            if flip_grid_coords:
                matrix[(...), :D, :D] = matrix[(...), :D, :D].flip(axis=(1, 2))
                matrix[(...), :D, (-1)] = matrix[(...), :D, (-1)].flip(axis=-1)
            data["affine"] = matrix
        if "translation" in self._transforms:
            if "translation" in pred:
                offset = pred["translation"]
            else:
                offset = pred["offset"]
            assert isinstance(offset, paddle.Tensor)
            if flip_grid_coords:
                offset = offset.flip(axis=-1)
            data["translation"] = offset
        if "rotation" in self._transforms:
            if "rotation" in pred:
                angles = pred["rotation"]
            else:
                angles = pred["angles"]
            assert isinstance(angles, paddle.Tensor)
            if flip_grid_coords:
                rotmodel = self.config.rotation_model
                rotation = euler_rotation_matrix(angles, order=rotmodel).flip((1, 2))
                angles = euler_rotation_angles(rotation, order=rotmodel)
            data["rotation"] = angles
        if "scaling" in self._transforms:
            if "scaling" in pred:
                scales = pred["scaling"]
            else:
                scales = pred["scales"]
            assert isinstance(scales, paddle.Tensor)
            if flip_grid_coords:
                scales = scales.flip(axis=-1)
            data["scaling"] = scales
        if "quaternion" in self._transforms:
            q = pred["quaternion"]
            assert isinstance(q, paddle.Tensor)
            if flip_grid_coords:
                m = quaternion_to_rotation_matrix(q)
                m = m.flip(axis=(1, 2))
                q = rotation_matrix_to_quaternion(m)
            data["quaternion"] = q
        if "nonrigid" in self._transforms:
            if "nonrigid" in pred:
                vfield = pred["nonrigid"]
            else:
                vfield = pred["vfield"]
            assert isinstance(vfield, paddle.Tensor)
            if flip_grid_coords:
                vfield = vfield.flip(axis=1)
            data["nonrigid"] = vfield
        return data

    def inverse(
        self, link: bool = False, update_buffers: bool = False
    ) -> GenericSpatialTransform:
        """Get inverse of this transformation.

        Args:
            link: Whether the inverse transformation keeps a reference to this transformation.
                If ``True``, the ``update()`` function of the inverse function will not recompute
                shared parameters (e.g., parameters obtained by a callable neural network), but
                directly access the parameters from this transformation. Note that when ``False``,
                the inverse transformation will still share parameters, modules, and buffers with
                this transformation, but these shared tensors may be replaced by a call of ``update()``
                (which is implicitly called as pre-forward hook when ``__call__()`` is invoked).
            update_buffers: Whether buffers of inverse transformation should be updated after creating
                the shallow copy. If ``False``, the ``update()`` function of the returned inverse
                transformation has to be called before it is used.

        Returns:
            Shallow copy of this transformation which computes and applied the inverse transformation.
            The inverse transformation will share the parameters with this transformation. Not all
            transformations may implement this functionality.

        Raises:
            NotImplementedError: When a transformation does not support sharing parameters with its inverse.

        """
        inv = super().inverse(link=link, update_buffers=update_buffers)
        if link:
            inv.params = self
        return inv

    def update(self) -> GenericSpatialTransform:
        """Update transformation parameters."""
        if self.params is not None:
            params = self._data()
            for k, p in params.items():
                transform = self._transforms[k]
                transform.data_(p)
        super().update()
        return self
