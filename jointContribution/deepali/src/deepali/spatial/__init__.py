r"""Spatial coordinate and input data transformation modules.

.. hint::

    The spatial transforms defined by this library can be used to implement co-registration
    approaches based on traditional optimization as well as those based on machine learning
    (amortized optimization).

A spatial transformation maps points from a target domain defined with respect to the unit cube
of a target sampling grid, to points defined with respect to the same domain, i.e., the domain
and codomain of the spatial coordinate map are identical. In order to transform an image defined
with respect to a different sampling grid, this transformation has to be followed by a mapping
from target cube domain to source cube domain. This is done, for example, by the spatial transformer
implemented by :class:`.ImageTransformer`.

The ``forward()`` method of a :class:`.SpatialTransform` can be used to spatially transform any set
of points defined with respect to the grid domain of the spatial transformation, including in particular
a tensor of shape ``(N, M, D)``, i.e., a batch of ``N`` point sets with cardinality ``M``. It can
also be applied to a tensor of grid points of shape ``(N, ..., X, D)`` regardless if the grid points
are located at the undeformed grid positions or those of an already deformed grid. In case of a
non-rigid deformation, the point displacements are by default sampled at the input points. The sampled
flow vectors :math:`u` are then added to the input points :math:`x`, producing the output :math:`y = x + u(x)`.
If the boolean flag ``grid=True`` is passed to the :meth:`.SpatialTransform.forward` function, it is assumed
that the coordinates correspond to the positions of undeformed spatial grid points with domain equal to the
domain of the transformation. In this special case, a simple interpolation to resize the vector field to the
size of the input tensor is used. In case of a linear transformation, :math:`y = Ax + t`.

The coordinate domain is :attr:`.Axes.CUBE_CORNERS` if ``grid.align_corners() == True`` (default),
and :attr:`.Axes.CUBE` otherwise.

"""

import sys
from typing import Any
from typing import Optional

from deepali.core.grid import Grid

# Base classes for type comparison and annotation
from .base import LinearTransform  # noqa
from .base import NonRigidTransform  # noqa
from .base import ReadOnlyParameters  # noqa
from .base import SpatialTransform

# Free-form deformations
from .bspline import BSplineTransform  # noqa
from .bspline import FreeFormDeformation
from .bspline import StationaryVelocityFreeFormDeformation

# Composite coordinate transformations
from .composite import CompositeTransform  # noqa
from .composite import MultiLevelTransform  # noqa
from .composite import SequentialTransform  # noqa

# Configurable generic transformation
from .generic import GenericSpatialTransform  # noqa
from .generic import TransformConfig  # noqa
from .generic import affine_first  # noqa
from .generic import has_affine_component  # noqa
from .generic import has_nonrigid_component  # noqa
from .generic import nonrigid_components  # noqa
from .generic import transform_components  # noqa

# Spatial transformers based on a coordinate transformation
from .image import ImageTransform  # noqa (deprecated)

# Composite linear transformations
# Elemental linear transformations
from .linear import AffineTransform
from .linear import AnisotropicScaling
from .linear import EulerRotation
from .linear import FullAffineTransform
from .linear import HomogeneousTransform
from .linear import IsotropicScaling  # noqa
from .linear import QuaternionRotation
from .linear import RigidQuaternionTransform
from .linear import RigidTransform
from .linear import Shearing
from .linear import SimilarityTransform
from .linear import Translation  # noqa

# Non-rigid deformations
from .nonrigid import DenseVectorFieldTransform  # noqa
from .nonrigid import DisplacementFieldTransform
from .nonrigid import StationaryVelocityFieldTransform

# Parametric transformation mix-in
from .parametric import ParametricTransform  # noqa
from .transformer import ImageTransformer  # noqa
from .transformer import PointSetTransformer  # noqa
from .transformer import SpatialTransformer  # noqa

# Aliases
Affine = AffineTransform
AffineWithShearing = FullAffineTransform
Disp = DisplacementFieldTransform
DispField = DisplacementFieldTransform
DDF = DisplacementFieldTransform
DVF = DisplacementFieldTransform
FFD = FreeFormDeformation
FullAffine = FullAffineTransform
MatrixTransform = HomogeneousTransform
Quaternion = QuaternionRotation
Rigid = RigidTransform
RigidQuaternion = RigidQuaternionTransform
Rotation = EulerRotation
Scaling = AnisotropicScaling
ShearTransform = Shearing
Similarity = SimilarityTransform
SVF = StationaryVelocityFieldTransform
SVField = StationaryVelocityFieldTransform
SVFFD = StationaryVelocityFreeFormDeformation


LINEAR_TRANSFORMS = (
    "Affine",
    "AffineTransform",
    "AffineWithShearing",
    "AnisotropicScaling",
    "BSplineTransform",
    "EulerRotation",
    "IsotropicScaling",
    "FullAffine",
    "FullAffineTransform",
    "HomogeneousTransform",
    "MatrixTransform",
    "Quaternion",
    "QuaternionRotation",
    "Rigid",
    "RigidTransform",
    "RigidQuaternion",
    "RigidQuaternionTransform",
    "Rotation",
    "Scaling",
    "Shearing",
    "ShearTransform",
    "Similarity",
    "SimilarityTransform",
    "Translation",
)

NONRIGID_TRANSFORMS = (
    "Disp",
    "DispField",
    "DisplacementFieldTransform",
    "DDF",
    "DVF",
    "FFD",
    "FreeFormDeformation",
    "StationaryVelocityFieldTransform",
    "StationaryVelocityFreeFormDeformation",
    "SVF",
    "SVField",
    "SVFFD",
)

COMPOSITE_TRANSFORMS = (
    "MultiLevelTransform",
    "SequentialTransform",
)

__all__ = (
    (
        "CompositeTransform",
        "DenseVectorFieldTransform",
        "GenericSpatialTransform",
        "ImageTransform",
        "ImageTransformer",
        "LinearTransform",
        "NonRigidTransform",
        "ParametricTransform",
        "PointSetTransformer",
        "ReadOnlyParameters",
        "SpatialTransform",
        "SpatialTransformer",
        "TransformConfig",
        "affine_first",
        "has_affine_component",
        "has_nonrigid_component",
        "is_linear_transform",
        "is_nonrigid_transform",
        "is_spatial_transform",
        "new_spatial_transform",
        "nonrigid_components",
        "transform_components",
    )
    + COMPOSITE_TRANSFORMS
    + LINEAR_TRANSFORMS
    + NONRIGID_TRANSFORMS
)


def is_spatial_transform(arg: Any) -> bool:
    r"""Whether given object or named transformation is a transformation type.

    Args:
        arg: Name of type or object.

    Returns:
        Whether type of ``arg`` object or name of type is a transformation model.

    """
    if isinstance(arg, str):
        return arg in LINEAR_TRANSFORMS or arg in NONRIGID_TRANSFORMS
    return isinstance(arg, SpatialTransform)


def is_linear_transform(arg: Any) -> bool:
    r"""Whether given object is a linear transformation type.

    Args:
        arg: Name of type or object.

    Returns:
        Whether type of ``arg`` object or name of type is a linear transformation.

    """
    if isinstance(arg, str):
        return arg in LINEAR_TRANSFORMS
    if isinstance(arg, SpatialTransform):
        return arg.linear
    return False


def is_nonrigid_transform(arg: Any) -> bool:
    r"""Whether given object is a non-rigid transformation type.

    Args:
        arg: Name of type or object.

    Returns:
        Whether type of ``arg`` object or name of type is a non-rigid transformation.

    """
    if isinstance(arg, str):
        return arg in NONRIGID_TRANSFORMS
    if isinstance(arg, SpatialTransform):
        return arg.nonrigid
    return False


def new_spatial_transform(
    name: str, grid: Grid, groups: Optional[int] = None, **kwargs
) -> SpatialTransform:
    r"""Initialize new transformation model of named type.

    Args:
        name: Name of transformation model.
        grid: Grid of transformation domain.
        groups: Number of transformations.
        kwargs: Optional keyword arguments of transformation model.

    Returns:
        New transformation module with optimizable parameters.

    """
    cls = getattr(sys.modules[__name__], name, None)
    if cls is not None and (name in LINEAR_TRANSFORMS or name in NONRIGID_TRANSFORMS):
        return cls(grid, groups=groups, **kwargs)
    raise ValueError(f"new_spatial_transform() 'name={name}' is not a valid transformation type")
