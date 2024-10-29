r"""Losses for image, point set, and/or surface registration.

Classes representing loss terms defined by this package are derived from ``paddle.nn.Layer``
and follow a stateful object oriented design. The underlying functions implementing these loss
terms are defined in the :mod:`.functional` module following a stateless functional API.

The following import statement can be used to access the functional API:

.. code-block:: python

    import deepali.losses.functional as L

"""
import sys
from typing import Any

import paddle
from paddle.nn import Layer

from .base import BSplineLoss
from .base import DisplacementLoss
from .base import NormalizedPairwiseImageLoss
from .base import PairwiseImageLoss
from .base import ParamsLoss
from .base import PointSetDistance
from .base import RegistrationLoss
from .base import RegistrationLosses
from .base import RegistrationResult
from .bspline import BSplineBending
from .bspline import BSplineBendingEnergy
from .flow import BE
from .flow import TV
from .flow import Bending
from .flow import BendingEnergy
from .flow import Curvature
from .flow import Diffusion
from .flow import Divergence
from .flow import Elasticity
from .flow import TotalVariation
from .image import DSC
from .image import LCC
from .image import LNCC
from .image import MAE
from .image import MI
from .image import MSE
from .image import NCC
from .image import NMI
from .image import SLCC
from .image import SSD
from .image import WLCC
from .image import Dice
from .image import HuberImageLoss
from .image import L1ImageLoss
from .image import L2ImageLoss
from .image import PatchLoss
from .image import PatchwiseImageLoss
from .image import SmoothL1ImageLoss
from .params import L1_Norm
from .params import L1Norm
from .params import L2_Norm
from .params import L2Norm
from .params import Sparsity
from .pointset import CPD
from .pointset import LPD
from .pointset import ClosestPointDistance
from .pointset import LandmarkPointDistance

__all__ = (
    # Base types
    "BSplineLoss",
    "DisplacementLoss",
    "NormalizedPairwiseImageLoss",
    "PairwiseImageLoss",
    "PatchwiseImageLoss",
    "PatchLoss",
    "ParamsLoss",
    "PointSetDistance",
    "RegistrationLoss",
    "RegistrationLosses",
    "RegistrationResult",
    "is_pairwise_image_loss",
    "is_displacement_loss",
    "is_pointset_distance",
    # Loss types
    "BE",
    "Bending",
    "BendingEnergy",
    "BSplineBending",
    "BSplineBendingEnergy",
    "ClosestPointDistance",
    "CPD",
    "Curvature",
    "Dice",
    "Diffusion",
    "Divergence",
    "DSC",
    "Elasticity",
    "HuberImageLoss",
    "L1ImageLoss",
    "L2ImageLoss",
    "L1Norm",
    "L1_Norm",
    "L2Norm",
    "L2_Norm",
    "LandmarkPointDistance",
    "LPD",
    "LCC",
    "LNCC",
    "MAE",
    "MI",
    "MSE",
    "NCC",
    "NMI",
    "SLCC",
    "SmoothL1ImageLoss",
    "Sparsity",
    "SSD",
    "TotalVariation",
    "TV",
    "WLCC",
    # Factory function
    "create_loss",
    "new_loss",
)


def is_pairwise_image_loss(arg: Any) -> bool:
    r"""Check if given argument is name or instance of pairwise image loss."""
    return is_loss_of_type(PairwiseImageLoss, arg)


def is_displacement_loss(arg: Any) -> bool:
    r"""Check if given argument is name or instance of displacement field loss."""
    return is_loss_of_type(DisplacementLoss, arg)


def is_pointset_distance(arg: Any) -> bool:
    r"""Check if given argument is name or instance of point set distance."""
    return is_loss_of_type(PointSetDistance, arg)


def is_loss_of_type(base, arg: Any) -> bool:
    r"""Check if given argument is name or instance of pairwise image loss."""
    cls = None
    if isinstance(arg, str):
        cls = getattr(sys.modules[__name__], arg, None)
    elif type(arg) is type:
        cls = arg
    elif arg is not None:
        cls = type(arg)
    if cls is not None:
        bases = list(cls.__bases__)
        while bases:
            b = bases.pop()
            if b is base:
                return True
            bases.extend(b.__bases__)
    return False


def new_loss(name: str, *args, **kwargs) -> paddle.nn.Layer:
    r"""Initialize new loss module.

    Args:
        name: Name of loss type.
        args: Loss arguments.
        kwargs: Loss keyword arguments.

    Returns:
        New loss module.

    """
    cls = getattr(sys.modules[__name__], name, None)
    if cls is None:
        raise ValueError(f"new_loss() unknown loss {name}")
    if cls is Layer or not issubclass(cls, Layer):
        raise TypeError(f"new_loss() '{name}' is not a subclass of paddle.nn.Layer")
    return cls(*args, **kwargs)


create_loss = new_loss
