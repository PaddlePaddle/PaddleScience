r"""Modules without learnable parameters.

This library defines subclasses of ``paddle.nn.Layer`` which expose the tensor operations
available in the :mod:`.core` library via a stateful functor object that can be used
in PyTorch models to perform predefined operations with in general no optimizable parameters.

"""

from .basic import GetItem
from .basic import Narrow
from .basic import Pad
from .basic import Reshape
from .basic import View
from .flow import Curl
from .flow import ExpFlow
from .image import BlurImage
from .image import FilterImage
from .image import GaussianConv
from .lambd import LambdaFunc
from .lambd import LambdaLayer
from .mixins import DeviceProperty
from .mixins import ReprWithCrossReferences
from .output import ToImmutableOutput
from .sample import AlignImage
from .sample import SampleImage
from .sample import TransformImage
from .utilities import remove_layers_in_state_dict
from .utilities import rename_layers_in_state_dict

__all__ = (
    "AlignImage",
    "BlurImage",
    "Curl",
    "DeviceProperty",
    "ExpFlow",
    "FilterImage",
    "GaussianConv",
    "GetItem",
    "LambdaFunc",
    "LambdaLayer",
    "Narrow",
    "Pad",
    "ReprWithCrossReferences",
    "Reshape",
    "SampleImage",
    "ToImmutableOutput",
    "TransformImage",
    "View",
    "remove_layers_in_state_dict",
    "rename_layers_in_state_dict",
)
