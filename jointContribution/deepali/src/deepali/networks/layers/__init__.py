r"""Basic network modules, usually with learnable parameters."""

from deepali.modules import Pad
from deepali.modules import Reshape
from deepali.modules import View

from .acti import Activation
from .acti import ActivationArg
from .acti import ActivationFunc
from .acti import activation
from .acti import is_activation
from .conv import Conv1d
from .conv import Conv2d
from .conv import Conv3d
from .conv import ConvLayer
from .conv import ConvTranspose1d
from .conv import ConvTranspose2d
from .conv import ConvTranspose3d
from .conv import conv_module
from .conv import convolution
from .conv import is_conv_module
from .conv import is_convolution
from .join import JoinFunc
from .join import JoinLayer
from .join import join_func
from .lambd import LambdaFunc
from .lambd import LambdaLayer
from .linear import Linear
from .norm import NormArg
from .norm import NormFunc
from .norm import NormLayer
from .norm import is_batch_norm
from .norm import is_group_norm
from .norm import is_instance_norm
from .norm import is_norm_layer
from .norm import norm_layer
from .norm import normalization
from .pool import PoolArg
from .pool import PoolFunc
from .pool import PoolLayer
from .pool import pool_layer
from .pool import pooling
from .upsample import SubpixelUpsample
from .upsample import Upsample
from .upsample import UpsampleMode

__all__ = (
    # Basic layers
    "Pad",
    "Reshape",
    "View",
    # Non-linear activations
    "Activation",
    "ActivationArg",
    "ActivationFunc",
    "activation",
    "is_activation",
    # Convolutional layers
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "ConvLayer",
    "conv_module",
    "convolution",
    "is_convolution",
    "is_conv_module",
    # Merge network branches
    "JoinLayer",
    "JoinFunc",
    "join_func",
    # Fully connected layer
    "Linear",
    # Normalization layers
    "NormArg",
    "NormFunc",
    "NormLayer",
    "norm_layer",
    "normalization",
    "is_batch_norm",
    "is_group_norm",
    "is_instance_norm",
    "is_norm_layer",
    # Pooling layers
    "PoolArg",
    "PoolFunc",
    "PoolLayer",
    "pool_layer",
    "pooling",
    # Resampling layers
    "Upsample",
    "UpsampleMode",
    "SubpixelUpsample",
    # Misc layers
    "LambdaLayer",
    "LambdaFunc",
)
