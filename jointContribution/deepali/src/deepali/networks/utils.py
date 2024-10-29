import paddle
from deepali.core.nnutils import conv_output_size
from deepali.core.nnutils import conv_transposed_output_size
from deepali.core.nnutils import pad_output_size
from deepali.core.nnutils import pool_output_size
from deepali.core.nnutils import unpool_output_size
from deepali.core.nnutils import upsample_output_size
from deepali.core.typing import ScalarOrTuple
from deepali.utils import paddle_aux  # noqa
from paddle import Tensor

from .blocks import SkipConnection
from .layers import Pad
from .layers import is_activation
from .layers import is_norm_layer


def module_output_size(module: paddle.nn.Layer, in_size: ScalarOrTuple[int]) -> ScalarOrTuple[int]:
    r"""Calculate spatial size of output tensor after the given module is applied."""
    if not isinstance(module, paddle.nn.Layer):
        raise TypeError("module_output_size() 'module' must be paddle.nn.Layer subclass")
    output_size = getattr(module, "output_size", None)
    if callable(output_size):
        return output_size(in_size)
    if output_size is not None:
        device = paddle.CPUPlace()
        m: Tensor = paddle.atleast_1d(paddle.to_tensor(data=in_size, dtype="int32", place=device))
        if m.ndim != 1:
            raise ValueError("module_output_size() 'in_size' must be scalar or sequence")
        ndim = tuple(m.shape)[0]
        s: Tensor = paddle.atleast_1d(
            paddle.to_tensor(data=output_size, dtype="int32", place=device)
        )
        if s.ndim != 1 or tuple(s.shape)[0] not in (1, ndim):
            raise ValueError(
                f"module_output_size() 'module.output_size' must be scalar or sequence of length {ndim}"
            )
        n = s.expand(shape=ndim)
        if isinstance(in_size, int):
            return n.item()
        return tuple(n.tolist())
    # Network blocks
    if isinstance(module, paddle.nn.Sequential):
        size = in_size
        for m in module:
            size = module_output_size(m, size)
        return size
    if isinstance(module, SkipConnection):
        return module_output_size(module.func, in_size)
    # Convolutional layers
    if isinstance(module, (paddle.nn.Conv1D, paddle.nn.Conv2D, paddle.nn.Conv3D)):
        return conv_output_size(
            in_size,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
        )
    if isinstance(
        module, (paddle.nn.Conv1DTranspose, paddle.nn.Conv2DTranspose, paddle.nn.Conv3DTranspose)
    ):
        return conv_transposed_output_size(
            in_size,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            output_padding=module.output_padding,
            dilation=module.dilation,
        )
    # Pooling layers
    if isinstance(
        module,
        (
            paddle.nn.AvgPool1D,
            paddle.nn.AvgPool2D,
            paddle.nn.AvgPool3D,
            paddle.nn.MaxPool1D,
            paddle.nn.MaxPool2D,
            paddle.nn.MaxPool3D,
        ),
    ):
        return pool_output_size(
            in_size,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode,
        )
    if isinstance(
        module,
        (
            paddle.nn.AdaptiveAvgPool1D,
            paddle.nn.AdaptiveAvgPool2D,
            paddle.nn.AdaptiveAvgPool3D,
            paddle.nn.AdaptiveMaxPool1D,
            paddle.nn.AdaptiveMaxPool2D,
            paddle.nn.AdaptiveMaxPool3D,
        ),
    ):
        return module.output_size
    if isinstance(module, (paddle.nn.MaxUnPool1D, paddle.nn.MaxUnPool2D, paddle.nn.MaxUnPool3D)):
        return unpool_output_size(
            in_size, kernel_size=module.kernel_size, stride=module.stride, padding=module.padding
        )
    if isinstance(module, Pad):
        raise NotImplementedError()
    if isinstance(
        module,
        (
            paddle.nn.Pad1D,
            paddle.nn.Pad2D,
            paddle.nn.Pad1D,
            paddle.nn.Pad2D,
            paddle.nn.Pad3D,
            paddle.nn.ZeroPad2D,
            paddle.nn.Pad1D,
            paddle.nn.Pad2D,
            paddle.nn.Pad3D,
        ),
    ):
        return pad_output_size(in_size, module.padding)
    # Upsampling
    if isinstance(module, paddle.nn.Upsample):
        return upsample_output_size(in_size, size=module.size, scale_factor=module.scale_factor)
    # Activation functions
    if is_activation(module) or isinstance(
        module,
        (
            paddle.nn.ELU,
            paddle.nn.Hardshrink,
            paddle.nn.Hardsigmoid,
            paddle.nn.Hardtanh,
            paddle.nn.Hardswish,
            paddle.nn.LeakyReLU,
            paddle.nn.LogSigmoid,
            paddle.nn.LogSoftmax,
            paddle.nn.PReLU,
            paddle.nn.ReLU,
            paddle.nn.ReLU6,
            paddle.nn.RReLU,
            paddle.nn.SELU,
            paddle.nn.CELU,
            paddle.nn.GELU,
            paddle.nn.Sigmoid,
            paddle.nn.Softmax,
            paddle.nn.Softmax,
            paddle_aux.Softmin,
            paddle.nn.Softplus,
            paddle.nn.Softshrink,
            paddle.nn.Softsign,
            paddle.nn.Tanh,
            paddle.nn.Tanhshrink,
            paddle.nn.ThresholdedReLU,
        ),
    ):
        return in_size
    # Normalization layers
    if is_norm_layer(module) or isinstance(
        module,
        (
            paddle.nn.BatchNorm1D,
            paddle.nn.BatchNorm2D,
            paddle.nn.BatchNorm3D,
            paddle.nn.SyncBatchNorm,
            paddle.nn.GroupNorm,
            paddle.nn.InstanceNorm1D,
            paddle.nn.InstanceNorm2D,
            paddle.nn.InstanceNorm3D,
            paddle.nn.LayerNorm,
            paddle.nn.LocalResponseNorm,
        ),
    ):
        return in_size
    # Dropout layers
    if isinstance(
        module,
        (paddle.nn.AlphaDropout, paddle.nn.Dropout, paddle.nn.Dropout2D, paddle.nn.Dropout3D),
    ):
        return in_size
    # Not implemented or invalid type
    if isinstance(module, (paddle.nn.LayerDict, paddle.nn.LayerList)):
        raise TypeError(
            "module_output_size() order of modules in ModuleDict or ModuleList is undetermined"
        )
    if isinstance(module, (paddle.nn.ParameterList)):
        raise TypeError(
            "module_output_size() 'module' cannot be paddle.nn.ParameterDict or paddle.nn.ParameterList"
        )
    raise NotImplementedError(
        f"module_output_size() not implemented for 'module' of type {type(module)}"
    )
