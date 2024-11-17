"""
Higher Order Convolution with CP decompositon
"""

# Author: Jean Kossaifi
# License: BSD 3 clause

import warnings

import numpy as np
import paddle
import paddle.nn as nn
import tensorly as tl

from ..factorized_tensors import CPTensor
from ..factorized_tensors import FactorizedTensor
from ..factorized_tensors import TTTensor
from ..functional.convolution import _get_factorized_conv

tl.set_backend("paddle")


def _ensure_list(order, value):
    """Ensures that `value` is a list of length `order`

    If `value` is an int, turns it into a list ``[value]*order``
    """
    if isinstance(value, int):
        return [value] * order
    assert len(value) == order
    return value


def _ensure_array(layers_shape, order, value, one_per_order=True):
    """Ensures that `value` is an array

    Parameters
    ----------
    layers_shape : tuple
        shape of the layer (n_weights)
    order : int
        order of the convolutional layer
    value : np.ndarray or int
        value to be checked
    one_per_order : bool, optional
        if true, then we must have one value per mode of the convolution
        otherwise, a single value per factorized layer is needed
        by default True

    Returns
    -------
    np.ndarray
        if one_per_order, of shape layers_shape
        otherwise, of shape (*layers_shape, order)
    """
    if one_per_order:
        target_shape = layers_shape + (order,)
    else:
        target_shape = layers_shape

    if isinstance(value, np.ndarray):
        assert value.shape == target_shape
        return value

    if isinstance(value, int):
        array = np.ones(target_shape, dtype=np.int32) * value
    else:
        assert len(value) == order
        array = np.ones(target_shape, dtype=np.int32)
        array[..., :] = value
    return array


def kernel_shape_to_factorization_shape(factorization, kernel_shape):
    """Returns the shape of the factorized weights to create depending on the factorization"""
    # For the TT case, the decomposition has a different shape than the kernel.
    if factorization.lower() == "tt":
        kernel_shape = list(kernel_shape)
        out_channel = kernel_shape.pop(0)
        kernel_shape.append(out_channel)
        return tuple(kernel_shape)

    # Other decompositions require no modification
    return kernel_shape


def factorization_shape_to_kernel_shape(factorization, factorization_shape):
    """Returns a convolutional kernel shape rom a factorized tensor shape"""
    if factorization.lower() == "tt":
        kernel_shape = list(factorization_shape)
        out_channel = kernel_shape.pop(-1)
        kernel_shape = [out_channel] + kernel_shape
        return tuple(kernel_shape)
    return factorization_shape


def kernel_to_tensor(factorization, kernel):
    """Returns a convolutional kernel ready to be factorized"""
    if factorization.lower() == "tt":
        kernel = tl.moveaxis(kernel, 0, -1)
    return kernel


def tensor_to_kernel(factorization, tensor):
    """Returns a kernel from a tensor factorization"""
    if factorization.lower() == "tt":
        tensor = tl.moveaxis(tensor, -1, 0)
    return tensor


class FactorizedConv(nn.Layer):
    """Create a factorized convolution of arbitrary order"""

    _version = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        order=None,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        has_bias=False,
        n_layers=1,
        factorization="cp",
        rank="same",
        implementation="factorized",
        fixed_rank_modes=None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # Check that order and kernel size are well defined and match
        if isinstance(kernel_size, int):
            if order is None:
                raise ValueError(
                    "If int given for kernel_size, order (dimension of the convolution) should also be provided."
                )
            if not isinstance(order, int) or order <= 0:
                raise ValueError(
                    f"order should be the (positive integer) order of the convolution"
                    f"but got order={order} of type {type(order)}."
                )
            else:
                kernel_size = (kernel_size,) * order
        else:
            kernel_size = tuple(kernel_size)
            order = len(kernel_size)

        self.order = order
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.implementation = implementation
        self.input_rank = rank
        self.n_layers = n_layers
        self.factorization = factorization

        # Shape to insert if multiple layers are parametrized
        if isinstance(n_layers, int):
            if n_layers == 1:
                layers_shape = ()
            else:
                layers_shape = (n_layers,)
        else:
            layers_shape = n_layers
        self.layers_shape = layers_shape

        # tensor of values for each parametrized conv
        self.padding = _ensure_array(layers_shape, order, padding)
        self.stride = _ensure_array(layers_shape, order, stride)
        self.dilation = _ensure_array(layers_shape, order, dilation)
        self.has_bias = _ensure_array(
            layers_shape, order, has_bias, one_per_order=False
        )

        if bias:
            self.bias = paddle.base.framework.EagerParamBase.from_tensor(
                paddle.empty(layers_shape, out_channels, device=device, dtype=dtype)
            )
        else:
            self.add_parameter("bias", None)

        if isinstance(factorization, FactorizedTensor):
            self.weight = factorization.to(device).to(dtype)
            kernel_shape = factorization_shape_to_kernel_shape(
                factorization._name, factorization.shape
            )
        else:
            kernel_shape = (out_channels, in_channels) + kernel_size
            # Some factorizations require permuting the dimensions, handled by kernel_shape_to_factorization_shape
            kernel_shape = kernel_shape_to_factorization_shape(
                factorization, kernel_shape
            )
            # In case we are parametrizing multiple layers
            factorization_shape = layers_shape + kernel_shape

            # For Tucker decomposition, we may want to not decomposed spatial dimensions
            if fixed_rank_modes is not None:
                if factorization.lower() != "tucker":
                    warnings.warn(
                        f"Got fixed_rank_modes={fixed_rank_modes} which is only used for factorization=tucker but got factorization={factorization}."
                    )
                elif fixed_rank_modes == "spatial":
                    fixed_rank_modes = list(
                        range(2 + len(layers_shape), 2 + len(layers_shape) + order)
                    )

            self.weight = FactorizedTensor.new(
                factorization_shape,
                rank=rank,
                factorization=factorization,
                fixed_rank_modes=fixed_rank_modes,
                device=device,
                dtype=dtype,
            )

        self.rank = self.weight.rank
        self.shape = self.weight.shape
        self.kernel_shape = kernel_shape
        # We pre-select the forward function to not waste time doing the check at each forward pass
        self.forward_fun = _get_factorized_conv(self.weight, self.implementation)

    def forward(self, x, indices=0):
        # Single layer parametrized
        if self.n_layers == 1:
            if indices == 0:
                return self.forward_fun(
                    x,
                    self.weight(),
                    bias=self.bias,
                    stride=self.stride.tolist(),
                    padding=self.padding.tolist(),
                    dilation=self.dilation.tolist(),
                )
            else:
                raise ValueError(
                    f"Only one convolution was parametrized (n_layers=1) but tried to access {indices}."
                )

        # Multiple layers parameterized
        if isinstance(self.n_layers, int):
            if not isinstance(indices, int):
                raise ValueError(
                    f"Expected indices to be in int but got indices={indices}"
                    f", but this conv was created with n_layers={self.n_layers}."
                )
        elif len(indices) != len(self.n_layers):
            raise ValueError(
                f"Got indices={indices}, but this conv was created with n_layers={self.n_layers}."
            )

        bias = self.bias[indices] if self.has_bias[indices] else None
        return self.forward_fun(
            x,
            self.weight(indices),
            bias=bias,
            stride=self.stride[indices].tolist(),
            padding=self.padding[indices].tolist(),
            dilation=self.dilation[indices].tolist(),
        )

    def reset_parameters(self, std=0.02):
        if self.bias is not None:
            self.bias.data.zero_()
        self.weight = self.weight.normal_(0, std)

    def set(self, indices, stride=1, padding=0, dilation=1, bias=None):
        """Sets the parameters of the conv self[indices]"""
        self.padding[indices] = _ensure_list(self.order, padding)
        self.stride[indices] = _ensure_list(self.order, stride)
        self.dilation[indices] = _ensure_list(self.order, dilation)
        if bias is not None:
            self.bias.data[indices] = bias.data
            self.has_bias[indices] = True

    def get_conv(self, indices):
        """Returns a sub-convolutional layer from the joint parametrize main-convolution

        The parametrization of sub-convolutional layers is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError(
                "A single convolution is parametrized, directly use the main class."
            )

        # if self.has_bias[indices]:
        #     bias = self.bias
        # else:
        #     bias = None

        return SubFactorizedConv(self, indices)
        # return SubFactorizedConv(self, indices, self.weight, bias)

    def __getitem__(self, indices):
        return self.get_conv(indices)

    @classmethod
    def from_factorization(
        cls,
        factorization,
        implementation="factorized",
        stride=1,
        padding=0,
        dilation=1,
        bias=None,
        n_layers=1,
    ):
        kernel_shape = factorization_shape_to_kernel_shape(
            factorization._name, factorization.shape
        )

        if n_layers == 1:
            out_channels, in_channels, *kernel_size = kernel_shape
        elif isinstance(n_layers, int):
            layer_size, out_channels, in_channels, *kernel_size = kernel_shape
            assert layer_size == n_layers
        else:
            layer_size = kernel_shape[: len(n_layers)]
            out_channels, in_channels, *kernel_size = kernel_shape[len(n_layers) :]

        order = len(kernel_size)

        instance = cls(
            in_channels,
            out_channels,
            kernel_size,
            order=order,
            implementation=implementation,
            padding=padding,
            stride=stride,
            bias=(bias is not None),
            n_layers=n_layers,
            dilation=dilation,
            factorization=factorization,
            rank=factorization.rank,
        )

        instance.weight = factorization

        if bias is not None:
            instance.bias.data = bias

        return instance

    @classmethod
    def from_conv(
        cls,
        conv_layer,
        rank="same",
        implementation="reconstructed",
        factorization="CP",
        decompose_weights=True,
        decomposition_kwargs=dict(),
        fixed_rank_modes=None,
        **kwargs,
    ):
        """Create a Factorized convolution from a regular convolutional layer

        Parameters
        ----------
        conv_layer : torch.nn.ConvND
        rank : rank of the decomposition, default is 'same'
        implementation : str, default is 'reconstructed'
        decomposed_weights : bool, default is True
            if True, the convolutional kernel is decomposed to initialize the factorized convolution
            otherwise, the factorized convolution's parameters are initialized randomly
        decomposition_kwargs : dict
            parameters passed directly on to the decompoosition function if `decomposed_weights` is True

        Returns
        -------
        New instance of the factorized convolution with equivalent weightss

        Todo
        ----
        Check that the decomposition of the given convolution and cls is the same.
        """
        padding = conv_layer.padding
        out_channels, in_channels, *kernel_size = conv_layer.weight.shape
        stride = conv_layer.stride[0]
        bias = conv_layer.bias is not None
        dilation = conv_layer.dilation

        instance = cls(
            in_channels,
            out_channels,
            kernel_size,
            factorization=factorization,
            implementation=implementation,
            rank=rank,
            dilation=dilation,
            padding=padding,
            stride=stride,
            bias=bias,
            fixed_rank_modes=fixed_rank_modes,
            **kwargs,
        )

        if decompose_weights:
            if conv_layer.bias is not None:
                instance.bias.data = conv_layer.bias.data

            with paddle.no_grad():
                kernel_tensor = kernel_to_tensor(factorization, conv_layer.weight.data)
                instance.weight.init_from_tensor(kernel_tensor, **decomposition_kwargs)
        else:
            instance.reset_parameters()

        return instance

    @classmethod
    def from_conv_list(
        cls,
        conv_list,
        rank="same",
        implementation="reconstructed",
        factorization="cp",
        decompose_weights=True,
        decomposition_kwargs=dict(),
        **kwargs,
    ):
        conv_layer = conv_list[0]
        padding = conv_layer.padding
        out_channels, in_channels, *kernel_size = conv_layer.weight.shape
        stride = conv_layer.stride[0]
        bias = True
        dilation = conv_layer.dilation

        instance = cls(
            in_channels,
            out_channels,
            kernel_size,
            implementation=implementation,
            rank=rank,
            factorization=factorization,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
            n_layers=len(conv_list),
            fixed_rank_modes=None,
            **kwargs,
        )

        if decompose_weights:
            with paddle.no_grad():
                weight_tensor = paddle.stack(
                    [
                        kernel_to_tensor(factorization, layer.weight.data)
                        for layer in conv_list
                    ]
                )
                instance.weight.init_from_tensor(weight_tensor, **decomposition_kwargs)
        else:
            instance.reset_parameters()

        for i, layer in enumerate(conv_list):
            instance.set(
                i,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                bias=layer.bias,
            )
            # instance.padding[i] = _ensure_list(instance.order, layer.padding)
            # instance.stride[i] = _ensure_list(instance.order, layer.stride)
            # instance.dilation[i] = _ensure_list(instance.order, layer.dilation)

        return instance

    def transduct(
        self,
        kernel_size,
        mode=0,
        padding=0,
        stride=1,
        dilation=1,
        fine_tune_transduction_only=True,
    ):
        """Transduction of the factorized convolution to add a new dimension

        Parameters
        ----------
        kernel_size : int
            size of the additional dimension
        mode : where to insert the new dimension, after the channels, default is 0
            by default, insert the new dimensions before the existing ones
            (e.g. add time before height and width)
        padding : int, default is 0
        stride : int: default is 1

        Returns
        -------
        self
        """
        if fine_tune_transduction_only:
            for param in self.parameters():
                param.requires_grad = False

        mode += len(self.layers_shape)
        self.order += 1
        padding = np.ones(self.layers_shape + (1,), dtype=int) * padding
        stride = np.ones(self.layers_shape + (1,), dtype=int) * stride
        dilation = np.ones(self.layers_shape + (1,), dtype=int) * dilation

        self.padding = np.concatenate(
            [self.padding[..., :mode], padding, self.padding[..., mode:]],
            len(self.layers_shape),
        )
        self.stride = np.concatenate(
            [self.stride[..., :mode], stride, self.stride[..., mode:]],
            len(self.layers_shape),
        )
        self.dilation = np.concatenate(
            [self.dilation[..., :mode], dilation, self.dilation[..., mode:]],
            len(self.layers_shape),
        )

        self.kernel_size = (
            self.kernel_size[:mode] + (kernel_size,) + self.kernel_size[mode:]
        )
        self.kernel_shape = (
            self.kernel_shape[: mode + 2]
            + (kernel_size,)
            + self.kernel_shape[mode + 2 :]
        )

        # Just to the frame-wise conv if adding time
        if isinstance(self.weight, CPTensor):
            new_factor = paddle.zeros(kernel_size, self.weight.rank)
            new_factor[kernel_size // 2, :] = 1
            transduction_mode = mode + 2
        elif isinstance(self.weight, TTTensor):
            new_factor = None
            transduction_mode = mode + 1
        else:
            transduction_mode = mode + 2
            new_factor = None

        self.weight = self.weight.transduct(kernel_size, transduction_mode, new_factor)

        return self

    def extra_repr(self):
        s = (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}"
            f", rank={self.rank}, order={self.order}"
        )
        if self.n_layers == 1:
            s += ", "
            if self.stride.tolist() != [1] * self.order:
                s += f"stride={self.stride.tolist()}, "
            if self.padding.tolist() != [0] * self.order:
                s += f"padding={self.padding.tolist()}, "
            if self.dilation.tolist() != [1] * self.order:
                s += f"dilation={self.dilation.tolist()}, "
            if self.bias is None:
                s += "bias=False"
            return s

        for idx in np.ndindex(self.n_layers):
            s += f"\n * Conv{idx}: "
            if self.stride[idx].tolist() != [1] * self.order:
                s += f"stride={self.stride[idx].tolist()}, "
            if self.padding[idx].tolist() != [0] * self.order:
                s += f"padding={self.padding[idx].tolist()}, "
            if self.dilation[idx].tolist() != [1] * self.order:
                s += f"dilation={self.dilation[idx].tolist()}, "
            if self.bias is None:
                s += "bias=False"
        return s


class SubFactorizedConv(nn.Layer):
    """Class representing one of the convolutions from the mother joint factorized convolution

    Parameters
    ----------

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules, they all point to the same data,
    which is shared.
    """

    def __init__(self, main_conv, indices):
        super().__init__()
        self.main_conv = main_conv
        self.indices = indices

    def forward(self, x):
        return self.main_conv.forward(x, self.indices)

    def __repr__(self):
        msg = f"SubConv {self.indices} from main factorized layer."
        msg += f"\n        {self.__class__.__name__}("
        msg += f"in_channels={self.main_conv.in_channels}, out_channels={self.main_conv.out_channels}"
        if self.main_conv.stride[self.indices].tolist() != [1] * self.main_conv.order:
            msg += f", stride={self.main_conv.stride[self.indices].tolist()}"
        if self.main_conv.padding[self.indices].tolist() != [0] * self.main_conv.order:
            msg += f", padding={self.main_conv.padding[self.indices].tolist()}"
        if self.main_conv.dilation[self.indices].tolist() != [1] * self.main_conv.order:
            msg += f", dilation={self.main_conv.dilation[self.indices].tolist()}"
        if self.main_conv.has_bias[self.indices]:
            msg += ", bias=False"
        msg += ")"
        return msg
