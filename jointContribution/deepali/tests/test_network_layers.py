r"""Test basic network layers."""
import paddle
import pytest
from deepali.core import PaddingMode
from deepali.networks.layers import Activation
from deepali.networks.layers import Conv2d
from deepali.networks.layers import Conv3d
from deepali.networks.layers import ConvLayer
from deepali.networks.layers import JoinLayer
from deepali.networks.layers import LambdaLayer
from deepali.networks.layers import NormLayer
from deepali.networks.layers import activation
from deepali.networks.layers import conv_module
from deepali.networks.layers import convolution
from deepali.networks.layers import join_func
from deepali.networks.layers import norm_layer
from deepali.networks.layers import normalization


def test_activation() -> None:
    r"""Test construction of non-linear activations."""

    act = activation("relu")
    assert isinstance(act, paddle.nn.ReLU)
    act = activation("lrelu", 0.1)
    assert isinstance(act, paddle.nn.LeakyReLU)
    assert act._negative_slope == 0.1
    # assert act.inplace is False
    act = activation(("LeakyReLU", {"negative_slope": 0.05}))
    assert isinstance(act, paddle.nn.LeakyReLU)
    assert act._negative_slope == 0.05
    # assert act.inplace is True
    act = activation("softmax", dim=2)
    assert isinstance(act, paddle.nn.Softmax)
    assert act._axis == 2

    def act_func(x: paddle.Tensor) -> paddle.Tensor:
        return x.sigmoid()

    act = activation(act_func)
    assert isinstance(act, Activation)
    assert isinstance(act, paddle.nn.Layer)
    assert act.func is act_func

    x = paddle.to_tensor(data=2.0)
    y = act(x)
    assert isinstance(y, paddle.Tensor)
    assert y.allclose(y=act_func(x)).item()

    act = Activation("elu")
    assert isinstance(act.func, paddle.nn.ELU)


def test_convolution() -> None:
    r"""Test construction of simple convolutional layer."""

    conv = convolution(1, 3, 16, 1)
    assert isinstance(conv, paddle.nn.Conv1D)
    assert tuple(conv.weight.shape) == (16, 3, 1)
    assert isinstance(conv.bias, paddle.Tensor)

    conv = convolution(2, 3, 32, kernel_size=1)
    assert isinstance(conv, paddle.nn.Conv2D)
    assert isinstance(conv, Conv2d)
    assert tuple(conv.weight.shape) == (32, 3, 1, 1)
    assert isinstance(conv.bias, paddle.Tensor)

    conv = convolution(3, 3, 8, kernel_size=1)
    assert isinstance(conv, paddle.nn.Conv3D)
    assert isinstance(conv, Conv3d)
    assert tuple(conv.weight.shape) == (8, 3, 1, 1, 1)
    assert isinstance(conv.bias, paddle.Tensor)

    conv = convolution(3, 3, 8, 5, bias=False)
    assert isinstance(conv, paddle.nn.Conv3D)
    assert tuple(conv.weight.shape) == (8, 3, 5, 5, 5)
    assert conv.bias is None
    # assert conv.output_padding == (0, 0, 0)

    conv = convolution(3, 3, 8, kernel_size=(5, 3, 1), output_padding=0)
    assert isinstance(conv, paddle.nn.Conv3D)
    assert tuple(conv.weight.shape) == (8, 3, 5, 3, 1)
    assert isinstance(conv.bias, paddle.Tensor)
    # assert conv.output_padding == (0, 0, 0)

    conv = conv_module(3, 3, 16, kernel_size=3, stride=2, output_padding=1, transposed=True)
    assert isinstance(conv, paddle.nn.Conv3DTranspose)
    # assert conv.transposed is True
    assert tuple(conv._kernel_size) == (3, 3, 3)
    assert tuple(conv._stride) == (2, 2, 2)
    # assert conv.output_padding == (1, 1, 1)

    conv = conv_module(
        spatial_dims=3,
        in_channels=6,
        out_channels=16,
        kernel_size=3,
        stride=2,
        dilation=4,
        padding=1,
        padding_mode=PaddingMode.REFLECT,
        groups=2,
        init="xavier",
        bias="zeros",
    )
    assert isinstance(conv, Conv3d)
    assert conv.weight_init == "xavier"
    assert conv.bias_init == "zeros"
    assert conv._in_channels == 6
    assert conv._out_channels == 16
    assert conv._groups == 2
    assert conv._padding_mode == "reflect"
    assert tuple(conv._kernel_size) == (3, 3, 3)
    assert tuple(conv._stride) == (2, 2, 2)
    assert tuple(conv._dilation) == (4, 4, 4)
    assert conv._padding == 1
    assert conv.output_padding == 0
    # assert conv.transposed is False


def test_conv_layer() -> None:
    r"""Test convolutional layer with optional normalization and/or activation."""
    layer = ConvLayer(2, 1, 8, 3)
    assert isinstance(layer, paddle.nn.Layer)
    assert hasattr(layer, "acti")
    assert hasattr(layer, "norm")
    assert hasattr(layer, "conv")
    assert layer.acti is None
    assert layer.norm is None
    assert isinstance(layer.conv, paddle.nn.Conv2D)

    layer = ConvLayer(
        1, in_channels=1, out_channels=16, kernel_size=3, acti=("lrelu", {"negative_slope": 0.1})
    )
    assert isinstance(layer.acti, paddle.nn.LeakyReLU)
    assert layer.norm is None
    assert isinstance(layer.conv, paddle.nn.Conv1D)
    assert layer.order == "CNA"
    assert layer.acti._negative_slope == 0.1

    layer = ConvLayer(2, in_channels=1, out_channels=16, kernel_size=3, acti="relu")
    assert isinstance(layer.acti, paddle.nn.ReLU)
    assert layer.norm is None
    assert isinstance(layer.conv, paddle.nn.Conv2D)
    assert layer.order == "CNA"

    layer = ConvLayer(3, in_channels=1, out_channels=16, kernel_size=3, acti="prelu", norm="batch")
    assert isinstance(layer.acti, paddle.nn.PReLU)
    assert isinstance(layer.norm, paddle.nn.BatchNorm3D)
    assert isinstance(layer.conv, paddle.nn.Conv3D)
    assert layer.order == "CNA"

    layer = ConvLayer(
        3, in_channels=1, out_channels=16, kernel_size=3, acti="relu", norm="instance", order="nac"
    )
    assert isinstance(layer.acti, paddle.nn.ReLU)
    assert isinstance(layer.norm, paddle.nn.InstanceNorm3D)
    assert isinstance(layer.conv, paddle.nn.Conv3D)
    assert layer.order == "NAC"

    x = paddle.randn(shape=(2, 1, 7, 9, 11))
    y = layer(x)
    z = layer.conv(layer.acti(layer.norm(x)))
    assert isinstance(y, paddle.Tensor)
    assert isinstance(z, paddle.Tensor)
    assert y.allclose(y=z).item()

    layer = ConvLayer(
        3, in_channels=1, out_channels=16, kernel_size=3, acti="relu", norm="instance", order="CNA"
    )
    assert isinstance(layer.acti, paddle.nn.ReLU)
    assert isinstance(layer.norm, paddle.nn.InstanceNorm3D)
    assert isinstance(layer.conv, paddle.nn.Conv3D)
    assert layer.order == "CNA"

    y = layer(x)
    assert isinstance(y, paddle.Tensor)
    assert not y.allclose(y=z).item()
    z = layer.acti(layer.norm(layer.conv(x)))
    assert isinstance(z, paddle.Tensor)
    assert y.allclose(y=z).item()


def test_join_layer() -> None:
    r"""Test layer which joins features of one or more input tensors."""

    with pytest.raises(ValueError):
        join_func("foo")
    with pytest.raises(ValueError):
        JoinLayer("bar")

    x = paddle.to_tensor(data=[[1.0, 1.5, 2.0]])
    y = paddle.to_tensor(data=[[0.4, 2.5, -0.1]])

    func = join_func("add")
    z = func([x, y])
    assert isinstance(z, paddle.Tensor)
    assert z.allclose(y=x + y).item()

    func = join_func("mul")
    z = func([x, y])
    assert isinstance(z, paddle.Tensor)
    assert z.allclose(y=x * y).item()

    func = join_func("cat")
    z = func([x, y])
    assert isinstance(z, paddle.Tensor)
    assert z.allclose(y=paddle.concat(x=[x, y], axis=1)).item()

    func = join_func("concat")
    z = func([x, y])
    assert isinstance(z, paddle.Tensor)
    assert z.allclose(y=paddle.concat(x=[x, y], axis=1)).item()

    join = JoinLayer("add", dim=0)
    z = join([x, y])
    assert isinstance(z, paddle.Tensor)
    assert z.allclose(y=x + y).item()

    join = JoinLayer("mul")
    z = join([x, y])
    assert isinstance(z, paddle.Tensor)
    assert z.allclose(y=x * y).item()

    join = JoinLayer("cat", dim=0)
    z = join([x, y])
    assert isinstance(z, paddle.Tensor)
    assert z.allclose(y=paddle.concat(x=[x, y], axis=0)).item()

    join = JoinLayer("concat", dim=1)
    z = join([x, y])
    assert isinstance(z, paddle.Tensor)
    assert z.allclose(y=paddle.concat(x=[x, y], axis=1)).item()


def test_lambda_layer() -> None:
    def square_func(x: paddle.Tensor) -> paddle.Tensor:
        return x.square()

    square_layer = LambdaLayer(square_func)
    assert isinstance(square_layer, paddle.nn.Layer)
    assert square_layer.func is square_func

    x = paddle.to_tensor(data=[[2.0, 0.5]])
    y = square_layer(x)
    assert isinstance(y, paddle.Tensor)
    assert y.allclose(y=paddle.to_tensor(data=[[4.0, 0.25]])).item()


def test_norm_layer() -> None:
    r"""Test construction of normalization layers."""

    # Batch normalization
    with pytest.raises(ValueError):
        norm_layer("batch", spatial_dims=3)
    with pytest.raises(ValueError):
        normalization("batch", spatial_dims=1)
    with pytest.raises(ValueError):
        normalization("batch", num_features=32)
    norm = norm_layer("batch", spatial_dims=1, num_features=16)
    assert isinstance(norm, paddle.nn.BatchNorm1D)
    # assert norm.affine is True
    assert norm._num_features == 16
    assert tuple(norm.bias.shape) == (16,)
    norm = normalization("batch", spatial_dims=2, num_features=32)
    assert isinstance(norm, paddle.nn.BatchNorm2D)
    # assert norm.affine is True
    assert norm._num_features == 32
    assert tuple(norm.bias.shape) == (32,)
    norm = norm_layer("BatchNorm", spatial_dims=3, num_features=64)
    assert isinstance(norm, paddle.nn.BatchNorm3D)
    # assert norm.affine is True
    assert norm._num_features == 64
    assert tuple(norm.bias.shape) == (64,)

    # Group normalization
    with pytest.raises(ValueError):
        norm_layer("group", 3)
    with pytest.raises(ValueError):
        norm_layer("group", spatial_dims=1)
    norm = norm_layer("group", num_features=32)
    assert isinstance(norm, paddle.nn.GroupNorm)
    assert norm._num_groups == 1
    assert norm._num_channels == 32
    # assert norm.affine is True
    norm = norm_layer("GroupNorm", num_channels=16)
    assert isinstance(norm, paddle.nn.GroupNorm)
    assert norm._num_groups == 1
    assert norm._num_channels == 16
    # assert norm.affine is True
    norm = normalization("group", num_groups=8, num_channels=64, affine=False)
    assert isinstance(norm, paddle.nn.GroupNorm)
    assert norm._num_groups == 8
    assert norm._num_channels == 64
    # assert norm.affine is False

    # Layer normalization
    with pytest.raises(ValueError):
        norm_layer("layer")
    with pytest.raises(ValueError):
        norm_layer("layer", spatial_dims=1)
    norm = normalization("layer", num_features=32)
    assert isinstance(norm, paddle.nn.GroupNorm)
    assert norm._num_groups == 1
    assert norm._num_channels == 32
    # assert norm.affine is True
    norm = norm_layer("LayerNorm", spatial_dims=1, num_features=64, affine=False)
    assert isinstance(norm, paddle.nn.GroupNorm)
    assert norm._num_groups == 1
    assert norm._num_channels == 64
    # assert norm.affine is False

    # Instance normalization
    with pytest.raises(ValueError):
        norm_layer("instance", 2)
    with pytest.raises(ValueError):
        norm_layer("instance", spatial_dims=1)
    with pytest.raises(ValueError):
        normalization("instance", num_features=32)
    norm = norm_layer("instance", spatial_dims=1, num_features=32)
    assert isinstance(norm, paddle.nn.InstanceNorm1D)
    norm = norm_layer("instance", spatial_dims=2, num_features=32)
    assert isinstance(norm, paddle.nn.InstanceNorm2D)
    norm = normalization("instance", spatial_dims=3, num_features=32)
    assert isinstance(norm, paddle.nn.InstanceNorm3D)
    assert norm._num_features == 32
    # assert norm.affine is False
    norm = normalization("InstanceNorm", spatial_dims=3, num_features=32, affine=True)
    assert isinstance(norm, paddle.nn.InstanceNorm3D)
    assert norm._num_features == 32
    # assert norm.affine is True
    norm = NormLayer("instance", spatial_dims=3, num_features=64)
    assert isinstance(norm.func, paddle.nn.InstanceNorm3D)
    assert norm.func._num_features == 64
    # assert norm.func.affine is False

    # Custom normalization
    def norm_func(x: paddle.Tensor) -> paddle.Tensor:
        return x

    norm = normalization(norm_func)
    assert isinstance(norm, paddle.nn.Layer)
    assert isinstance(norm, NormLayer)
    assert norm.func is norm_func

    norm = NormLayer(norm_func)
    assert norm.func is norm_func
