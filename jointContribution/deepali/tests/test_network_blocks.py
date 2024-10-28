r"""Test basic network blocks."""
import paddle
import pytest
from deepali.networks.blocks import DenseBlock
from deepali.networks.blocks import ResidualUnit
from deepali.networks.blocks import SkipConnection
from deepali.networks.layers import ConvLayer
from deepali.networks.layers import JoinLayer
from deepali.networks.layers import LambdaLayer


def test_dense_block() -> None:
    r"""Test block with dense skip connections."""

    x = paddle.to_tensor(data=[[1.0, -2.0, 3.0, 4.0, -5.0]])

    scale = LambdaLayer(lambda a: 2 * a)
    square = LambdaLayer(lambda a: a.square())
    clamp = LambdaLayer(lambda a: a.clip(min=0))

    block = DenseBlock(scale, square, clamp)
    assert isinstance(block.layers, paddle.nn.LayerDict)
    assert "0" in block.layers
    assert "1" in block.layers
    assert "2" in block.layers
    assert block.layers["0"] is scale
    assert block.layers["1"] is square
    assert block.layers["2"] is clamp

    block = DenseBlock({"scale": scale, "square": square, "clamp": clamp}, join="concat")
    assert isinstance(block.layers, paddle.nn.LayerDict)
    assert "scale" in block.layers
    assert "square" in block.layers
    assert "clamp" in block.layers
    assert block.layers["scale"] is scale
    assert block.layers["square"] is square
    assert block.layers["clamp"] is clamp

    y = block(x)
    assert isinstance(y, paddle.Tensor)

    a = scale(x)
    b = square(paddle.concat(x=[x, a], axis=1))
    c = clamp(paddle.concat(x=[x, a, b], axis=1))
    assert isinstance(c, paddle.Tensor)
    assert tuple(y.shape) == tuple(c.shape)
    assert y.allclose(y=c).item()

    block = DenseBlock({"scale": scale, "square": square, "clamp": clamp}, join="add")

    y = block(x)
    assert isinstance(y, paddle.Tensor)
    assert tuple(y.shape) == tuple(x.shape)

    a = scale(x)
    b = square(x + a)
    c = clamp(x + a + b)
    assert isinstance(c, paddle.Tensor)
    assert tuple(y.shape) == tuple(c.shape)
    assert y.allclose(y=c).item()


def test_residual_unit() -> None:
    r"""Test convoluational residual block."""

    with pytest.raises(TypeError):
        ResidualUnit(2, 1, 1, num_layers=1.0)
    with pytest.raises(ValueError):
        ResidualUnit(2, 1, 1, num_layers=0)
    with pytest.raises(ValueError):
        ResidualUnit(2, in_channels=64, out_channels=64, num_channels=32, num_layers=2)

    x = paddle.to_tensor(data=[[[1.0, -2.0, 3.0, 4.0, -5.0]]])

    block = ResidualUnit(spatial_dims=1, in_channels=1, order="nac")
    assert isinstance(block, paddle.nn.Layer)
    assert isinstance(block, SkipConnection)
    assert isinstance(block.func, paddle.nn.Sequential)
    assert len(block.func) == 2
    assert isinstance(block.func[0], ConvLayer)
    assert isinstance(block.func[1], ConvLayer)
    assert isinstance(block.skip, paddle.nn.Identity)

    y = block.join([x, x])
    assert isinstance(y, paddle.Tensor)
    assert y.equal(y=x + x).astype("bool").all()

    block = ResidualUnit(spatial_dims=1, in_channels=1, out_channels=2)
    assert block.spatial_dims == 1
    assert block.in_channels == 1
    assert block.out_channels == 2
    assert isinstance(block.skip, paddle.nn.Conv1D)
    assert block.skip._out_channels == 2

    block = ResidualUnit(
        spatial_dims=1,
        in_channels=1,
        out_channels=2,
        acti="relu",
        norm="group",
        num_layers=1,
        order="nac",
    )
    assert len(block.func) == 1
    assert isinstance(block.func[0].conv, paddle.nn.Conv1D)
    assert isinstance(block.func[0].norm, paddle.nn.GroupNorm)
    assert isinstance(block.func[0].acti, paddle.nn.ReLU)
    assert block.func[0].order == "NAC"
    assert not isinstance(block.join, paddle.nn.Sequential)

    block = ResidualUnit(
        spatial_dims=1,
        in_channels=1,
        out_channels=2,
        acti="relu",
        norm="group",
        num_layers=3,
        order="cna",
    )
    assert len(block.func) == 3
    assert isinstance(block.func[0].conv, paddle.nn.Conv1D)
    assert isinstance(block.func[0].norm, paddle.nn.GroupNorm)
    assert isinstance(block.func[0].acti, paddle.nn.ReLU)
    assert isinstance(block.func[1].conv, paddle.nn.Conv1D)
    assert isinstance(block.func[1].norm, paddle.nn.GroupNorm)
    assert isinstance(block.func[1].acti, paddle.nn.ReLU)
    assert isinstance(block.func[2].conv, paddle.nn.Conv1D)
    assert isinstance(block.func[2].norm, paddle.nn.GroupNorm)
    assert block.func[2].acti is None
    assert block.func[0].order == "CNA"
    assert isinstance(block.join, paddle.nn.Sequential)
    assert isinstance(block.join[0], JoinLayer)
    assert isinstance(block.join[1], paddle.nn.ReLU)

    y = block(x)
    assert isinstance(y, paddle.Tensor)
    assert tuple(y.shape) == (tuple(x.shape)[0], 2) + tuple(x.shape)[2:]


def test_skip_connection() -> None:
    r"""Test skip connection block."""

    x = paddle.to_tensor(data=[[1.0, 2.0, 3.0]])
    module = LambdaLayer(lambda a: a)

    # skip: identity, join: cat
    block = SkipConnection(module)
    assert isinstance(block, paddle.nn.Layer)
    assert isinstance(block.skip, paddle.nn.Identity)
    assert block.func is module

    y = block.join([x, x])
    assert isinstance(y, paddle.Tensor)
    assert tuple(y.shape) == (tuple(x.shape)[0], 2 * tuple(x.shape)[1])

    y = block(x)
    assert isinstance(y, paddle.Tensor)
    assert tuple(y.shape) == (tuple(x.shape)[0], 2 * tuple(x.shape)[1])
    assert y.equal(y=paddle.concat(x=[x, x], axis=1)).astype("bool").all()

    # skip: identity, join: add
    block = SkipConnection(module, name="residual", join="add")
    assert isinstance(block, paddle.nn.Layer)
    assert isinstance(block.skip, paddle.nn.Identity)
    assert block.func is module

    y = block(x)
    assert isinstance(y, paddle.Tensor)
    assert tuple(y.shape) == tuple(x.shape)
    assert y.equal(y=x + x).astype("bool").all()

    # skip: identity, join: mul
    block = SkipConnection(module, skip="identity", join="mul")
    assert isinstance(block, paddle.nn.Layer)
    assert isinstance(block.skip, paddle.nn.Identity)
    assert block.func is module

    y = block(x)
    assert isinstance(y, paddle.Tensor)
    assert tuple(y.shape) == tuple(x.shape)
    assert y.equal(y=x * x).astype("bool").all()

    # skip: custom, join: cat
    skip = LambdaLayer(lambda b: 2 * b)
    block = SkipConnection(module, skip=skip, join="cat")
    assert block.skip is skip
    assert block.func is module

    y = block(x)
    assert isinstance(y, paddle.Tensor)
    assert tuple(y.shape) == (tuple(x.shape)[0], 2 * tuple(x.shape)[1])
    assert y.equal(y=paddle.concat(x=[x, 2 * x], axis=1)).astype("bool").all()
