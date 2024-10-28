import itertools

import pytest
from deepali.core.enum import FlowChannelIndex
from deepali.core.enum import FlowDerivativeKeys
from deepali.core.enum import SpatialDerivativeKeys
from deepali.core.enum import SpatialDim


def test_flow_derivative_keys_from_arg() -> None:
    for D, order in itertools.product([2, 3], [0, 1, 2]):
        keys = FlowDerivativeKeys.from_arg(spatial_dims=D, order=1)
        expected = FlowDerivativeKeys.all(spatial_dims=D, order=1)
        assert keys == expected

    keys = ["du/dx", "du/dxx", "dv/dy", "dv/dxy"]
    assert FlowDerivativeKeys.from_arg(spatial_dims=2, which=keys) == keys
    assert FlowDerivativeKeys.from_arg(spatial_dims=2, which=keys, order=0) == []
    assert FlowDerivativeKeys.from_arg(spatial_dims=2, which=keys, order=1) == ["du/dx", "dv/dy"]
    assert FlowDerivativeKeys.from_arg(spatial_dims=2, which=keys, order=2) == ["du/dxx", "dv/dxy"]

    assert FlowDerivativeKeys.from_arg(spatial_dims=2, which="x") == ["du/dx", "dv/dx"]
    assert FlowDerivativeKeys.from_arg(spatial_dims=3, which="x") == ["du/dx", "dv/dx", "dw/dx"]


def test_flow_derivative_keys_split() -> None:
    assert FlowDerivativeKeys.split("du/dx") == (FlowChannelIndex.U, "x")
    assert FlowDerivativeKeys.split("du/dxy") == (FlowChannelIndex.U, "xy")
    assert FlowDerivativeKeys.split("dv/dzy") == (FlowChannelIndex.V, "zy")
    assert FlowDerivativeKeys.split("dw/dxxyz") == (FlowChannelIndex.W, "xxyz")

    assert FlowDerivativeKeys.split(["du/dxy", "dv/dzy"]) == [
        (FlowChannelIndex.U, "xy"),
        (FlowChannelIndex.V, "zy"),
    ]


def test_flow_derivative_keys_symbol() -> None:
    assert FlowDerivativeKeys.symbol(0, 0) == "du/dx"
    assert FlowDerivativeKeys.symbol(0, 1) == "du/dy"
    assert FlowDerivativeKeys.symbol(0, 2) == "du/dz"
    assert FlowDerivativeKeys.symbol(0, 3) == "du/dt"

    assert FlowDerivativeKeys.symbol(1, 0) == "dv/dx"
    assert FlowDerivativeKeys.symbol(1, 1) == "dv/dy"
    assert FlowDerivativeKeys.symbol(1, 2) == "dv/dz"
    assert FlowDerivativeKeys.symbol(1, 3) == "dv/dt"

    assert FlowDerivativeKeys.symbol(2, 0) == "dw/dx"
    assert FlowDerivativeKeys.symbol(2, 1) == "dw/dy"
    assert FlowDerivativeKeys.symbol(2, 2) == "dw/dz"
    assert FlowDerivativeKeys.symbol(2, 3) == "dw/dt"

    assert FlowDerivativeKeys.symbol("u", 0, "y", SpatialDim.Z) == "du/dxyz"
    assert FlowDerivativeKeys.symbol("v", "xyz") == "dv/dxyz"

    with pytest.raises(TypeError):
        FlowDerivativeKeys.symbol(0, 2.3)


def test_flow_derivative_keys_unique() -> None:
    value = FlowDerivativeKeys.unique("du/dx")
    assert isinstance(value, str)
    assert value == "du/dx"

    value = FlowDerivativeKeys.unique(["du/dx"])
    assert isinstance(value, set)
    assert value == {"du/dx"}

    assert FlowDerivativeKeys.unique(["du/dxy", "du/dyx"]) == {"du/dxy"}
    assert FlowDerivativeKeys.unique(["du/dz", "du/dxy", "du/dyx"]) == {"du/dxy", "du/dz"}
    assert FlowDerivativeKeys.unique(["dv/dxy", "du/dxy", "du/dyx"]) == {"du/dxy", "dv/dxy"}


def test_flow_derivative_keys_all() -> None:
    for d, order in itertools.product([2, 3], [0, 1, 2]):
        channel_keys = ["u", "v", "w"][:d]
        spatial_keys = SpatialDerivativeKeys.all(spatial_dims=d, order=order)
        expected = [f"d{a}/d{b}" for a, b in itertools.product(channel_keys, spatial_keys)]
        assert FlowDerivativeKeys.all(spatial_dims=d, order=order) == expected


def test_flow_derivative_keys_gradient() -> None:
    assert FlowDerivativeKeys.gradient(spatial_dims=3, channel="v") == ["dv/dx", "dv/dy", "dv/dz"]
    assert FlowDerivativeKeys.gradient(spatial_dims=3, channel=[0, "v"]) == [
        # fmt: off
        "du/dx", "du/dy", "du/dz",
        "dv/dx", "dv/dy", "dv/dz",
        # fmt: on
    ]


def test_flow_derivative_keys_jacobian() -> None:
    for d in [2, 3]:
        expected = FlowDerivativeKeys.all(spatial_dims=d, order=1)
        assert FlowDerivativeKeys.jacobian(spatial_dims=d) == expected


def test_flow_derivative_keys_divergence() -> None:
    assert FlowDerivativeKeys.divergence(spatial_dims=2) == ["du/dx", "dv/dy"]
    assert FlowDerivativeKeys.divergence(spatial_dims=3) == ["du/dx", "dv/dy", "dw/dz"]


def test_flow_derivative_keys_curvature() -> None:
    assert FlowDerivativeKeys.curvature(spatial_dims=2) == [
        # fmt: off
        "du/dxx", "du/dyy",
        "dv/dxx", "dv/dyy",
        # fmt: on
    ]
    assert FlowDerivativeKeys.curvature(spatial_dims=3) == [
        # fmt: off
        "du/dxx", "du/dyy", "du/dzz",
        "dv/dxx", "dv/dyy", "dv/dzz",
        "dw/dxx", "dw/dyy", "dw/dzz",
        # fmt: on
    ]


def test_flow_derivative_keys_hessian() -> None:
    for d, c in itertools.product([2, 3], [None, "u", "v", ["u", "v"]]):
        expected = FlowDerivativeKeys.all(spatial_dims=d, channel=c, order=2)
        assert FlowDerivativeKeys.hessian(spatial_dims=d, channel=c) == expected


def test_flow_derivative_keys_unmixed() -> None:
    for d, order in itertools.product([2, 3], [0, 1, 2]):
        if order == 0:
            expected = []
        elif order == 1:
            expected = FlowDerivativeKeys.all(spatial_dims=d, order=1)
        elif order == 2:
            expected = FlowDerivativeKeys.curvature(spatial_dims=d)
        else:
            raise AssertionError(f"unexpected order: {order}")
        assert FlowDerivativeKeys.unmixed(spatial_dims=d, order=order) == expected
