from typing import Sequence

import deepali.core.bspline as B
import deepali.core.functional as U
import numpy as np
import paddle
import pytest
from deepali.core import Grid
from deepali.core.enum import FlowDerivativeKeys
from deepali.utils import paddle_aux  # noqa

PERIODIC_FLOW_X_SCALE = 2 * np.pi
PERIODIC_FLOW_U_SCALE = 0.1


def periodic_flow(p: paddle.Tensor) -> paddle.Tensor:
    q = p.mul(PERIODIC_FLOW_X_SCALE)
    start_0 = q.shape[1] + 0 if 0 < 0 else 0
    u = paddle.slice(q, [1], [start_0], [start_0 + 1]).sin().neg_()
    start_1 = q.shape[1] + 1 if 1 < 0 else 1
    v = paddle.slice(q, [1], [start_1], [start_1 + 1]).cos()
    f = paddle.concat(x=[u, v], axis=1)
    return f.multiply_(y=paddle.to_tensor(PERIODIC_FLOW_U_SCALE))


def periodic_flow_du_dx(p: paddle.Tensor) -> paddle.Tensor:
    start_2 = p.shape[1] + 0 if 0 < 0 else 0
    g = paddle.slice(p, [1], [start_2], [start_2 + 1]).mul(PERIODIC_FLOW_X_SCALE).cos()
    g = g.multiply_(y=paddle.to_tensor(-PERIODIC_FLOW_X_SCALE * PERIODIC_FLOW_U_SCALE))
    return g


def periodic_flow_du_dy(p: paddle.Tensor) -> paddle.Tensor:
    return paddle.zeros(shape=(tuple(p.shape)[0], 1) + tuple(p.shape)[2:], dtype=p.dtype)


def periodic_flow_du_dxx(p: paddle.Tensor) -> paddle.Tensor:
    start_3 = p.shape[1] + 0 if 0 < 0 else 0
    g = paddle.slice(p, [1], [start_3], [start_3 + 1]).mul(PERIODIC_FLOW_X_SCALE).sin()
    g = g.multiply_(
        y=paddle.to_tensor(PERIODIC_FLOW_X_SCALE * PERIODIC_FLOW_X_SCALE * PERIODIC_FLOW_U_SCALE)
    )
    return g


def periodic_flow_du_dyy(p: paddle.Tensor) -> paddle.Tensor:
    return paddle.zeros(shape=(tuple(p.shape)[0], 1) + tuple(p.shape)[2:], dtype=p.dtype)


def periodic_flow_dv_dx(p: paddle.Tensor) -> paddle.Tensor:
    return paddle.zeros(shape=(tuple(p.shape)[0], 1) + tuple(p.shape)[2:], dtype=p.dtype)


def periodic_flow_dv_dy(p: paddle.Tensor) -> paddle.Tensor:
    start_4 = p.shape[1] + 1 if 1 < 0 else 1
    g = paddle.slice(p, [1], [start_4], [start_4 + 1]).mul(PERIODIC_FLOW_X_SCALE).sin()
    g = g.multiply_(y=paddle.to_tensor(-PERIODIC_FLOW_X_SCALE * PERIODIC_FLOW_U_SCALE))
    return g


def periodic_flow_dv_dxx(p: paddle.Tensor) -> paddle.Tensor:
    return paddle.zeros(shape=(tuple(p.shape)[0], 1) + tuple(p.shape)[2:], dtype=p.dtype)


def periodic_flow_dv_dyy(p: paddle.Tensor) -> paddle.Tensor:
    start_5 = p.shape[1] + 1 if 1 < 0 else 1
    g = paddle.slice(p, [1], [start_5], [start_5 + 1]).mul(PERIODIC_FLOW_X_SCALE).cos()
    g = g.multiply_(
        y=paddle.to_tensor(-PERIODIC_FLOW_X_SCALE * PERIODIC_FLOW_X_SCALE * PERIODIC_FLOW_U_SCALE)
    )
    return g


def periodic_flow_deriv(p: paddle.Tensor, which: str) -> paddle.Tensor:
    deriv_fn = {
        "du/dx": periodic_flow_du_dx,
        "du/dy": periodic_flow_du_dy,
        "dv/dx": periodic_flow_dv_dx,
        "dv/dy": periodic_flow_dv_dy,
        "du/dxx": periodic_flow_du_dxx,
        "du/dyy": periodic_flow_du_dyy,
        "dv/dxx": periodic_flow_dv_dxx,
        "dv/dyy": periodic_flow_dv_dyy,
    }
    return deriv_fn[which](p)


def periodic_flow_divergence(p: paddle.Tensor) -> paddle.Tensor:
    du_dx = periodic_flow_du_dx(p)
    dv_dy = periodic_flow_dv_dy(p)
    return du_dx.add(dv_dy)


def random_svf(size: Sequence[int], stride: int = 1, generator=None) -> paddle.Tensor:
    cp_grid_size = B.cubic_bspline_control_point_grid_size(size, stride=stride)
    data = paddle.randn(shape=(1, 3) + cp_grid_size)
    data = U.fill_border(data, margin=3, value=0, inplace=True)
    return B.evaluate_cubic_bspline(data, size=size, stride=stride)


def difference(a: paddle.Tensor, b: paddle.Tensor, margin: int = 0) -> paddle.Tensor:
    assert tuple(a.shape) == tuple(b.shape)
    i = [
        (slice(0, n, 1) if dim < 2 else slice(margin, n - margin, 1))
        for dim, n in enumerate(tuple(a.shape))
    ]
    # paddle need to set the corresponding elements.
    shape = a.shape
    if len(shape) == 4:
        n1, n2, n3, n4 = shape
        if b.shape == shape:
            return a[slice(0, n1, 1), slice(0, n2, 1), slice(0, n3, 1), slice(0, n4, 1)].sub(
                b[slice(0, n1, 1), slice(0, n2, 1), slice(0, n3, 1), slice(0, n4, 1)]
            )
    if len(shape) == 5:
        n1, n2, n3, n4, n5 = shape
        if b.shape == shape:
            return a[
                slice(0, n1, 1), slice(0, n2, 1), slice(0, n3, 1), slice(0, n4, 1), slice(0, n5, 1)
            ].sub(
                b[
                    slice(0, n1, 1),
                    slice(0, n2, 1),
                    slice(0, n3, 1),
                    slice(0, n4, 1),
                    slice(0, n5, 1),
                ]
            )
    raise AssertionError(f"Unsupported shape {a.shape} slice {i}")
    # return a[i].sub(b[i])


def test_flow_curl() -> None:
    # 2-dimensional vector field
    p = U.move_dim(Grid(size=(32, 24)).coords().unsqueeze_(0), -1, 1)
    start_6 = p.shape[1] + 0 if 0 < 0 else 0
    x = paddle.slice(p, [1], [start_6], [start_6 + 1])
    start_7 = p.shape[1] + 1 if 1 < 0 else 1
    y = paddle.slice(p, [1], [start_7], [start_7 + 1])
    flow = paddle.concat(x=[y.neg(), x], axis=1)
    curl = U.curl(flow)
    assert isinstance(curl, paddle.Tensor)
    assert tuple(curl.shape) == (tuple(flow.shape)[0], 1) + tuple(flow.shape)[2:]
    assert curl.dtype == flow.dtype
    assert paddle_aux.is_eq_place(curl.place, flow.place)
    assert curl.sub(2).abs().less_than(y=paddle.to_tensor(1e-06)).astype("bool").all()

    # 3-dimensional vector field
    p = U.move_dim(Grid(size=(64, 32, 16)).coords().unsqueeze_(0), -1, 1)
    start_8 = p.shape[1] + 0 if 0 < 0 else 0
    x = paddle.slice(p, [1], [start_8], [start_8 + 1])
    start_9 = p.shape[1] + 1 if 1 < 0 else 1
    y = paddle.slice(p, [1], [start_9], [start_9 + 1])
    start_10 = p.shape[1] + 2 if 2 < 0 else 2
    z = paddle.slice(p, [1], [start_10], [start_10 + 1])

    flow = paddle.concat(x=[x.mul(z), y.mul(z), x.mul(y)], axis=1)

    curl = U.curl(flow)
    assert isinstance(curl, paddle.Tensor)
    assert tuple(curl.shape) == tuple(flow.shape)
    assert curl.dtype == flow.dtype
    assert paddle_aux.is_eq_place(curl.place, flow.place)

    expected = x.sub(y)
    expected = paddle.concat(x=[expected, expected, paddle.zeros_like(x=expected)], axis=1)
    error = difference(curl, expected, margin=1)
    assert error.abs().max().less_than(y=paddle.to_tensor(1e-05))

    div = U.divergence(curl)
    assert div.abs().max().less_than(y=paddle.to_tensor(1e-05))


def test_flow_derivatives() -> None:
    # 2D vector field defined by periodic functions
    p = U.move_dim(Grid(size=(120, 100)).coords().unsqueeze_(0), -1, 1)

    flow = periodic_flow(p)

    which = FlowDerivativeKeys.all(spatial_dims=2, order=1)
    which.append("du/dxx")
    which.append("dv/dyy")

    deriv = U.flow_derivatives(flow, which=which)
    assert isinstance(deriv, dict)
    assert all(isinstance(k, str) for k in deriv.keys())
    assert all(isinstance(v, paddle.Tensor) for v in deriv.values())
    assert all(
        tuple(v.shape) == (tuple(flow.shape)[0], 1) + tuple(flow.shape)[2:] for v in deriv.values()
    )

    for key, value in deriv.items():
        expected = periodic_flow_deriv(p, key)
        order = FlowDerivativeKeys.order(key)
        dif = difference(value, expected, margin=order)
        tol = 0.003 * 10 ** (order - 1)
        assert dif.abs().max().less_than(y=paddle.to_tensor(tol)), f"flow derivative {key}"

    # 3D vector field
    p = U.move_dim(Grid(size=(64, 32, 16)).coords().unsqueeze_(0), -1, 1)
    start_11 = p.shape[1] + 0 if 0 < 0 else 0
    x = paddle.slice(p, [1], [start_11], [start_11 + 1])
    start_12 = p.shape[1] + 1 if 1 < 0 else 1
    y = paddle.slice(p, [1], [start_12], [start_12 + 1])
    start_13 = p.shape[1] + 2 if 2 < 0 else 2
    z = paddle.slice(p, [1], [start_13], [start_13 + 1])
    flow = paddle.concat(x=[x.mul(z), y.mul(z), x.mul(y)], axis=1)
    deriv = U.flow_derivatives(flow, order=1)
    assert difference(deriv["du/dx"], z).abs().max().less_than(y=paddle.to_tensor(1e-05))
    assert deriv["du/dy"].abs().max().less_than(y=paddle.to_tensor(1e-05))
    assert difference(deriv["du/dz"], x).abs().max().less_than(y=paddle.to_tensor(1e-05))
    assert deriv["dv/dx"].abs().max().less_than(y=paddle.to_tensor(1e-05))
    assert difference(deriv["dv/dy"], z).abs().max().less_than(y=paddle.to_tensor(1e-05))
    assert difference(deriv["dv/dz"], y).abs().max().less_than(y=paddle.to_tensor(1e-05))
    assert difference(deriv["dw/dx"], y).abs().max().less_than(y=paddle.to_tensor(1e-05))
    assert difference(deriv["dw/dy"], x).abs().max().less_than(y=paddle.to_tensor(1e-05))
    assert deriv["dw/dz"].abs().max().less_than(y=paddle.to_tensor(1e-05))
    deriv = U.flow_derivatives(flow, which=["du/dxz", "dv/dzy", "dw/dxy"])
    assert deriv["du/dxz"].sub(1).abs().max().less_than(y=paddle.to_tensor(0.0001))
    assert deriv["dv/dzy"].sub(1).abs().max().less_than(y=paddle.to_tensor(0.0001))
    assert deriv["dw/dxy"].sub(1).abs().max().less_than(y=paddle.to_tensor(0.0001))


def test_flow_divergence() -> None:
    grid = Grid(size=(16, 14))
    offset = U.translation([0.1, 0.2]).unsqueeze_(0)
    flow = U.affine_flow(offset, grid)
    div = U.divergence(flow)
    assert isinstance(div, paddle.Tensor)
    assert tuple(div.shape) == (tuple(flow.shape)[0], 1) + tuple(flow.shape)[2:]
    assert div.abs().max().less_than(y=paddle.to_tensor(1e-05))

    points = U.move_dim(Grid(size=(64, 64)).coords().unsqueeze_(0), -1, 1)
    flow = periodic_flow(points)
    div = U.divergence(flow)
    assert isinstance(div, paddle.Tensor)
    assert tuple(div.shape) == (tuple(flow.shape)[0], 1) + tuple(flow.shape)[2:]
    expected = periodic_flow_divergence(points)
    dif = difference(div, expected, margin=1)
    assert dif.abs().max().less_than(y=paddle.to_tensor(0.07))

    p = U.move_dim(Grid(size=(64, 32, 16)).coords().unsqueeze_(0), -1, 1)
    start_14 = p.shape[1] + 0 if 0 < 0 else 0
    x = paddle.slice(p, [1], [start_14], [start_14 + 1])
    start_15 = p.shape[1] + 1 if 1 < 0 else 1
    y = paddle.slice(p, [1], [start_15], [start_15 + 1])
    start_16 = p.shape[1] + 2 if 2 < 0 else 2
    z = paddle.slice(p, [1], [start_16], [start_16 + 1])
    flow = paddle.concat(x=[x.mul(z), y.mul(z), x.mul(y)], axis=1)
    div = U.divergence(flow)
    assert isinstance(div, paddle.Tensor)
    assert tuple(div.shape) == (tuple(flow.shape)[0], 1) + tuple(flow.shape)[2:]
    error = difference(div, z.mul(2))
    assert error.abs().max().less_than(y=paddle.to_tensor(1e-05))


def test_flow_divergence_free() -> None:
    data = paddle.randn(shape=(1, 1, 16, 24)).multiply_(y=paddle.to_tensor(0.01))
    flow = U.divergence_free_flow(data)
    assert tuple(flow.shape) == (tuple(data.shape)[0], 2) + tuple(data.shape)[2:]
    div = U.divergence(flow)
    assert div.abs().max().less_than(y=paddle.to_tensor(1e-05))
    data = paddle.randn(shape=(3, 2, 16, 24, 32)).multiply_(y=paddle.to_tensor(0.01))
    flow = U.divergence_free_flow(data, sigma=2.0)
    assert tuple(flow.shape) == (tuple(data.shape)[0], 3) + tuple(data.shape)[2:]
    div = U.divergence(flow)
    assert div[0, 0, 1:-1, 1:-1, 1:-1].abs().max().less_than(y=paddle.to_tensor(0.001))

    # coef = F.pad(data, (1, 2, 1, 2, 1, 2))
    # flow = U.divergence_free_flow(coef, mode="bspline", sigma=1.0)
    # assert flow.shape == (data.shape[0], 3) + data.shape[2:]
    # div = U.divergence(flow)
    # assert div[0, 0, 1:-1, 1:-1, 1:-1].abs().max().lt(1e-4)

    # flow = U.divergence_free_flow(data, mode="gaussian", sigma=0.7355)
    # assert flow.shape == (data.shape[0], 3) + data.shape[2:]
    # div = U.divergence(flow)
    # assert div[0, 0, 1:-1, 1:-1, 1:-1].abs().max().lt(1e-4)

    # constructing a divergence-free field using curl() seems to work best given
    # the higher magnitude and no need for Gaussian blurring of the random field
    # where each component is sampled i.i.d. from a normal distribution
    data = paddle.randn(shape=(5, 3, 16, 24, 32)).multiply_(y=paddle.to_tensor(0.2))
    flow = U.divergence_free_flow(data)
    assert tuple(flow.shape) == tuple(data.shape)
    div = U.divergence(flow)
    assert div.abs().max().less_than(y=paddle.to_tensor(0.0001))


def test_flow_jacobian() -> None:
    # 2D flow field
    p = Grid(size=(64, 32)).coords(channels_last=False).unsqueeze_(0)
    start_17 = p.shape[1] + 0 if 0 < 0 else 0
    x = paddle.slice(p, [1], [start_17], [start_17 + 1])
    start_18 = p.shape[1] + 1 if 1 < 0 else 1
    y = paddle.slice(p, [1], [start_18], [start_18 + 1])
    # interior = [slice(1, n - 1) for n in tuple(p.shape)[2:]]

    # u = [x^2, xy]
    flow = paddle.concat(x=[x.square(), x.mul(y)], axis=1)

    jac = paddle.zeros(shape=(tuple(p.shape)[0],) + tuple(p.shape)[2:] + (2, 2))
    jac[..., 0, 0] = x.squeeze(axis=1).mul(2)
    jac[..., 1, 0] = y.squeeze(axis=1)
    jac[..., 1, 1] = x.squeeze(axis=1)

    slice_tmp4 = [
        slice(None, None, None),
        slice(None, None, None),
        slice(1, 31, None),
        slice(1, 63, None),
    ]
    derivs = U.jacobian_dict(flow)
    for (i, j), deriv in derivs.items():
        atol = 1e-05
        error = difference(jac[..., i, j].unsqueeze(axis=1), deriv)
        if (i, j) == (0, 0):
            error = error[slice_tmp4[0], slice_tmp4[1], slice_tmp4[2], slice_tmp4[3]]
        if error.abs().max().greater_than(y=paddle.to_tensor(atol)):
            raise AssertionError(f"max absolute difference of jac[{i}, {j}] > {atol}")
    slice_tmp3 = [slice(None, None, None), slice(1, 31, None), slice(1, 63, None)]
    mat = U.jacobian_matrix(flow)
    assert paddle.allclose(
        x=mat[slice_tmp3[0], slice_tmp3[1], slice_tmp3[2]],
        y=jac[slice_tmp3[0], slice_tmp3[1], slice_tmp3[2]],
        atol=1e-05,
    ).item()

    jac[..., 0, 0] += 1
    jac[..., 1, 1] += 1

    mat = U.jacobian_matrix(flow, add_identity=True)
    assert paddle.allclose(
        x=mat[slice_tmp3[0], slice_tmp3[1], slice_tmp3[2]],
        y=jac[slice_tmp3[0], slice_tmp3[1], slice_tmp3[2]],
        atol=1e-05,
    ).item()

    slice_tmp4_0 = [slice(None, None, None), 0, slice(1, 31, None), slice(1, 63, None)]
    det = U.jacobian_det(flow)
    assert paddle.allclose(
        x=det[slice_tmp4_0[0], slice_tmp4_0[1], slice_tmp4_0[2], slice_tmp4_0[3]],
        y=paddle.linalg.det(jac[slice_tmp3[0], slice_tmp3[1], slice_tmp3[2]]),
        atol=1e-05,
    ).item()

    # 3D flow field
    p = Grid(size=(64, 32, 16)).coords(channels_last=False).unsqueeze_(0)
    start_19 = p.shape[1] + 0 if 0 < 0 else 0
    x = paddle.slice(p, [1], [start_19], [start_19 + 1])
    start_20 = p.shape[1] + 1 if 1 < 0 else 1
    y = paddle.slice(p, [1], [start_20], [start_20 + 1])
    start_21 = p.shape[1] + 2 if 2 < 0 else 2
    z = paddle.slice(p, [1], [start_21], [start_21 + 1])
    # interior = [slice(1, n - 1) for n in tuple(p.shape)[2:]]
    slice_tmp3 = [slice(1, 15, None), slice(1, 31, None), slice(1, 63, None)]
    slice_tmp4 = [
        slice(None, None, None),
        slice(1, 15, None),
        slice(1, 31, None),
        slice(1, 63, None),
    ]
    slice_tmp5_0 = [
        slice(None, None, None),
        0,
        slice(1, 15, None),
        slice(1, 31, None),
        slice(1, 63, None),
    ]
    slice_tmp5 = [
        slice(None, None, None),
        slice(None, None, None),
        slice(1, 15, None),
        slice(1, 31, None),
        slice(1, 63, None),
    ]

    # u = [z^2, 0, xy]
    flow = paddle.concat(x=[z.square(), paddle.zeros_like(x=y), x.mul(y)], axis=1)
    jac = paddle.zeros(shape=(tuple(p.shape)[0],) + tuple(p.shape)[2:] + (3, 3))
    jac[..., 0, 2] = z.squeeze(axis=1).mul(2)
    jac[..., 2, 0] = y.squeeze(axis=1)
    jac[..., 2, 1] = x.squeeze(axis=1)
    derivs = U.jacobian_dict(flow)
    for (i, j), deriv in derivs.items():
        atol = 1e-05
        error = difference(jac[..., i, j].unsqueeze(axis=1), deriv)
        if (i, j) == (0, 2):
            error = error[slice_tmp5[0], slice_tmp5[1], slice_tmp5[2], slice_tmp5[3], slice_tmp5[4]]
        if error.abs().max().greater_than(y=paddle.to_tensor(atol)):
            raise AssertionError(f"max absolute difference of jac[{i}, {j}] > {atol}")

    mat = U.jacobian_matrix(flow)
    assert paddle.allclose(
        x=mat[slice_tmp4[0], slice_tmp4[1], slice_tmp4[2], slice_tmp4[3]],
        y=jac[slice_tmp4[0], slice_tmp4[1], slice_tmp4[2], slice_tmp4[3]],
        atol=1e-05,
    ).item()

    jac[..., 0, 0] += 1
    jac[..., 1, 1] += 1
    jac[..., 2, 2] += 1

    mat = U.jacobian_matrix(flow, add_identity=True)
    assert paddle.allclose(
        x=mat[slice_tmp4[0], slice_tmp4[1], slice_tmp4[2], slice_tmp4[3]],
        y=jac[slice_tmp4[0], slice_tmp4[1], slice_tmp4[2], slice_tmp4[3]],
        atol=1e-05,
    ).item()

    det = U.jacobian_det(flow)
    assert paddle.allclose(
        x=det[slice_tmp5_0[0], slice_tmp5_0[1], slice_tmp5_0[2], slice_tmp5_0[3], slice_tmp5_0[4]],
        y=paddle.linalg.det(jac[slice_tmp4[0], slice_tmp4[1], slice_tmp4[2], slice_tmp4[3]]),
        atol=1e-05,
    ).item()

    # u = [0, x + y^3, yz]
    flow = paddle.concat(x=[paddle.zeros_like(x=x), x.add(y.pow(y=3)), y.mul(z)], axis=1)

    jac = paddle.zeros(shape=(tuple(p.shape)[0],) + tuple(p.shape)[2:] + (3, 3))
    jac[..., 1, 0] = 1
    jac[..., 1, 1] = y.squeeze(axis=1).square().mul(3)
    jac[..., 2, 1] = z.squeeze(axis=1)
    jac[..., 2, 2] = y.squeeze(axis=1)

    derivs = U.jacobian_dict(flow)
    for (i, j), deriv in derivs.items():
        atol = 1e-05
        error = difference(jac[..., i, j].unsqueeze(axis=1), deriv)
        if (i, j) == (1, 1):
            atol = 0.005
            error = error[slice_tmp5[0], slice_tmp5[1], slice_tmp5[2], slice_tmp5[3], slice_tmp5[4]]
        if error.abs().max().greater_than(y=paddle.to_tensor(atol)):
            raise AssertionError(f"max absolute difference of jac[{i}, {j}] > {atol}")

    mat = U.jacobian_matrix(flow)
    assert paddle.allclose(
        x=mat[slice_tmp4[0], slice_tmp4[1], slice_tmp4[2], slice_tmp4[3]],
        y=jac[slice_tmp4[0], slice_tmp4[1], slice_tmp4[2], slice_tmp4[3]],
        atol=0.005,
    ).item()

    jac[..., 0, 0] += 1
    jac[..., 1, 1] += 1
    jac[..., 2, 2] += 1

    mat = U.jacobian_matrix(flow, add_identity=True)
    assert paddle.allclose(
        x=mat[slice_tmp4[0], slice_tmp4[1], slice_tmp4[2], slice_tmp4[3]],
        y=jac[slice_tmp4[0], slice_tmp4[1], slice_tmp4[2], slice_tmp4[3]],
        atol=0.005,
    ).item()

    det = U.jacobian_det(flow)
    assert paddle.allclose(
        x=det[slice_tmp5_0[0], slice_tmp5_0[1], slice_tmp5_0[2], slice_tmp5_0[3], slice_tmp5_0[4]],
        y=paddle.linalg.det(jac[slice_tmp4[0], slice_tmp4[1], slice_tmp4[2], slice_tmp4[3]]),
        atol=0.01,
    ).item()


def test_flow_lie_bracket() -> None:
    p = U.move_dim(Grid(size=(64, 32, 16)).coords().unsqueeze_(0), -1, 1)
    start_22 = p.shape[1] + 0 if 0 < 0 else 0
    x = paddle.slice(p, [1], [start_22], [start_22 + 1])
    start_23 = p.shape[1] + 1 if 1 < 0 else 1
    y = paddle.slice(p, [1], [start_23], [start_23 + 1])
    start_24 = p.shape[1] + 2 if 2 < 0 else 2
    z = paddle.slice(p, [1], [start_24], [start_24 + 1])

    # u = [yz, xz, xy] and v = [x, y, z]
    u = paddle.concat(x=[y.mul(z), x.mul(z), x.mul(y)], axis=1)
    v = paddle.concat(x=[x, y, z], axis=1)
    w = u
    lb_uv = U.lie_bracket(u, v)
    assert paddle.allclose(x=U.lie_bracket(v, u), y=lb_uv.neg()).item()
    assert U.lie_bracket(u, u).abs().less_than(paddle.to_tensor(1e-06)).all()
    assert paddle.allclose(x=lb_uv, y=w, atol=1e-06).item()

    # u = [z^2, 0, xy] and v = [0, x + y^3, yz]
    u = paddle.concat(x=[z.square(), paddle.zeros_like(x=y), x.mul(y)], axis=1)
    v = paddle.concat(x=[paddle.zeros_like(x=x), x.add(y.pow(y=3)), y.mul(z)], axis=1)
    w = paddle.concat(x=[-2 * y * z**2, z**2, x * y**2 - x**2 - x * y**3], axis=1).neg_()

    lb_uv = U.lie_bracket(u, v)
    assert paddle.allclose(x=U.lie_bracket(v, u), y=lb_uv.neg()).item()
    assert U.lie_bracket(u, u).abs().less_than(paddle.to_tensor(1e-06)).all()
    error = difference(lb_uv, w).abs()
    assert error[:, :, 1:-1, 1:-1, 1:-1].max().less_than(y=paddle.to_tensor(1e-05))
    assert error.max().less_than(y=paddle.to_tensor(0.134))


def test_flow_logv() -> None:
    size = 128, 128, 128
    generator = paddle.framework.core.default_cpu_generator().manual_seed(42)
    v = random_svf(size, stride=8, generator=generator).multiply_(y=paddle.to_tensor(0.1))
    u = U.expv(v)
    w = U.logv(u)
    error = w.sub(v).norm(axis=1, keepdim=True)
    assert error.mean().less_than(y=paddle.to_tensor(0.002))
    assert error.max().less_than(y=paddle.to_tensor(0.06))


def test_flow_compose_svfs() -> None:
    # 3D flow fields
    p = U.move_dim(Grid(size=(64, 32, 16)).coords().unsqueeze_(0), -1, 1)
    start_25 = p.shape[1] + 0 if 0 < 0 else 0
    x = paddle.slice(p, [1], [start_25], [start_25 + 1])
    start_26 = p.shape[1] + 1 if 1 < 0 else 1
    y = paddle.slice(p, [1], [start_26], [start_26 + 1])
    start_27 = p.shape[1] + 2 if 2 < 0 else 2
    z = paddle.slice(p, [1], [start_27], [start_27 + 1])

    with pytest.raises(ValueError):
        U.compose_svfs(p, p, bch_terms=-1)
    with pytest.raises(NotImplementedError):
        U.compose_svfs(p, p, bch_terms=6)

    # u = [yz, xz, xy] and v = u
    u = v = paddle.concat(x=[y.mul(z), x.mul(z), x.mul(y)], axis=1)

    w = U.compose_svfs(u, v, bch_terms=0)
    assert paddle.allclose(x=w, y=u.add(v)).item()
    w = U.compose_svfs(u, v, bch_terms=1)
    assert paddle.allclose(x=w, y=u.add(v)).item()
    w = U.compose_svfs(u, v, bch_terms=2)
    assert paddle.allclose(x=w, y=u.add(v)).item()
    w = U.compose_svfs(u, v, bch_terms=3)
    assert paddle.allclose(x=w, y=u.add(v)).item()
    w = U.compose_svfs(u, v, bch_terms=4)
    assert paddle.allclose(x=w, y=u.add(v), atol=1e-05).item()
    w = U.compose_svfs(u, v, bch_terms=5)
    assert paddle.allclose(x=w, y=u.add(v), atol=1e-05).item()

    # u = [yz, xz, xy] and v = [x, y, z]
    u = paddle.concat(x=[y.mul(z), x.mul(z), x.mul(y)], axis=1)
    v = paddle.concat(x=[x, y, z], axis=1)
    w = U.compose_svfs(u, v, bch_terms=0)
    assert paddle.allclose(x=w, y=u.add(v)).item()
    w = U.compose_svfs(u, v, bch_terms=1)
    assert paddle.allclose(x=w, y=u.mul(0.5).add(v), atol=2e-06).item()

    # u = random_svf(), u -> 0 at boundary
    # v = random_svf(), v -> 0 at boundary
    size = 64, 64, 64
    generator = paddle.framework.core.default_cpu_generator().manual_seed(42)
    u = random_svf(size, stride=4, generator=generator).multiply_(y=paddle.to_tensor(0.1))
    v = random_svf(size, stride=4, generator=generator).multiply_(y=paddle.to_tensor(0.05))
    w = U.compose_svfs(u, v, bch_terms=5)

    flow_u = U.expv(u)
    flow_v = U.expv(v)
    flow_w = U.expv(w)
    flow = U.compose_flows(flow_u, flow_v)

    error = flow_w.sub(flow).norm(axis=1)
    assert error.max().less_than(y=paddle.to_tensor(0.01))
