import paddle
import pytest
from paddle import nn

from ppsci import equation


def jacobian(y: paddle.Tensor, x: paddle.Tensor) -> paddle.Tensor:
    return paddle.grad(y, x, create_graph=True)[0]


def hessian(y: paddle.Tensor, x: paddle.Tensor) -> paddle.Tensor:
    return jacobian(jacobian(y, x), x)


def continuity_compute_func(x, y, u, v, dim, w=None, z=None):
    continuity = jacobian(u, x) + jacobian(v, y)
    if dim == 3:
        continuity += jacobian(w, z)
    return continuity


def momentum_x_compute_func(
    nu, p, rho, x, y, u, v, dim, time=False, w=None, z=None, t=None
):
    momentum_x = (
        u * jacobian(u, x)
        + v * jacobian(u, y)
        - nu / rho * hessian(u, x)
        - nu / rho * hessian(u, y)
        + 1 / rho * jacobian(p, x)
    )

    if time:
        momentum_x += jacobian(u, t)
    if dim == 3:
        momentum_x += w * jacobian(u, z)
        momentum_x -= nu / rho * hessian(u, z)
    return momentum_x


def momentum_y_compute_func(
    nu, p, rho, x, y, u, v, dim, time=False, w=None, z=None, t=None
):
    momentum_y = (
        u * jacobian(v, x)
        + v * jacobian(v, y)
        - nu / rho * hessian(v, x)
        - nu / rho * hessian(v, y)
        + 1 / rho * jacobian(p, y)
    )

    if time:
        momentum_y += jacobian(v, t)
    if dim == 3:
        momentum_y += w * jacobian(v, z)
        momentum_y -= nu / rho * hessian(v, z)
    return momentum_y


def momentum_z_compute_func(
    nu, p, rho, x, y, u, v, dim, time=False, w=None, z=None, t=None
):
    momentum_z = (
        u * jacobian(w, x)
        + v * jacobian(w, y)
        + w * jacobian(w, z)
        - nu / rho * hessian(w, x)
        - nu / rho * hessian(w, y)
        - nu / rho * hessian(w, z)
        + 1 / rho * jacobian(p, z)
    )
    if time:
        momentum_z += jacobian(w, t)
    return momentum_z


@pytest.mark.parametrize(
    "nu,rho,dim,time",
    [
        (0.1, 1.0, 3, False),
        (0.1, 1.0, 2, False),
        (0.1, 1.0, 3, True),
        (0.1, 1.0, 2, True),
    ],
)
def test_navierstokes(nu, rho, dim, time):
    batch_size = 13
    # generate input data
    x = paddle.randn([batch_size, 1])
    y = paddle.randn([batch_size, 1])
    x.stop_gradient = False
    y.stop_gradient = False
    input_dims = 2
    inputs = (x, y)
    if time:
        t = paddle.randn([batch_size, 1])
        t.stop_gradient = False
        inputs = (t,) + inputs
        input_dims += 1
    if dim == 3:
        z = paddle.randn([batch_size, 1])
        z.stop_gradient = False
        inputs = inputs + (z,)
        input_dims += 1
    input_data = paddle.concat(inputs, axis=1)

    """
    Use the relatively simple Multilayer Perceptron 
    to represent the mapping function from (t, x, y, z) to (u, v, w, p):
    f(x, y) = (u, v, p) or
    f(t, x, y) = (u, v, p) or
    f(t, x, y, z) = (u, v, w, p)
    """
    model = nn.Sequential(
        nn.Linear(input_dims, 3 if dim == 2 else 4),
        nn.Tanh(),
    )

    # manually generate output
    output = model(input_data)

    if dim == 2:
        u, v, p = paddle.split(output, num_or_sections=output.shape[1], axis=1)
        w, z = None, None
    else:
        u, v, w, p = paddle.split(output, num_or_sections=output.shape[1], axis=1)
    if not time:
        t = None
    expected_continuity = continuity_compute_func(x=x, y=y, u=u, v=v, dim=dim, w=w, z=z)
    expected_momentum_x = momentum_x_compute_func(
        nu=nu, p=p, rho=rho, x=x, y=y, u=u, v=v, dim=dim, time=time, w=w, z=z, t=t
    )
    expected_momentum_y = momentum_y_compute_func(
        nu=nu, p=p, rho=rho, x=x, y=y, u=u, v=v, dim=dim, time=time, w=w, z=z, t=t
    )
    if dim == 3:
        expected_momentum_z = momentum_z_compute_func(
            nu=nu, p=p, rho=rho, x=x, y=y, u=u, v=v, dim=dim, time=time, w=w, z=z, t=t
        )

    # compute result using NavierStokes class
    navier_stokes_equation = equation.NavierStokes(nu=nu, rho=rho, dim=dim, time=time)

    data_dict = {"x": x, "y": y, "u": u, "v": v, "p": p}
    if time:
        data_dict["t"] = t
    if dim == 3:
        data_dict["z"] = z
        data_dict["w"] = w

    test_output_names = [
        "continuity",
        "momentum_x",
        "momentum_y",
    ]

    if dim == 3:
        test_output_names.append(
            "momentum_z",
        )

    test_output = {}
    for name in test_output_names:
        test_output[name] = navier_stokes_equation.equations[name](data_dict)

    expected_output = {
        "continuity": expected_continuity,
        "momentum_x": expected_momentum_x,
        "momentum_y": expected_momentum_y,
    }
    if dim == 3:
        expected_output["momentum_z"] = expected_momentum_z

    # check result whether is equal
    for name in test_output_names:
        assert paddle.allclose(expected_output[name], test_output[name])


if __name__ == "__main__":
    pytest.main()
