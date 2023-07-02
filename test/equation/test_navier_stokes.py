import paddle
import pytest
from paddle import nn
from paddle.nn import initializer

from ppsci import equation


@pytest.mark.parametrize("dim,time", [(2, 3), (True, False)])
def test_momentum_x_compute_func(dim, time):
    """Test for navier-stokes equation."""
    batch_size = 13
    nu = 0.1
    rho = 1.0
    # generate input data
    x = paddle.randn([batch_size, 1])
    y = paddle.randn([batch_size, 1])
    u = paddle.randn([batch_size, 1])
    v = paddle.randn([batch_size, 1])
    p = paddle.randn([batch_size, 1])
    x.stop_gradient = False
    y.stop_gradient = False
    u.stop_gradient = False
    v.stop_gradient = False
    p.stop_gradient = False
    input_dims = 5
    input_data = paddle.concat([x, y, u, v, p], axis=1)
    if time == True:
        t = paddle.randn([batch_size, 1])
        t.stop_gradient = False
        input_data = paddle.concat([input_data, t], axis=1)
        input_dims += 1
    if dim == 3:
        z = paddle.randn([batch_size, 1])
        z.stop_gradient = False
        w = paddle.randn([batch_size, 1])
        w.stop_gradient = False
        input_data = paddle.concat([input_data, z, w], axis=1)
        input_dims += 2
    model = nn.Sequential(
        nn.Linear(len(input_dims), 1),
        nn.Tanh(),
    )

    # manually generate output
    eta = model(input_data)

    def jacobian(y: paddle.Tensor, x: paddle.Tensor) -> paddle.Tensor:
        return paddle.grad(y, x, create_graph=True)[0]

    def hessian(y: paddle.Tensor, x: paddle.Tensor) -> paddle.Tensor:
        return jacobian(jacobian(y, x), x)

    expected_result = (
        u * jacobian(u, x)
        + v * jacobian(u, y)
        - nu / rho * hessian(u, x)
        - nu / rho * hessian(u, y)
        + 1 / rho * jacobian(p, x)
    )

    # compute result using NavierStokes class
    navier_stokes_equation = equation.NavierStokes(nu=nu, rho=rho, dim=dim, time=time)
    if time == True:
        expected_result += jacobian(u, t)
    if dim == 3:
        expected_result += w * jacobian(u, z)
        expected_result -= nu / rho * hessian(u, z)
    # compute result using built-in Laplace module
    poisson_equation = equation.NavierStokes(dim=dim)
    data_dict = {
        "x": x,
        "y": y,
        "u": u,
        "v": v,
        "p": p,
    }
    if time == True:
        data_dict["t"] = t
    if dim == 3:
        data_dict["z"] = z
        data_dict["w"] = w
    test_result = poisson_equation.equations["poisson"](data_dict)
    # check result whether is equal
    assert paddle.allclose(expected_result, test_result)


if __name__ == "__main__":
    pytest.main()
