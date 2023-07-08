import paddle
import pytest
from paddle import nn

from ppsci import equation


def jacobian(y: paddle.Tensor, x: paddle.Tensor) -> paddle.Tensor:
    return paddle.grad(y, x, create_graph=True)[0]

def hessian(y: paddle.Tensor, x: paddle.Tensor) -> paddle.Tensor:
    return jacobian(jacobian(y, x), x)

def continuity_compute_func_expected_result(x, y, u, v, dim, w=None, z=None):                 
    continuity = jacobian(u, x) + jacobian(v, y)
    if self.dim == 3:
        continuity += jacobian(w, z)
    return continuity

def momentum_x_compute_func_expected_result(nu, p, rho, x, y, u, v, dim, time= False, w=None, z=None, t=None):
    momentum_x = 
        u * jacobian(u, x)
        + v * jacobian(u, y)
        - nu / rho * hessian(u, x)
        - nu / rho * hessian(u, y)
        + 1 / rho * jacobian(p, x)
    
    if time is True:
        momentum_x += jacobian(u, t)
    if dim == 3:
        momentum_x += w * jacobian(u, z)
        momentum_x -= nu / rho * hessian(u, z)
    return momentum_x

def momentum_y_compute_func_expected_result(nu, p, rho, x, y, u, v, dim, time= False, w=None, z=None, t=None):
    momentum_y = 
        u * jacobian(v, x)
        + v * jacobian(v, y)
        - nu / rho * hessian(v, x)
        - nu / rho * hessian(v, y)
        + 1 / rho * jacobian(p, y)
    
    if time is True:
        momentum_y += jacobian(v, t)
    if dim == 3:
        momentum_y += w * jacobian(v, z)
        momentum_y -= nu / rho * hessian(v, z)
    return momentum_y

def momentum_z_compute_func_expected_result(nu, p, rho, x, y, u, v, dim, time= False, w=None, z=None, t=None):
    momentum_z = 
        u * jacobian(w, x)
        + v * jacobian(w, y)
        + w * jacobian(w, z)
        - nu / rho * hessian(w, x)
        - nu / rho * hessian(w, y)
        - nu / rho * hessian(w, z)
        + 1 / rho * jacobian(p, z)
    
    if time is True:
        momentum_z += jacobian(w, t)
    return momentum_z

@pytest.mark.parametrize("dim,time", [(2, 3), (True, False)])
def test_navier_stokes(nu, rho, dim, time):
    batch_size = 13
    # generate input data
    x = paddle.randn([batch_size, 1])
    y = paddle.randn([batch_size, 1])
    x.stop_gradient = False
    y.stop_gradient = False
    input_dims = 2
    input_data = paddle.concat([x, y], axis=1)
    if time is True:
        t = paddle.randn([batch_size, 1])
        t.stop_gradient = False
        input_data = paddle.concat([t, input_data], axis=1)
        input_dims += 1
    if dim == 3:
        z = paddle.randn([batch_size, 1])
        z.stop_gradient = False
        input_data = paddle.concat([input_data, z], axis=1)
        input_dims += 1
    model = nn.Sequential(
        nn.Linear(len(input_dims), 1),
        nn.Tanh(),
    )

    # manually generate output
    output = model(input_data)

    expected_result_continuity_compute_func = continuity_compute_func_expected_result(x, y, u, v, dim, w=None, z=None)
    expected_result_momentum_x_compute_func = momentum_x_compute_func_expected_result(nu, p, rho, x, y, u, v, dim, time= False, w=None, z=None, t=None)
    expected_result_momentum_x_compute_func = momentum_x_compute_func_expected_result(nu, p, rho, x, y, u, v, dim, time= False, w=None, z=None, t=None)
    expected_result_momentum_x_compute_func = momentum_x_compute_func_expected_result(nu, p, rho, x, y, u, v, dim, time= False, w=None, z=None, t=None)

    # compute result using NavierStokes class
    navier_stokes_equation = equation.NavierStokes(nu=nu, rho=rho, dim=dim, time=time)
    if time == True:
        expected_result += jacobian(eta, t)
    if dim == 3:
        expected_result += w * jacobian(eta, z)
        expected_result -= nu / rho * hessian(eta, z)
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
