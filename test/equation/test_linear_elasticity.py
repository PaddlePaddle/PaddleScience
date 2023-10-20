import paddle
import pytest
import sympy as sp

import ppsci
from ppsci import arch
from ppsci import equation


def jacobian(y: paddle.Tensor, x: paddle.Tensor) -> paddle.Tensor:
    return paddle.grad(y, x, create_graph=True)[0]


def hessian(y: paddle.Tensor, x: paddle.Tensor) -> paddle.Tensor:
    return jacobian(jacobian(y, x), x)


def stress_disp_xx_expected_result(u, v, w, x, y, z, lambda_, mu, dim, sigma_xx):
    stress_disp_xx = (
        lambda_ * (jacobian(u, x) + jacobian(v, y)) + 2 * mu * jacobian(u, x) - sigma_xx
    )
    if dim == 3:
        stress_disp_xx += lambda_ * jacobian(w, z)
    return stress_disp_xx


def stress_disp_yy_expected_result(u, v, w, x, y, z, lambda_, mu, dim, sigma_yy):
    stress_disp_yy = (
        lambda_ * (jacobian(u, x) + jacobian(v, y)) + 2 * mu * jacobian(v, y) - sigma_yy
    )
    if dim == 3:
        stress_disp_yy += lambda_ * jacobian(w, z)
    return stress_disp_yy


def stress_disp_zz_expected_result(u, v, w, x, y, z, lambda_, mu, sigma_zz):
    stress_disp_zz = (
        lambda_ * (jacobian(u, x) + jacobian(v, y) + jacobian(w, z))
        + 2 * mu * jacobian(w, z)
        - sigma_zz
    )
    return stress_disp_zz


def stress_disp_xy_expected_result(u, v, x, y, mu, sigma_xy):
    stress_disp_xy = mu * (jacobian(u, y) + jacobian(v, x)) - sigma_xy
    return stress_disp_xy


def stress_disp_xz_expected_result(u, w, x, z, mu, sigma_xz):
    stress_disp_xz = mu * (jacobian(u, z) + jacobian(w, x)) - sigma_xz
    return stress_disp_xz


def stress_disp_yz_expected_result(v, w, y, z, mu, sigma_yz):
    stress_disp_yz = mu * (jacobian(v, z) + jacobian(w, y)) - sigma_yz
    return stress_disp_yz


def equilibrium_x_expected_result(
    u, x, y, z, t, rho, dim, time, sigma_xx, sigma_xy, sigma_xz=None
):
    equilibrium_x = -jacobian(sigma_xx, x) - jacobian(sigma_xy, y)
    if dim == 3:
        equilibrium_x -= jacobian(sigma_xz, z)
    if time:
        equilibrium_x += rho * hessian(u, t)
    return equilibrium_x


def equilibrium_y_expected_result(
    v, x, y, z, t, rho, dim, time, sigma_yy, sigma_xy, sigma_yz=None
):
    equilibrium_y = -jacobian(sigma_xy, x) - jacobian(sigma_yy, y)
    if dim == 3:
        equilibrium_y -= jacobian(sigma_yz, z)
    if time:
        equilibrium_y += rho * hessian(v, t)
    return equilibrium_y


def equilibrium_z_expected_result(
    w, x, y, z, t, rho, time, sigma_xz, sigma_yz, sigma_zz
):
    equilibrium_z = (
        -jacobian(sigma_xz, x) - jacobian(sigma_yz, y) - jacobian(sigma_zz, z)
    )
    if time:
        equilibrium_z += rho * hessian(w, t)
    return equilibrium_z


def traction_x_expected_result(
    normal_x, normal_y, sigma_xx, sigma_xy, normal_z=None, sigma_xz=None
):
    traction_x = normal_x * sigma_xx + normal_y * sigma_xy
    if normal_z is not None and sigma_xz is not None:
        traction_x += normal_z * sigma_xz
    return traction_x


def traction_y_expected_result(
    normal_x, normal_y, sigma_xy, sigma_yy, normal_z=None, sigma_yz=None
):
    traction_y = normal_x * sigma_xy + normal_y * sigma_yy
    if normal_z is not None and sigma_yz is not None:
        traction_y += normal_z * sigma_yz
    return traction_y


def traction_z_expected_result(
    normal_x, normal_y, normal_z, sigma_xz, sigma_yz, sigma_zz
):
    traction_z = normal_x * sigma_xz + normal_y * sigma_yz + normal_z * sigma_zz
    return traction_z


@pytest.mark.parametrize(
    "E, nu, lambda_, mu, rho, dim, time",
    [
        (None, None, 1e3, 1e3, 1, 2, False),
        (None, None, 1e3, 1e3, 1, 2, True),
        (None, None, 1e3, 1e3, 1, 3, False),
        (None, None, 1e3, 1e3, 1, 3, True),
    ],
)
def test_linear_elasticity(E, nu, lambda_, mu, rho, dim, time):
    paddle.seed(42)
    batch_size = 13
    input_dims = ("x", "y", "z")[:dim]
    if time:
        input_dims += ("t",)
    output_dims = (
        (
            "u",
            "v",
            "sigma_xx",
            "sigma_yy",
            "sigma_xy",
        )
        if dim == 2
        else (
            "u",
            "v",
            "w",
            "sigma_xx",
            "sigma_yy",
            "sigma_xy",
            "sigma_zz",
            "sigma_xz",
            "sigma_yz",
        )
    )
    x = paddle.randn([batch_size, 1])
    y = paddle.randn([batch_size, 1])
    z = paddle.randn([batch_size, 1]) if dim == 3 else None
    t = paddle.randn([batch_size, 1]) if time else None
    normal_x = paddle.randn([batch_size, 1])
    normal_y = paddle.randn([batch_size, 1])
    normal_z = paddle.randn([batch_size, 1]) if dim == 3 else None

    x.stop_gradient = False
    y.stop_gradient = False
    if time:
        t.stop_gradient = False
    if dim == 3:
        z.stop_gradient = False

    input_data = paddle.concat([x, y], axis=1)
    if time:
        input_data = paddle.concat([t, input_data], axis=1)
    if dim == 3:
        input_data = paddle.concat([input_data, z], axis=1)

    model = arch.MLP(input_dims, output_dims, 2, 16)

    # model = nn.Sequential(
    #     nn.Linear(input_data.shape[1], 9 if dim == 3 else 5),
    #     nn.Tanh(),
    # )

    output = model.forward_tensor(input_data)

    u, v, *other_outputs = paddle.split(output, num_or_sections=output.shape[1], axis=1)

    if dim == 3:
        w = other_outputs[0]
        sigma_xx, sigma_xy, sigma_xz, sigma_yy, sigma_yz, sigma_zz = other_outputs[1:]
    else:
        w = None
        sigma_xx, sigma_xy, sigma_yy = other_outputs[0:3]
        sigma_xz, sigma_yz, sigma_zz = None, None, None

    expected_stress_disp_xx = stress_disp_xx_expected_result(
        u, v, w, x, y, z, lambda_, mu, dim, sigma_xx
    )
    expected_stress_disp_yy = stress_disp_yy_expected_result(
        u, v, w, x, y, z, lambda_, mu, dim, sigma_yy
    )
    expected_stress_disp_xy = stress_disp_xy_expected_result(u, v, x, y, mu, sigma_xy)
    expected_equilibrium_x = equilibrium_x_expected_result(
        u, x, y, z, t, rho, dim, time, sigma_xx, sigma_xy, sigma_xz
    )
    expected_equilibrium_y = equilibrium_y_expected_result(
        v, x, y, z, t, rho, dim, time, sigma_yy, sigma_xy, sigma_yz
    )
    expected_traction_x = traction_x_expected_result(
        normal_x, normal_y, sigma_xx, sigma_xy, normal_z, sigma_xz
    )
    expected_traction_y = traction_y_expected_result(
        normal_x, normal_y, sigma_xy, sigma_yy, normal_z, sigma_yz
    )
    if dim == 3:
        expected_stress_disp_zz = stress_disp_zz_expected_result(
            u, v, w, x, y, z, lambda_, mu, sigma_zz
        )
        expected_stress_disp_xz = stress_disp_xz_expected_result(
            u, w, x, z, mu, sigma_xz
        )
        expected_stress_disp_yz = stress_disp_yz_expected_result(
            v, w, y, z, mu, sigma_yz
        )
        expected_equilibrium_z = equilibrium_z_expected_result(
            w, x, y, z, t, rho, time, sigma_xz, sigma_yz, sigma_zz
        )
        expected_traction_z = traction_z_expected_result(
            normal_x, normal_y, normal_z, sigma_xz, sigma_yz, sigma_zz
        )

    linear_elasticity = equation.LinearElasticity(
        E=E, nu=nu, lambda_=lambda_, mu=mu, rho=rho, dim=dim, time=time
    )
    for name, expr in linear_elasticity.equations.items():
        if isinstance(expr, sp.Basic):
            linear_elasticity.equations[name] = ppsci.lambdify(
                expr,
                model,
            )
    data_dict = {
        "t": t,
        "x": x,
        "y": y,
        "z": z,
        "u": u,
        "v": v,
        "w": w,
        "sigma_xx": sigma_xx,
        "sigma_xy": sigma_xy,
        "sigma_xz": sigma_xz,
        "sigma_yy": sigma_yy,
        "sigma_yz": sigma_yz,
        "sigma_zz": sigma_zz,
        "normal_x": normal_x,
        "normal_y": normal_y,
        "normal_z": normal_z,
    }
    if not time:
        data_dict.pop("t")
    if dim == 2:
        data_dict.pop("w")
        data_dict.pop("sigma_xz")
        data_dict.pop("sigma_yz")
        data_dict.pop("sigma_zz")
        data_dict.pop("normal_z")

    test_output_names = [
        "stress_disp_xx",
        "stress_disp_yy",
        "stress_disp_xy",
        "equilibrium_x",
        "equilibrium_y",
        "traction_x",
        "traction_y",
    ]

    if dim == 3:
        test_output_names.extend(
            [
                "stress_disp_zz",
                "stress_disp_xz",
                "stress_disp_yz",
                "equilibrium_z",
                "traction_z",
            ]
        )

    test_output = {}
    for name in test_output_names:
        test_output[name] = linear_elasticity.equations[name](data_dict)

    expected_output = {
        "stress_disp_xx": expected_stress_disp_xx,
        "stress_disp_yy": expected_stress_disp_yy,
        "stress_disp_xy": expected_stress_disp_xy,
        "equilibrium_x": expected_equilibrium_x,
        "equilibrium_y": expected_equilibrium_y,
        "traction_x": expected_traction_x,
        "traction_y": expected_traction_y,
    }
    if dim == 3:
        expected_output.update(
            {
                "stress_disp_zz": expected_stress_disp_zz,
                "stress_disp_xz": expected_stress_disp_xz,
                "stress_disp_yz": expected_stress_disp_yz,
                "equilibrium_z": expected_equilibrium_z,
                "traction_z": expected_traction_z,
            }
        )

    for name in test_output_names:
        assert paddle.allclose(expected_output[name], test_output[name], atol=1e-7)


if __name__ == "__main__":
    pytest.main()
