import numpy as np
import paddle
import pytest

import ppsci

paddle.seed(42)
np.random.seed(42)


def test_equation_detach():
    # use N-S equation for test
    all_items = [
        "u",
        "u__x",
        "u__y",
        "u__x__x",
        "v",
        "v__x",
        "v__y",
        "v__x__x",
        "p",
        "p__x",
        "p__y",
    ]
    model1 = ppsci.arch.MLP(("x", "y"), ("u", "v", "p"), 3, 16)
    model2 = ppsci.arch.MLP(("x", "y"), ("u", "v", "p"), 3, 16)
    input_data = {
        "x": paddle.randn([16, 1]),
        "y": paddle.randn([16, 1]),
    }
    input_data["x"].stop_gradient = False
    input_data["y"].stop_gradient = False
    for ii, in_state in enumerate(range(0, 1 << len(all_items), 5)):
        detach_keys = [
            item for i, item in enumerate(all_items) if ((1 << i) & in_state)
        ]
        nu = 1.314
        rho = 0.156
        ns = ppsci.equation.NavierStokes(nu, rho, 2, False, detach_keys=detach_keys)
        model2.set_state_dict(model1.state_dict())

        exprs = ppsci.lambdify(
            list(ns.equations.values()),
            model1,
            fuse_derivative=True,
        )
        for name, f in zip(ns.equations, exprs):
            input_data[name] = f(input_data)

        def compute_loss(data_dict):
            u = data_dict["u"]
            v = data_dict["v"]
            p = data_dict["p"]

            u__x = data_dict["u__x"]
            u__y = data_dict["u__y"]
            u__x__x = data_dict["u__x__x"]
            u__y__y = data_dict["u__y__y"]

            v = data_dict["v"]
            v__x = data_dict["v__x"]
            v__y = data_dict["v__y"]
            v__x__x = data_dict["v__x__x"]
            v__y__y = data_dict["v__y__y"]

            p = data_dict["p"]
            p__x = data_dict["p__x"]
            p__y = data_dict["p__y"]

            if "u" in detach_keys:
                u = u.detach()
            if "v" in detach_keys:
                v = v.detach()
            if "p" in detach_keys:
                p = p.detach()
            if "u__x" in detach_keys:
                u__x = u__x.detach()
            if "u__y" in detach_keys:
                u__y = u__y.detach()
            if "u__x__x" in detach_keys:
                u__x__x = u__x__x.detach()
            if "u__y__y" in detach_keys:
                u__y__y = u__y__y.detach()
            if "v__x" in detach_keys:
                v__x = v__x.detach()
            if "v__y" in detach_keys:
                v__y = v__y.detach()
            if "v__x__x" in detach_keys:
                v__x__x = v__x__x.detach()
            if "v__y__y" in detach_keys:
                v__y__y = v__y__y.detach()
            if "p__x" in detach_keys:
                p__x = p__x.detach()
            if "p__y" in detach_keys:
                p__y = p__y.detach()

            # continuity
            continuity = u__x + v__y
            # momentum_x
            momentum_x = (
                u * u__x + v * u__y - nu * (u__x__x + u__y__y) + (1 / rho) * p__x
            )
            # momentum_y
            momentum_y = (
                u * v__x + v * v__y - nu * (v__x__x + v__y__y) + (1 / rho) * p__y
            )

            return (
                (continuity**2).sum()
                + (momentum_x**2).sum()
                + (momentum_y**2).sum()
            )

        loss1 = compute_loss(input_data)

        loss1.backward()

        ppsci.autodiff.clear()

        input_data = {
            "x": input_data["x"],
            "y": input_data["y"],
        }
        x, y = input_data["x"], input_data["y"]
        t = model2(input_data)
        u, v, p = t["u"], t["v"], t["p"]

        u__x = ppsci.autodiff.jacobian(u, x)
        u__y = ppsci.autodiff.jacobian(u, y)
        u__x__x = ppsci.autodiff.hessian(u, x)
        u__y__y = ppsci.autodiff.hessian(u, y)

        v__x = ppsci.autodiff.jacobian(v, x)
        v__y = ppsci.autodiff.jacobian(v, y)
        v__x__x = ppsci.autodiff.hessian(v, x)
        v__y__y = ppsci.autodiff.hessian(v, y)

        p__x = ppsci.autodiff.jacobian(p, x)
        p__y = ppsci.autodiff.jacobian(p, y)

        loss2 = compute_loss(
            {
                "u": u,
                "v": v,
                "p": p,
                "u__x": u__x,
                "u__y": u__y,
                "u__x__x": u__x__x,
                "u__y__y": u__y__y,
                "v__x": v__x,
                "v__y": v__y,
                "v__x__x": v__x__x,
                "v__y__y": v__y__y,
                "p__x": p__x,
                "p__y": p__y,
            }
        )
        loss2.backward()

        np.testing.assert_allclose(loss1.numpy(), loss2.numpy(), 0.0, 0.0)

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if (p1.grad is None) ^ (p2.grad is None):
                raise AssertionError()
            if p1.grad is not None and p2.grad is not None:
                np.testing.assert_allclose(p1.grad.numpy(), p2.grad.numpy(), 1e-5, 1e-5)

        ppsci.autodiff.clear()
        model1.clear_gradients()
        model2.clear_gradients()


if __name__ == "__main__":
    pytest.main()
