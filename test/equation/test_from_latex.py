import numpy as np
import paddle
import pytest

import ppsci
from ppsci import arch
from ppsci import equation

__all__ = []


@pytest.mark.parametrize("dim", (3,))
def test_l1loss_mean(dim):
    """Test for only mean."""
    batch_size = 13
    input_dims = ("x", "y", "z")[:dim]
    output_dims = ("u", "v", "w")[:dim] + ("p",)

    # generate input data
    x = paddle.randn([batch_size, 1])
    y = paddle.randn([batch_size, 1])
    x.stop_gradient = False
    y.stop_gradient = False
    if dim == 3:
        z = paddle.randn([batch_size, 1])
        z.stop_gradient = False

    # build NN model
    model = arch.MLP(input_dims, output_dims, 2, 16)

    # manually generate output
    eq1 = equation.PDE.from_latex(
        r"\frac{d}{dx}(u{(t,x,y,z)}) + \frac{d}{dy}(v{(t,x,y,z)}) + \frac{d}{dz}(w{(t,x,y,z)})"
    )

    eq2 = equation.NavierStokes(1, 1, 3, True)

    eq1_out = ppsci.lambdify(
        eq1.equations["expr_1"],
        model,
    )({"x": x, "y": y, "z": z})
    eq2_out = ppsci.lambdify(
        eq2.equations["continuity"],
        model,
    )({"x": x, "y": y, "z": z})

    np.testing.assert_allclose(eq1_out.numpy(), eq2_out.numpy())


if __name__ == "__main__":
    pytest.main()
