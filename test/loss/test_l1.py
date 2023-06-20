import numpy as np
import paddle
import pytest
from paddle import nn

from ppsci import loss

__all__ = []


@pytest.mark.parametrize("reduction", ("mean", "sum", "dummy"))
@pytest.mark.parametrize(
    "loss_weight", (None, 3.1415926, {"u": 1.314, "v": 0.795, "w": 0.002})
)
@pytest.mark.parametrize("use_weight_dict", (None, {}, "scalar_dict", "tensor_dict"))
@pytest.mark.parametrize("use_area", (True, False))
def test_l1loss(reduction, loss_weight, use_weight_dict, use_area):
    """Test for L1Loss."""
    batch_size = 13
    input_dims = ("x", "y", "z")
    output_dims = ("u", "v", "w")

    # generate input data
    x = paddle.randn([batch_size, 1])
    y = paddle.randn([batch_size, 1])
    z = paddle.randn([batch_size, 1])
    x.stop_gradient = False
    y.stop_gradient = False
    z.stop_gradient = False
    input_data = paddle.concat([x, y, z], axis=1)

    # build NN model
    model = nn.Sequential(
        nn.Linear(len(input_dims), len(output_dims)),
        nn.Tanh(),
    )

    # manually generate output
    output_data = model(input_data)
    u, v, w = paddle.split(output_data, len(output_dims), axis=1)

    def jacobian(y: "paddle.Tensor", x: "paddle.Tensor") -> "paddle.Tensor":
        return paddle.grad(y, x, create_graph=True)[0]

    def hessian(y: "paddle.Tensor", x: "paddle.Tensor") -> "paddle.Tensor":
        return jacobian(jacobian(y, x), x)

    u__x__x = hessian(u, x)

    # manually build output_dict, label_dict, weight_dict
    output_dict = {
        "u": u,
        "v": v,
        "w": w,
        "u__x__x": u__x__x,
    }
    label_dict = {k: paddle.randn([batch_size, 1]) for k in output_dict}
    if use_area:
        output_dict["area"] = paddle.randn(label_dict["u"].shape)

    if use_weight_dict is None:
        weight_dict = None
    elif use_weight_dict == {}:
        weight_dict = {}
    elif use_weight_dict == "tensor_dict":
        weight_dict = {k: paddle.randn([batch_size, 1]) for k in label_dict}
    else:
        weight_dict = {k: np.random.randn() for k in label_dict}

    # compute expected result
    expected_result = 0
    for key in label_dict:
        loss_ = paddle.abs(output_dict[key] - label_dict[key])
        if weight_dict:
            loss_ *= weight_dict[key]

        if use_area:
            loss_ *= output_dict["area"]

        loss_ = loss_.sum(axis=1)

        if reduction == "mean":
            loss_ = paddle.mean(loss_)
        else:
            loss_ = paddle.sum(loss_)

        if isinstance(loss_weight, (float, int)):
            loss_ *= loss_weight
        elif isinstance(loss_weight, dict) and key in loss_weight:
            loss_ *= loss_weight[key]

        expected_result += loss_

    if reduction == "dummy":
        try:
            _ = loss.L1Loss(reduction, loss_weight)(
                output_dict, label_dict, weight_dict
            )
        except Exception as e:
            assert isinstance(e, ValueError)
    else:
        test_result = loss.L1Loss(reduction, loss_weight)(
            output_dict, label_dict, weight_dict
        )
        assert paddle.allclose(expected_result, test_result)


if __name__ == "__main__":
    pytest.main()
