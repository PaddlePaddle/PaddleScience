import paddle
import pytest
from neuralop.models import FNOGNO
from tensorly import tenalg

tenalg.set_backend("einsum")


@pytest.mark.parametrize(
    "gno_transform_type", ["linear", "nonlinear_kernelonly", "nonlinear"]
)
@pytest.mark.parametrize("fno_n_modes", [(8,), (8, 8), (8, 8, 8)])
def test_fnogno(gno_transform_type, fno_n_modes):
    if paddle.device.cuda.device_count() >= 1:
        device = "gpu:0"
    else:
        device = "cpu"

    paddle.set_device(device)
    in_channels = 3
    out_channels = 2
    n_dim = len(fno_n_modes)
    model = FNOGNO(
        in_channels=in_channels,
        out_channels=out_channels,
        gno_radius=0.2,
        gno_coord_dim=n_dim,
        gno_transform_type=gno_transform_type,
        fno_n_modes=fno_n_modes,
        fno_norm="ada_in",
        fno_ada_in_features=4,
    )

    in_p_shape = [
        32,
    ] * n_dim
    in_p_shape.append(n_dim)
    in_p = paddle.randn(in_p_shape)

    out_p = paddle.randn([100, n_dim])

    f_shape = [
        32,
    ] * n_dim
    f_shape.append(in_channels)
    f = paddle.randn(f_shape)

    ada_in = paddle.randn(
        [
            1,
        ]
    )

    # Test forward pass
    out = model(in_p, out_p, f, ada_in)

    # Check output size
    assert list(out.shape) == [100, out_channels]

    # Check backward pass
    loss = out.sum()
    loss.backward()

    n_unused_params = 0
    for param in model.parameters():
        if param.grad is None:
            n_unused_params += 1
    assert n_unused_params == 0, f"{n_unused_params} parameters were unused!"
