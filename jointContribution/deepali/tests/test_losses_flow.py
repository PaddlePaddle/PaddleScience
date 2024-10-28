import deepali.core.functional as U
import deepali.losses.functional as L
import paddle
from deepali.core import Grid
from deepali.utils import paddle_aux  # noqa


def test_losses_flow_bending() -> None:
    data = paddle.randn(shape=(1, 2, 16, 24))
    a = L.bending_energy(data, mode="bspline", stride=1, reduction="none")
    b = L.bspline_bending_energy(data, stride=1, reduction="none")
    assert paddle.allclose(x=a, y=b).item()


def test_losses_flow_curvature() -> None:
    grid = Grid(size=(16, 14))
    offset = U.translation([0.1, 0.2]).unsqueeze_(0)
    flow = U.affine_flow(offset, grid)
    flow.stop_gradient = not True
    loss = L.curvature_loss(flow)
    assert not loss.stop_gradient
    loss = loss.detach()
    assert loss.abs().max().less_than(y=paddle.to_tensor(1e-05))
