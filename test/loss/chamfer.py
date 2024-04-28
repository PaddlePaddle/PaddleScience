import numpy as np
import paddle
import pytest

from ppsci import loss

__all__ = []


def test_chamfer_loss():
    """Test for chamfer distance loss."""
    N1 = 100
    N2 = 50
    output_dict = {"s1": paddle.randn([1, N1, 3])}
    label_dict = {"s1": paddle.randn([1, N2, 3])}
    chamfer_loss = loss.ChamferLoss()
    result = chamfer_loss(output_dict, label_dict)

    loss_cd_s1 = 0.0
    for i in range(N1):
        min_i = None
        for j in range(N2):
            disij = ((output_dict["s1"][0, i] - label_dict["s1"][0, j]) ** 2).sum()
            if min_i is None or disij < min_i:
                min_i = disij
        loss_cd_s1 += min_i
    loss_cd_s1 /= N1

    loss_cd_s2 = 0.0
    for j in range(N2):
        min_j = None
        for i in range(N1):
            disij = ((output_dict["s1"][0, i] - label_dict["s1"][0, j]) ** 2).sum()
            if min_j is None or disij < min_j:
                min_j = disij
        loss_cd_s2 += min_j
    loss_cd_s2 /= N2

    loss_cd = loss_cd_s1 + loss_cd_s2
    np.testing.assert_allclose(loss_cd.item(), result.item())


if __name__ == "__main__":
    pytest.main()
