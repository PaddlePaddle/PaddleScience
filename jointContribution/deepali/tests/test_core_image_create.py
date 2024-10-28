import deepali.utils.paddle_aux  # noqa
import paddle
from deepali.core import functional as U


def test_circle_image() -> None:
    data = U.circle_image(size=(64, 33), center=(16, 16))
    assert tuple(data.shape) == (1, 1, 33, 64)  # (N, C, Y, X)

    expected_row_col_sum = paddle.to_tensor(
        data=[
            # fmt: off
            0, 1, 11, 15, 19, 21, 23, 25, 25, 27, 27, 29, 29, 29, 29, 29, 31, 29, 29, 29, 29, 29, 27, 27, 25, 25, 23, 21, 19, 15, 11, 1, 0
            # fmt: on
        ],
        dtype=data.dtype,
    )
    assert data.sum(axis=3).equal(y=expected_row_col_sum).astype("bool").all()
    assert data[:, :, :, :33].sum(axis=2).equal(y=expected_row_col_sum).astype("bool").all()
    assert data[:, :, :, 33:].sum().equal(y=0).astype("bool").all()

    data = U.circle_image(size=(64, 32), center=(16, 16), num=0)
    assert tuple(data.shape) == (1, 32, 64)  # (C, Y, X)

    data = U.circle_image(size=(64, 32), center=(16, 16), num=1)
    assert tuple(data.shape) == (1, 1, 32, 64)  # (N, C, Y, X)

    data = U.circle_image(size=(64, 32), center=(16, 16), num=3)
    assert tuple(data.shape) == (3, 1, 32, 64)  # (N, C, Y, X)


def test_cshape_image() -> None:
    data = U.cshape_image(size=(64, 33), center=(16, 16))
    assert tuple(data.shape) == (1, 1, 33, 64)  # (N, C, Y, X)

    expected_col_sum = paddle.to_tensor(
        data=[
            # fmt: off
            0, 1, 11, 15, 19, 21, 23, 25, 24, 20, 16, 16, 16, 14, 14, 14, 14, 14, 14, 14, 16, 16, 16, 0
            # fmt: on
        ],
        dtype=data.dtype,
    )
    expected_row_sum = paddle.to_tensor(
        data=[
            # fmt: off
            0, 1, 11, 14, 16, 17, 18, 19, 18, 13, 9, 8, 8, 7, 7, 7, 7, 7, 7, 7, 8, 8, 9, 13, 18, 19, 18, 17, 16, 14, 11, 1, 0
            # fmt: on
        ],
        dtype=data.dtype,
    )

    assert data.sum(axis=3).equal(y=expected_row_sum).astype("bool").all()
    assert data[:, :, :, :24].sum(axis=2).equal(y=expected_col_sum).astype("bool").all()
    assert data[:, :, :, 24:].sum().equal(y=0).astype("bool").all()
