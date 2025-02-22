import numpy as np
import paddle
import pytest

from ppsci import arch
from ppsci import equation

__all__ = []


@pytest.mark.parametrize("dim", (1, 2, 3))
def test_helmholtz(dim):
    """Test for only mean."""
    nrs = [3, 4, 5][:dim]
    input_keys = ("x", "y", "z")[:dim]
    output_keys = ("u",)

    # generate input data
    input_dict = {
        input_keys[i]: paddle.to_tensor(
            np.random.randn(nrs[i], 1).astype(np.float32), stop_gradient=False
        )
        for i in range(dim)
    }
    model = arch.SPINN(input_keys, output_keys, r=16, num_layers=2, hidden_size=8)
    y = model(input_dict)["u"]
    assert y.shape == [*nrs, 1]
    data_dict = {
        **input_dict,
        "u": y,
    }
    helmholtz_obj = equation.Helmholtz(dim, 1.0, model)

    helmholtz_out = helmholtz_obj.equations["helmholtz"](data_dict)

    # check result whether is equal
    assert helmholtz_out.shape == [*nrs, 1]


if __name__ == "__main__":
    pytest.main()
