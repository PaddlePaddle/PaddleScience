import paddle
import pytest

from ppsci import loss

__all__ = []


def test_non_tensor_return_type():
    """Test for biharmonic equation."""

    def loss_func_return_tensor(input_dict, label_dict, weight_dict):
        return (0.5 * (input_dict["x"] - label_dict["x"]) ** 2).sum()

    def loss_func_reuturn_builtin_float(input_dict, label_dict, weight_dict):
        return (0.5 * (input_dict["x"] - label_dict["x"]) ** 2).sum().item()

    wrapped_loss1 = loss.FunctionalLoss(loss_func_return_tensor)
    wrapped_loss2 = loss.FunctionalLoss(loss_func_reuturn_builtin_float)

    input_dict = {"x": paddle.randn([10, 1])}
    label_dict = {"x": paddle.zeros([10, 1])}

    wrapped_loss1(input_dict, label_dict)
    with pytest.raises(AssertionError):
        wrapped_loss2(input_dict, label_dict)


if __name__ == "__main__":
    pytest.main()
