import torch

__all__ = []


input = [[1.0, 2.0], [3.0, 4.0]]
# input = [[1.0, 2.0],]
print("input = ", input)
x = torch.tensor(input, requires_grad=True)
print("\ninput.shape = ", x.shape)

# y = x
def exp_reducer(x):
    return x.exp()


J = torch.autograd.functional.jacobian(exp_reducer, x)
print("torch Jacobian matrix = ", J)
print("torch Jacobian shape  = ", J.shape)


# paddle.fluid.core.set_prim_eager_enabled(True)
# input = ((1.0, 2.0),)
# x = paddle.to_tensor(input)
# print("\ninput.shape = ", x.shape)

# x.stop_gradient = False
# y = x.exp()
# J = jacobian(y, x)
# print("paddle Jacobian matrix= ", J)

# input_list = [
#     ((1.0, 2.0),),
#     ((1.0, 2.0),(3.0, 4.0)),
# ]
# @pytest.mark.parametrize("input",  input_list)
# def test_matrix_jacobian(input):
#     x = paddle.to_tensor(input)
#     print("\ninput.shape = ", x.shape)
#     x.stop_gradient = False
#     y = x
#     J = jacobian(y, x)
#     test_result = J
#     x = torch.tensor(input, requires_grad=True)
#     def exp_reducer(x):
#         return x
#     J = torch.autograd.functional.jacobian(exp_reducer, x)
#     expected_result = paddle.to_tensor(J.numpy())

#     # check result whether is equal
#     assert paddle.allclose(expected_result, test_result)
