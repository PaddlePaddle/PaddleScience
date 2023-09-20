"""
https://github.com/PredictiveIntelligenceLab/PINNsNTK/blob/18ef519e1fe924e32ef96d4ba9a814b480bd9b05/Compute_Jacobian.py
"""
import paddle


def jacobian(output, inputs):
    """Computes jacobian of `output` w.r.t. `inputs`.
    Args:
        output: A tensor.
        inputs: A tensor or a nested structure of tensor objects.
    Returns:
        A tensor or a nested structure of tensors with the same structure as
        `inputs`.
    """
    output_size = int(output.shape[0])
    result = []
    for i in range(output_size):
        out = paddle.grad(output[i], inputs, allow_unused=True)[0]
        if out is None:
            out = paddle.to_tensor([0.0]).broadcast_to(inputs.shape)
        result.append(out)
    return paddle.to_tensor(result)
