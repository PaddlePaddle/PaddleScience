import paddle

from paddle_geometric import is_compiling


def is_in_onnx_export() -> bool:
    r"""Returns :obj:`True` in case PaddlePaddle is exporting to ONNX via
    :meth:`paddle.onnx.export`.
    """
    if is_compiling():
        return False
    if paddle.jit.to_static:  # Paddle 没有完全等价于 `torch.jit.is_scripting` 的函数，用 `to_static` 替代
        return False
    return paddle.onnx.is_in_onnx_export()
