from typing import Any
import paddle


def is_mps_available() -> bool:
    """
    Returns a bool indicating if Metal Performance Shaders (MPS) is currently available in PaddlePaddle.
    Note: PaddlePaddle does not support MPS directly, so this always returns False.
    """
    # Placeholder for PaddlePaddle as it doesn't support MPS
    return False


def is_xpu_available() -> bool:
    """
    Returns a bool indicating if XPU (Intel Extension for PaddlePaddle) is currently available.
    """
    try:
        from paddle_xpu import is_compiled_with_xpu
        return is_compiled_with_xpu()
    except ImportError:
        return False


def device(device: Any) -> paddle.device:
    """
    Returns a PaddlePaddle device.

    If 'auto' is specified, returns the optimal device depending on available hardware.
    """
    if device != 'auto':
        return paddle.device.set_device(device)
    if paddle.device.is_compiled_with_cuda():
        return paddle.device.set_device('gpu')
    if is_xpu_available():
        return paddle.device.set_device('xpu')
    return paddle.device.set_device('cpu')