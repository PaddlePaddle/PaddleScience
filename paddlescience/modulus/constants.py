"""
constant values used by Modulus
"""

import paddle
import numpy as np

# string used to determine derivatives
diff_str: str = "__"

# function to apply diff string
def diff(y: str, x: str, degree: int = 1) -> str:
    return diff_str.join([y] + degree * [x])


# for changing to float16 or float64
paddle_dt = paddle.float32
np_dt = np.float32

# tensorboard naming
TF_SUMMARY = False

# Pytorch Version for which JIT will be default on
# Torch version of NGC container 22.05
JIT_PYTORCH_VERSION = "1.12.0a0+8a1a93a"
