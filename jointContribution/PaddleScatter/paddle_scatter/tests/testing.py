from typing import Any

import paddle

reductions = ["sum", "add", "mean", "min", "max"]

dtypes = [paddle.float32, paddle.float64, paddle.int32, paddle.int64]
dtypes_half = [paddle.float16, paddle.bfloat16]
ind_dtypes = [paddle.int32, paddle.int64]
grad_dtypes = [paddle.float32, paddle.float64]

places = ["cpu"]
if paddle.core.is_compiled_with_cuda():
    places.append("gpu")

device = (
    paddle.CUDAPlace(0) if paddle.core.is_compiled_with_cuda() else paddle.CPUPlace()
)


def tensor(x: Any, dtype: paddle.dtype):
    return None if x is None else paddle.to_tensor(x).astype(dtype)
