import os
from typing import Dict
from typing import Tuple

import numpy as np
import paddle

from ppsci.autodiff import jacobian

if __name__ == "__main__":
    paddle.seed(42)
    np.random.seed(42)
    os.chdir(
        "/workspace/wangguan/PaddleScience_Surrogate/examples/pipe"
    )  # working folder

    open_bug = True
    if open_bug is True:
        from paddle.fluid import core

        core.set_prim_eager_enabled(True)
    else:
        pass
    x = paddle.to_tensor(
        np.array([6, 6, 6]).reshape(3, 1), dtype="float32", stop_gradient=False
    )
    y = paddle.to_tensor(
        np.array([6, 6, 6]).reshape(3, 1), dtype="float32", stop_gradient=False
    )
    nu = paddle.to_tensor(
        np.array([6, 6, 6]).reshape(3, 1), dtype="float32", stop_gradient=False
    )
    input_dict = {"x": x, "x": x, "y": y, "nu": nu}

    weight = np.array([1, 2, 3, 4]).reshape(4, 1)
    bias = np.array([6, 6, 6]).reshape(3, 1)

    w_para = paddle.nn.initializer.Assign(weight)
    b_para = paddle.nn.initializer.Assign(bias)

    last_fc = paddle.nn.Linear(
        4,
        1,
        weight_attr=paddle.ParamAttr(initializer=w_para),
        bias_attr=paddle.ParamAttr(initializer=b_para),
    )

    def concat_to_tensor(data_dict, keys, axis=-1):
        data = [data_dict[key] for key in keys]
        return paddle.concat(data, axis)

    def split_to_dict(data_tensor, keys, axis=-1):
        data = paddle.split(data_tensor, len(keys), axis=axis)
        return {key: data[i] for i, key in enumerate(keys)}

    def forward_tensor(last_fc, _x):
        _y = _x
        _y = last_fc(_y)
        return _y

    output = concat_to_tensor(input_dict, ["x", "x", "y", "nu"], axis=-1)
    output = forward_tensor(last_fc, output)
    output = split_to_dict(output, ["v"], axis=-1)
    _v = output["v"]
    d_vdy_paddle = jacobian(_v, y)
