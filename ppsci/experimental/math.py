# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle


def bessel_i0(x: paddle.Tensor) -> paddle.Tensor:
    """Zero-order modified Bézier curve functions of the first kind.

    Args:
        x (paddle.Tensor): Input data of the formula.

    Examples:
        >>> import paddle
        >>> import ppsci
        >>> res = ppsci.experimental.bessel_i0(paddle.to_tensor([0, 1, 2, 3, 4], dtype="float32"))
    """
    return paddle.i0(x)


def bessel_i0e(x: paddle.Tensor) -> paddle.Tensor:
    """Exponentially scaled zero-order modified Bézier curve functions of the first kind.

    Args:
        x (paddle.Tensor): Input data of the formula.

    Examples:
        >>> import paddle
        >>> import ppsci
        >>> res = ppsci.experimental.bessel_i0e(paddle.to_tensor([0, 1, 2, 3, 4], dtype="float32"))
    """
    return paddle.i0e(x)


def bessel_i1(x: paddle.Tensor) -> paddle.Tensor:
    """First-order modified Bézier curve functions of the first kind.

    Args:
        x (paddle.Tensor): Input data of the formula.

    Examples:
        >>> import paddle
        >>> import ppsci
        >>> res = ppsci.experimental.bessel_i1(paddle.to_tensor([0, 1, 2, 3, 4], dtype="float32"))
    """
    return paddle.i1(x)


def bessel_i1e(x: paddle.Tensor) -> paddle.Tensor:
    """Exponentially scaled first-order modified Bézier curve functions of the first kind.

    Args:
        x (paddle.Tensor): Input data of the formula.

    Examples:
        >>> import paddle
        >>> import ppsci
        >>> res = ppsci.experimental.bessel_i1e(paddle.to_tensor([0, 1, 2, 3, 4], dtype="float32"))
    """
    return paddle.i1e(x)
