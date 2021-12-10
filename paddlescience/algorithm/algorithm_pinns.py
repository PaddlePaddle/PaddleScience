# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from .algorithm_base import AlgorithmBase


class PINNs(AlgorithmBase):
    """
    The Physics Informed Neural Networks Algorithm.

    Parameters:
        net(NetworkBase): The NN network used in PINNs algorithm.
        loss(LossBase): The loss used in PINNs algorithm.

    Example:
        >>> import paddlescience as psci
        >>> algo = psci.algorithm.PINNs(net=net, loss=loss)
    """

    def __init__(self, net, loss):
        super(PINNs, self).__init__()
        self.net = net
        self.loss = loss

    def batch_run(self, batch_id):
        return self.loss.batch_run(self.net, batch_id)
