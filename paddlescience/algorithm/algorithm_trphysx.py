# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import os
import paddle

from .algorithm_base import AlgorithmBase


class TrPhysx(AlgorithmBase):
    """
    Methods related with the TrPhysx algorithm. 
    """

    def __init__(self, net, **kwargs):
        super(TrPhysx, self).__init__()
        self.net = net

    def compute(self, inputs, **kwargs):
        """ model training """
        self.net.train()
        losses = self.net.compute_loss(inputs, **kwargs)
        return losses

    @paddle.no_grad()
    def eval(self, inputs, **kwargs):
        """ model evalution """
        self.net.eval()
        losses = self.net.evaluate(inputs, **kwargs)
        return losses
