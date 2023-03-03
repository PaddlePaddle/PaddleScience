# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.optimizer.lr import LRScheduler
import math


class CosineAnnealingWarmRestarts(LRScheduler):
    def __init__(self,
                 learning_rate,
                 T_0,
                 T_mult=1,
                 eta_min=0,
                 last_epoch=-1,
                 verbose=False):

        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(
                "Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(
                T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super(CosineAnnealingWarmRestarts, self).__init__(learning_rate,
                                                          last_epoch, verbose)

    def get_lr(self):
        return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(
            math.pi * self.T_cur / self.T_i)) / 2

    def step(self, epoch=None):

        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError(
                    "Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(
                        math.log((epoch / self.T_0 * (self.T_mult - 1) + 1),
                                 self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (
                        self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult**(n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        self.last_lr = self.get_lr()
