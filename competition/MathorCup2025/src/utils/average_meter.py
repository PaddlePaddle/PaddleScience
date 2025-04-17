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


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Average Meter with dictionary values
class AverageMeterDict:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = {}
        self.avg = {}
        self.sum = {}
        self.count = {}

    def update(self, val, n=1):
        for k, v in val.items():
            if k not in self.val:
                self.val[k] = 0
                self.sum[k] = 0
                self.count[k] = 0
            self.val[k] = v
            self.sum[k] += v * n
            self.count[k] += n
            self.avg[k] = self.sum[k] / self.count[k]
