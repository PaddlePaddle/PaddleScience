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


class LabelIndex(int):
    def __new__(cls, value, *args, **kwargs):
        return super(cls, cls).__new__(cls, value)

    # def __init__(self, rhs):
    #     self.rhs = None
    #     self.weight = None
    #     self.u_n = None
    #     self.parameter = None

    #     self.idx_rhs = 0
    #     self.idx_weight = 0
    #     self.idx_u_n = 0
    #     self.idx_parameter = 0
