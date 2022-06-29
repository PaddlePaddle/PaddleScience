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

import paddle

_dtype = 'float32'


def set_dtype(dtype):
    global _dtype
    _dtype = dtype


def get_dtype():
    global _dtype
    return _dtype


def enable_static():
    paddle.enable_static()


def enable_prim():
    paddle.incubate.autograd.enable_prim()


def prim_enabled():
    return paddle.incubate.autograd.prim_enabled()


def disable_prim():
    paddle.incubate.autograd.disable_prim()


def prim2orig(*args):
    return paddle.incubate.autograd.prim2orig(*args)
