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
    """
    Use static graph mode.
    """
    paddle.enable_static()


def enable_prim():
    '''
    Enable the automatic differentiation.
    '''
    paddle.incubate.autograd.enable_prim()


def disable_prim():
    '''
    Disable the automatic differentiation.
    '''
    paddle.incubate.autograd.disable_prim()


def prim_enabled():
    '''
    Determine whether automatic differentiation is enabled.
    '''
    return paddle.incubate.autograd.prim_enabled()


def prim2orig(*args):
    '''
    All operators in the target block are processed as follows.
    If it is an automatic differential basic operator, it will be
    transformed into one or a series of original operators with
    equivalent function to support execution.
    '''
    return paddle.incubate.autograd.prim2orig(*args)
