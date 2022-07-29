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
import os

_dtype = 'float32'
_use_visualdl = False
_compute_backend = "paddle"


def set_dtype(dtype):
    global _dtype
    _dtype = dtype


def get_dtype():
    global _dtype
    return _dtype


def set_compute_backend(compute_backend):
    global _compute_backend
    _compute_backend = compute_backend


def enable_visualdl():
    global _use_visualdl
    _use_visualdl = True


def visualdl_enabled():
    return _use_visualdl


def enable_static():
    """
    Use static graph mode.
    """
    paddle.enable_static()


def enable_prim():
    '''
    Enable automatic differentiation mechanism based on 
    automatic differentiation basic operator.
    '''
    if paddle.in_dynamic_mode():
        pass  # TODO: error out
    else:
        paddle.incubate.autograd.enable_prim()


def disable_prim():
    '''
    Disable automatic differentiation mechanism based on 
    automatic differentiation basic operator.
    '''
    if paddle.in_dynamic_mode():
        pass
    else:
        paddle.incubate.autograd.disable_prim()


def prim_enabled():
    '''
    Determine whether automatic differentiation based on 
    automatic differentiation basic operator is enabled.
    '''
    if paddle.in_dynamic_mode():
        return False
    else:
        return paddle.incubate.autograd.prim_enabled()


def prim2orig(*args):
    '''
    All operators in the target block are processed as follows.
    If it is an automatic differential basic operator, it will be
    transformed into one or a series of original operators with
    equivalent function to support execution.
    '''
    if paddle.in_dynamic_mode():
        pass
    else:
        return paddle.incubate.autograd.prim2orig(*args)


def cinn_enabled():
    '''
    Determine whether CINN is enabled. Ref https://github.com/PaddlePaddle/CINN
    '''
    cinn_flag = os.getenv('FLAGS_use_cinn', '0')
    check_cinn_set = cinn_flag == "1" or cinn_flag.lower() == "true"
    return check_cinn_set


def enable_cinn():
    '''
    Enable CINN based on static graph mode and automatic differentiation basic operator.
    Please ensure FLAGS_use_cinn is set to 1 or true. Ref https://github.com/PaddlePaddle/CINN
    '''
    if cinn_enabled():
        enable_static()
        enable_prim()
