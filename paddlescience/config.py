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

from paddle.incubate.autograd import enable_prim, disable_prim

_dtype = 'float32'
_ad_api_mode = 'functional'


def set_ad_api_mode(mode):
    assert mode in ['functional', 'procedural'
                    ], "Invalid AD API mode, must be functional or procedural."
    global _ad_api_mode
    _ad_api_mode = mode


def get_ad_api_mode():
    global _ad_api_mode
    return _ad_api_mode


def set_dtype(dtype):
    global _dtype
    _dtype = dtype


def get_dtype():
    global _dtype
    return _dtype
