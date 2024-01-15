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

"""
This module is for experimental API
"""

from ppsci.experimental.math_module import bessel_i0
from ppsci.experimental.math_module import bessel_i0e
from ppsci.experimental.math_module import bessel_i1
from ppsci.experimental.math_module import bessel_i1e
from ppsci.experimental.math_module import fractional_diff
from ppsci.experimental.math_module import gaussian_integrate
from ppsci.experimental.math_module import trapezoid_integrate

__all__ = [
    "bessel_i0",
    "bessel_i0e",
    "bessel_i1",
    "bessel_i1e",
    "fractional_diff",
    "gaussian_integrate",
    "trapezoid_integrate",
]
