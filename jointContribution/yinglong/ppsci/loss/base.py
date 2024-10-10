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

from typing import Dict
from typing import Optional
from typing import Union

from paddle import nn
from typing_extensions import Literal


class Loss(nn.Layer):
    """Base class for loss."""

    def __init__(
        self,
        reduction: Literal["mean", "sum"],
        weight: Optional[Union[float, Dict[str, float]]] = None,
    ):
        super().__init__()
        self.reduction = reduction
        self.weight = weight

    def __str__(self):
        return f"{self.__class__.__name__}(reduction={self.reduction}, weight={self.weight})"
