# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from ppsci.utils import logger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

def register_cls_to_module(module: ModuleType, cls: type):
    cls_name = cls.__name__

    if hasattr(module, cls_name):
        logger.warning(f"Class '{cls_name}' already exists. Overriding...")

    setattr(module, cls_name, cls)

    if hasattr(module, "__all__") and cls_name not in module.__all__:
        module.__all__.append(cls_name)

    logger.debug(f"Registered class {cls_name} to module {module}")
    return cls
