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

import importlib.util
from typing import Tuple
from typing import Union

from ppsci.utils import logger


def check_module(module_name: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if module can be imported.

    Args:
        module_name (Union[str, Tuple[str, ...]]): Module name of list of module names.

    Returns:
        bool: Whether given module_name all exist.
    """
    if isinstance(module_name, str):
        module_spec = importlib.util.find_spec(module_name)
        if not module_spec:
            logger.error(f"Module {module_name} should be installed first.")
    else:
        for _module_name in module_name:
            module_spec = importlib.util.find_spec(_module_name)
            if not module_spec:
                logger.error(f"Module {_module_name} should be installed first.")
                return False
    return True
