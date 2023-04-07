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
from typing import Dict
from typing import Tuple
from typing import Union

from ppsci.utils import logger


def dynamic_import_to_globals(
    names: Union[str, Tuple[str, ...]], alias: Dict[str, str] = None
) -> bool:
    """Import module and add it to globals() by given names dynamically.

    Args:
        names (Union[str, Tuple[str, ...]]): Module name or list of module names.
        alias (Dict[str, str]): Alias name of module when imported into globals().

    Returns:
        bool: Whether given names all exist.
    """
    if isinstance(names, str):
        names = (names,)

    if alias is None:
        alias = {}

    for name in names:
        # find module in environment by it's name and alias(if given)
        module_spec = importlib.util.find_spec(name)
        if module_spec is None and name in alias:
            module_spec = importlib.util.find_spec(alias[name])

        # log error and return False if module do not exist
        if not module_spec:
            logger.error(f"Module {name} should be installed first.")
            return False

        # module exist, add to globals() if not in globals()
        add_name = name
        if add_name in alias:
            add_name = alias[add_name]
        if add_name not in globals():
            globals()[add_name] = importlib.import_module(name)

    return True
