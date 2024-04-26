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

from ppsci import arch  # isort:skip
from ppsci import autodiff  # isort:skip
from ppsci import constraint  # isort:skip
from ppsci import data  # isort:skip
from ppsci import equation  # isort:skip
from ppsci import geometry  # isort:skip
from ppsci import loss  # isort:skip
from ppsci import metric  # isort:skip
from ppsci import optimizer  # isort:skip
from ppsci import utils  # isort:skip
from ppsci import visualize  # isort:skip
from ppsci import validate  # isort:skip
from ppsci import solver  # isort:skip
from ppsci import experimental  # isort:skip

from ppsci.utils.checker import run_check  # isort:skip
from ppsci.utils.checker import run_check_mesh  # isort:skip
from ppsci.utils import lambdify  # isort:skip


try:
    # import auto-generated version information from '._version' file, using
    # setuptools_scm via 'pip install'. Details of versioning rule can be referd to:
    # https://peps.python.org/pep-0440/#public-version-identifiers
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown version"

__all__ = [
    "arch",
    "autodiff",
    "constraint",
    "data",
    "equation",
    "geometry",
    "loss",
    "metric",
    "optimizer",
    "utils",
    "visualize",
    "validate",
    "solver",
    "experimental",
    "run_check",
    "run_check_mesh",
    "lambdify",
]


# NOTE: Register custom solvers for parsing values from omegaconf more flexible
def _register_config_solvers():
    import numpy as np
    from omegaconf import OmegaConf

    # register solver for "${numpy: xxx}" item, e.g. pi: "${numpy: pi}"
    if not OmegaConf.has_resolver("numpy"):
        OmegaConf.register_new_resolver("numpy", lambda x: getattr(np, x))


_register_config_solvers()
