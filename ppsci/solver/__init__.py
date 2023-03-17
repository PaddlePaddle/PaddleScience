"""Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from ppsci.solver import eval
from ppsci.solver import train
from ppsci.solver.eval import eval_func
from ppsci.solver.solver import Solver
from ppsci.solver.train import train_epoch_func
from ppsci.solver.train import train_LBFGS_epoch_func

__all__ = [
    "eval",
    "train",
    "eval_func",
    "Solver",
    "train_epoch_func",
    "train_LBFGS_epoch_func",
]
