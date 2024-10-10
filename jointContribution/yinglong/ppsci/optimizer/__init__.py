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

import copy

from ppsci.optimizer import lr_scheduler
from ppsci.optimizer.optimizer import LBFGS
from ppsci.optimizer.optimizer import SGD
from ppsci.optimizer.optimizer import Adam
from ppsci.optimizer.optimizer import AdamW
from ppsci.optimizer.optimizer import Momentum
from ppsci.optimizer.optimizer import OptimizerList
from ppsci.optimizer.optimizer import RMSProp

__all__ = [
    "LBFGS",
    "SGD",
    "Adam",
    "AdamW",
    "Momentum",
    "RMSProp",
    "OptimizerList",
    "lr_scheduler",
]


def build_lr_scheduler(cfg, epochs, iters_per_epoch):
    """Build learning rate scheduler.

    Args:
        cfg (AttrDict): Learing rate scheduler config.
        epochs (int): Total epochs.
        iters_per_epoch (int): Number of iterations of one epoch.

    Returns:
        LRScheduler: Learing rate scheduler.
    """
    cfg = copy.deepcopy(cfg)
    cfg.update({"epochs": epochs, "iters_per_epoch": iters_per_epoch})
    lr_scheduler_cls = cfg.pop("name")
    lr_scheduler_ = eval(lr_scheduler_cls)(**cfg)
    return lr_scheduler_()


def build_optimizer(cfg, model_list, epochs, iters_per_epoch):
    """Build optimizer and learing rate scheduler

    Args:
        cfg (AttrDict): Learing rate scheduler config.
        model_list (Tuple[nn.Layer, ...]): Tuple of model(s).
        epochs (int): Total epochs.
        iters_per_epoch (int): Number of iterations of one epoch.

    Returns:
        Optimizer, LRScheduler: Optimizer and learing rate scheduler.
    """
    # build lr_scheduler
    cfg = copy.deepcopy(cfg)
    lr_cfg = cfg.pop("lr")
    if isinstance(lr_cfg, float):
        lr_scheduler = lr_cfg
    else:
        lr_scheduler = build_lr_scheduler(lr_cfg, epochs, iters_per_epoch)

    # build optimizer
    opt_cls = cfg.pop("name")
    optimizer = eval(opt_cls)(learning_rate=lr_scheduler, **cfg)(model_list)

    if isinstance(lr_scheduler, float):
        return optimizer, None
    return optimizer, lr_scheduler
