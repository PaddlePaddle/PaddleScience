from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional

from paddle import ParamAttr
from paddle.optimizer import Adam, SGD, Optimizer
from paddle.optimizer.lr import CosineAnnealingDecay, MultiStepDecay, StepDecay

import paddle_geometric.graphgym.register as register
from paddle_geometric.graphgym.config import from_config


@dataclass
class OptimizerConfig:
    optimizer: str = 'adam'  # ['sgd', 'adam']
    base_lr: float = 0.01
    weight_decay: float = 5e-4
    momentum: float = 0.9  # 'sgd' policy


@register.register_optimizer('adam')
def adam_optimizer(params: Iterator[ParamAttr], base_lr: float,
                   weight_decay: float) -> Adam:
    return Adam(parameters=params, learning_rate=base_lr, weight_decay=weight_decay)


@register.register_optimizer('sgd')
def sgd_optimizer(params: Iterator[ParamAttr], base_lr: float, momentum: float,
                  weight_decay: float) -> SGD:
    return SGD(parameters=params, learning_rate=base_lr, momentum=momentum,
               weight_decay=weight_decay)


def create_optimizer(params: Iterator[ParamAttr], cfg: Any) -> Any:
    """Creates a config-driven optimizer."""
    params = filter(lambda p: p.stop_gradient is False, params)
    func = register.optimizer_dict.get(cfg.optimizer, None)
    if func is not None:
        return from_config(func)(params, cfg=cfg)
    raise ValueError(f"Optimizer '{cfg.optimizer}' not supported")


@dataclass
class SchedulerConfig:
    scheduler: Optional[str] = 'cos'  # [None, 'steps', 'cos']
    steps: List[int] = field(default_factory=lambda: [30, 60, 90])  # 'steps' policy
    lr_decay: float = 0.1  # 'steps' policy
    max_epoch: int = 200


@register.register_scheduler(None)
@register.register_scheduler('none')
def none_scheduler(optimizer: Optimizer, max_epoch: int) -> StepDecay:
    return StepDecay(learning_rate=optimizer.get_lr(), step_size=max_epoch + 1)


@register.register_scheduler('step')
def step_scheduler(optimizer: Optimizer, steps: List[int],
                   lr_decay: float) -> MultiStepDecay:
    return MultiStepDecay(learning_rate=optimizer.get_lr(), milestones=steps, gamma=lr_decay)


@register.register_scheduler('cos')
def cos_scheduler(optimizer: Optimizer, max_epoch: int) -> CosineAnnealingDecay:
    return CosineAnnealingDecay(learning_rate=optimizer.get_lr(), T_max=max_epoch)


def create_scheduler(optimizer: Optimizer, cfg: Any) -> Any:
    """Creates a config-driven learning rate scheduler."""
    func = register.scheduler_dict.get(cfg.scheduler, None)
    if func is not None:
        return from_config(func)(optimizer, cfg=cfg)
    raise ValueError(f"Scheduler '{cfg.scheduler}' not supported")
