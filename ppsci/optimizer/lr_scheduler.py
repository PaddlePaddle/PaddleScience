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

import abc
import math
from typing import Tuple
from typing import Union

from paddle.optimizer import lr

from ppsci.utils import logger

__all__ = [
    "Linear",
    "Cosine",
    "Step",
    "Piecewise",
    "MultiStepDecay",
    "ExponentialDecay",
    "CosineWarmRestarts",
]


class LRBase:
    """Base class for custom learning rates.

    Args:
        epochs (int): total epoch(s).
        iters_per_epoch (int): number of iterations within an epoch.
        learning_rate (float): learning rate.
        warmup_epoch (int): number of warmup epochs.
        warmup_start_lr (float): start learning rate within warmup.
        last_epoch (int): last epoch.
        by_epoch (bool): learning rate decays by epoch when by_epoch is True, else by iter.
        verbose (bool): If True, prints a message to stdout for each update. Defaults to False.
    """

    def __init__(
        self,
        epochs: int,
        iters_per_epoch: int,
        learning_rate: float,
        warmup_epoch: int,
        warmup_start_lr: float,
        last_epoch: int,
        by_epoch: bool,
        verbose: bool = False,
    ) -> None:
        """Initialize and record the necessary parameters"""
        super().__init__()
        if warmup_epoch >= epochs:
            msg = (
                "When using warm up, the value of 'Global.epochs' should be greater "
                "than value of 'Optimizer.lr.warmup_epoch'. The value of "
                f"'Optimizer.lr.warmup_epoch' has been set to {epochs}."
            )
            logger.warning(msg)
            warmup_epoch = epochs
        self.epochs = epochs
        self.iters_per_epoch = iters_per_epoch
        self.learning_rate = learning_rate
        self.warmup_epoch = warmup_epoch
        self.warmup_steps = (
            self.warmup_epoch
            if by_epoch
            else round(self.warmup_epoch * self.iters_per_epoch)
        )
        self.warmup_start_lr = warmup_start_lr
        self.last_epoch = last_epoch
        self.by_epoch = by_epoch
        self.verbose = verbose

    @abc.abstractmethod
    def __call__(self, *kargs, **kwargs) -> lr.LRScheduler:
        """Generate an learning rate scheduler.

        Returns:
            lr.LinearWarmup: learning rate scheduler.
        """
        pass

    def linear_warmup(
        self, learning_rate: Union[float, lr.LRScheduler]
    ) -> lr.LinearWarmup:
        """Add an Linear Warmup before learning_rate.

        Args:
            learning_rate (Union[float, lr.LRScheduler]): original learning rate without
                warmup.

        Returns:
            lr.LinearWarmup: learning rate scheduler with warmup.
        """
        warmup_lr = lr.LinearWarmup(
            learning_rate=learning_rate,
            warmup_steps=self.warmup_steps,
            start_lr=self.warmup_start_lr,
            end_lr=self.learning_rate,
            last_epoch=self.last_epoch,
            verbose=self.verbose,
        )
        return warmup_lr


class Constant(lr.LRScheduler):
    """Constant learning rate Class implementation.

    Args:
        learning_rate (float): The initial learning rate.
        last_epoch (int, optional): The index of last epoch. Default: -1.
    """

    def __init__(self, learning_rate: float, last_epoch: int = -1):
        self.learning_rate = learning_rate
        self.last_epoch = last_epoch
        super().__init__()

    def get_lr(self) -> float:
        """Always return the same learning rate"""
        return self.learning_rate


class Linear(LRBase):
    """Linear learning rate decay.

    Args:
        epochs (int): total epoch(s).
        iters_per_epoch (int): number of iterations within an epoch.
        learning_rate (float): learning rate.
        end_lr (float, optional): The minimum final learning rate. Defaults to 0.0.
        power (float, optional): Power of polynomial. Defaults to 1.0.
        warmup_epoch (int): number of warmup epochs.
        warmup_start_lr (float): start learning rate within warmup.
        last_epoch (int): last epoch.
        by_epoch (bool): learning rate decays by epoch when by_epoch is True, else by iter.

    Examples:
        >>> import ppsci
        >>> lr = ppsci.optimizer.lr_scheduler.Linear(10, 2, 0.001)
    """

    def __init__(
        self,
        epochs: int,
        iters_per_epoch: int,
        learning_rate: float,
        end_lr: float = 0.0,
        power: float = 1.0,
        cycle: bool = False,
        warmup_epoch: int = 0,
        warmup_start_lr: float = 0.0,
        last_epoch: int = -1,
        by_epoch: bool = False,
    ):
        super().__init__(
            epochs,
            iters_per_epoch,
            learning_rate,
            warmup_epoch,
            warmup_start_lr,
            last_epoch,
            by_epoch,
        )
        self.decay_steps = (epochs - self.warmup_epoch) * iters_per_epoch
        self.end_lr = end_lr
        self.power = power
        self.cycle = cycle
        self.warmup_steps = round(self.warmup_epoch * iters_per_epoch)
        if self.by_epoch:
            self.decay_steps = self.epochs - self.warmup_epoch

    def __call__(self):
        learning_rate = (
            lr.PolynomialDecay(
                learning_rate=self.learning_rate,
                decay_steps=self.decay_steps,
                end_lr=self.end_lr,
                power=self.power,
                cycle=self.cycle,
                last_epoch=self.last_epoch,
            )
            if self.decay_steps > 0
            else Constant(self.learning_rate)
        )

        if self.warmup_steps > 0:
            learning_rate = self.linear_warmup(learning_rate)

        setattr(learning_rate, "by_epoch", self.by_epoch)
        return learning_rate


class ExponentialDecay(LRBase):
    """ExponentialDecay learning rate decay.

    Args:
        epochs (int): total epoch(s).
        iters_per_epoch (int): number of iterations within an epoch.
        learning_rate (float): learning rate.
        warmup_epoch (int): number of warmup epochs.
        warmup_start_lr (float): start learning rate within warmup.
        last_epoch (int): last epoch.
        by_epoch (bool): learning rate decays by epoch when by_epoch is True, else by iter.

    Examples:
        >>> import ppsci
        >>> lr = ppsci.optimizer.lr_scheduler.ExponentialDecay(10, 2, 1e-3, 0.95, 3)
    """

    def __init__(
        self,
        epochs: int,
        iters_per_epoch: int,
        learning_rate: float,
        gamma: float,
        decay_steps: int,
        warmup_epoch: int = 0,
        warmup_start_lr: float = 0.0,
        last_epoch: int = -1,
        by_epoch: bool = False,
    ):
        super().__init__(
            epochs,
            iters_per_epoch,
            learning_rate,
            warmup_epoch,
            warmup_start_lr,
            last_epoch,
            by_epoch,
        )
        self.decay_steps = decay_steps
        self.gamma = gamma
        self.warmup_steps = round(self.warmup_epoch * iters_per_epoch)
        if self.by_epoch:
            self.decay_steps /= iters_per_epoch

    def __call__(self):
        learning_rate = lr.ExponentialDecay(
            learning_rate=self.learning_rate,
            gamma=self.gamma ** (1 / self.decay_steps),
            last_epoch=self.last_epoch,
        )

        if self.warmup_steps > 0:
            learning_rate = self.linear_warmup(learning_rate)

        setattr(learning_rate, "by_epoch", self.by_epoch)
        return learning_rate


class Cosine(LRBase):
    r"""Cosine learning rate decay.

    lr = 0.05 * (math.cos(epoch * (math.pi / epochs)) + 1)

    Args:
        epochs (int): total epoch(s).
        iters_per_epoch (int): number of iterations within an epoch.
        learning_rate (float): learning rate.
        eta_min (float, optional): Minimum learning rate. Defaults to 0.0.
        warmup_epoch (int, optional): The epoch numbers for LinearWarmup. Defaults to 0.
        warmup_start_lr (float, optional): start learning rate within warmup. Defaults to 0.0.
        last_epoch (int, optional): last epoch. Defaults to -1.
        by_epoch (bool, optional): learning rate decays by epoch when by_epoch is True,
            else by iter. Defaults to False.

    Examples:
        >>> import ppsci
        >>> lr = ppsci.optimizer.lr_scheduler.Cosine(10, 2, 1e-3)
    """

    def __init__(
        self,
        epochs: int,
        iters_per_epoch: int,
        learning_rate: float,
        eta_min: float = 0.0,
        warmup_epoch: int = 0,
        warmup_start_lr: float = 0.0,
        last_epoch: int = -1,
        by_epoch: bool = False,
    ):
        super().__init__(
            epochs,
            iters_per_epoch,
            learning_rate,
            warmup_epoch,
            warmup_start_lr,
            last_epoch,
            by_epoch,
        )
        self.T_max = (self.epochs - self.warmup_epoch) * self.iters_per_epoch
        self.eta_min = eta_min
        if self.by_epoch:
            self.T_max = self.epochs - self.warmup_epoch

    def __call__(self):
        learning_rate = (
            lr.CosineAnnealingDecay(
                learning_rate=self.learning_rate,
                T_max=self.T_max,
                eta_min=self.eta_min,
                last_epoch=self.last_epoch,
            )
            if self.T_max > 0
            else Constant(self.learning_rate)
        )

        if self.warmup_steps > 0:
            learning_rate = self.linear_warmup(learning_rate)

        setattr(learning_rate, "by_epoch", self.by_epoch)
        return learning_rate


class Step(LRBase):
    """Step learning rate decay.

    Args:
        epochs (int): total epoch(s).
        iters_per_epoch (int): number of iterations within an epoch.
        learning_rate (float): learning rate.
        step_size (int): the interval to update.
        gamma (float, optional): The Ratio that the learning rate will be reduced.
            ``new_lr = origin_lr * gamma``. It should be less than 1.0. Default: 0.1.
        warmup_epoch (int, optional): The epoch numbers for LinearWarmup. Defaults to 0.
        warmup_start_lr (float, optional): start learning rate within warmup. Defaults to 0.0.
        last_epoch (int, optional): last epoch. Defaults to -1.
        by_epoch (bool, optional): learning rate decays by epoch when by_epoch is True,
            else by iter. Defaults to False.

    Examples:
        >>> import ppsci
        >>> lr = ppsci.optimizer.lr_scheduler.Step(10, 1, 1e-3, 2, 0.95)
    """

    def __init__(
        self,
        epochs: int,
        iters_per_epoch: int,
        learning_rate: float,
        step_size: int,
        gamma: float,
        warmup_epoch: int = 0,
        warmup_start_lr: float = 0.0,
        last_epoch: int = -1,
        by_epoch: bool = False,
    ):
        super().__init__(
            epochs,
            iters_per_epoch,
            learning_rate,
            warmup_epoch,
            warmup_start_lr,
            last_epoch,
            by_epoch,
        )
        self.step_size = step_size * iters_per_epoch
        self.gamma = gamma
        if self.by_epoch:
            self.step_size = step_size

    def __call__(self):
        learning_rate = lr.StepDecay(
            learning_rate=self.learning_rate,
            step_size=self.step_size,
            gamma=self.gamma,
            last_epoch=self.last_epoch,
        )

        if self.warmup_steps > 0:
            learning_rate = self.linear_warmup(learning_rate)

        setattr(learning_rate, "by_epoch", self.by_epoch)
        return learning_rate


class Piecewise(LRBase):
    """Piecewise learning rate decay

    Args:
        epochs (int): total epoch(s)
        iters_per_epoch (int): number of iterations within an epoch
        decay_epochs (Tuple[int, ...]): A list of steps numbers. The type of element in the
            list is python int.
        values (Tuple[float, ...]): Tuple of learning rate values that will be picked during
            different epoch boundaries.
        warmup_epoch (int, optional): The epoch numbers for LinearWarmup. Defaults to 0.
        warmup_start_lr (float, optional): start learning rate within warmup. Defaults to 0.0.
        last_epoch (int, optional): last epoch. Defaults to -1.
        by_epoch (bool, optional): learning rate decays by epoch when by_epoch is True,
            else by iter. Defaults to False.

    Examples:
        >>> import ppsci
        >>> lr = ppsci.optimizer.lr_scheduler.Piecewise(10, 1, [2, 4], (1e-3, 1e-4))
    """

    def __init__(
        self,
        epochs: int,
        iters_per_epoch: int,
        decay_epochs: Tuple[int, ...],
        values: Tuple[float, ...],
        warmup_epoch: int = 0,
        warmup_start_lr: float = 0.0,
        last_epoch: int = -1,
        by_epoch: bool = False,
    ):
        super().__init__(
            epochs,
            iters_per_epoch,
            values[0],
            warmup_epoch,
            warmup_start_lr,
            last_epoch,
            by_epoch,
        )
        self.values = values
        self.boundaries_steps = [e * iters_per_epoch for e in decay_epochs]
        if self.by_epoch is True:
            self.boundaries_steps = decay_epochs

    def __call__(self):
        learning_rate = lr.PiecewiseDecay(
            boundaries=self.boundaries_steps,
            values=self.values,
            last_epoch=self.last_epoch,
        )

        if self.warmup_steps > 0:
            learning_rate = self.linear_warmup(learning_rate)

        setattr(learning_rate, "by_epoch", self.by_epoch)
        return learning_rate


class MultiStepDecay(LRBase):
    """MultiStepDecay learning rate decay

    Args:
        epochs (int): total epoch(s)
        iters_per_epoch (int): number of iterations within an epoch
        learning_rate (float): learning rate
        milestones (Tuple[int, ...]): Tuple of each boundaries. should be increasing.
        gamma (float, optional): The Ratio that the learning rate will be reduced.
            `new_lr = origin_lr * gamma`. It should be less than 1.0. Defaults to 0.1.
        warmup_epoch (int, optional): The epoch numbers for LinearWarmup. Defaults to 0.
        warmup_start_lr (float, optional): start learning rate within warmup. Defaults to 0.0.
        last_epoch (int, optional): last epoch. Defaults to -1.
        by_epoch (bool, optional): learning rate decays by epoch when by_epoch is True,
            else by iter. Defaults to False.

    Examples:
        >>> import ppsci
        >>> lr = ppsci.optimizer.lr_scheduler.MultiStepDecay(10, 1, 1e-3, (4, 5))
    """

    def __init__(
        self,
        epochs: int,
        iters_per_epoch: int,
        learning_rate: float,
        milestones: Tuple[int, ...],
        gamma: float = 0.1,
        warmup_epoch: int = 0,
        warmup_start_lr: float = 0.0,
        last_epoch: int = -1,
        by_epoch: bool = False,
    ):
        super().__init__(
            epochs,
            iters_per_epoch,
            learning_rate,
            warmup_epoch,
            warmup_start_lr,
            last_epoch,
            by_epoch,
        )
        self.milestones = [x * iters_per_epoch for x in milestones]
        self.gamma = gamma
        if self.by_epoch:
            self.milestones = milestones

    def __call__(self):
        learning_rate = lr.MultiStepDecay(
            learning_rate=self.learning_rate,
            milestones=self.milestones,
            gamma=self.gamma,
            last_epoch=self.last_epoch,
        )

        if self.warmup_steps > 0:
            learning_rate = self.linear_warmup(learning_rate)

        setattr(learning_rate, "by_epoch", self.by_epoch)
        return learning_rate


class CosineAnnealingWarmRestarts(lr.LRScheduler):
    """The implementation of cosine annealing schedule with warm restarts.

    Args:
        learning_rate (float): Learning rate
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases T_i after a restart. Defaults to 1.
        eta_min (float, optional): Minimum learning rate. Defaults to 0.
        last_epoch (int, optional): The index of last epoch. Defaults to -1.
        verbose (bool, optional): If `True`, prints a message to stdout for each update. Defaults to False.
    """

    def __init__(
        self,
        learning_rate: float,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super().__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        return (
            self.eta_min
            + (self.base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
        )

    def step(self, epoch=None):
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)
        self.last_lr = self.get_lr()


class CosineWarmRestarts(LRBase):
    """Set the learning rate using a cosine annealing schedule with warm restarts.

    Args:
        epochs (int): Total epoch(s)
        iters_per_epoch (int): Number of iterations within an epoch
        learning_rate (float): Learning rate
        T_0 (int): Number of iterations for the first restart.
        T_mult (int): A factor increases T_i after a restart
        eta_min (float, optional): Minimum learning rate. Defaults to 0.0.
        warmup_epoch (int, optional): The epoch numbers for LinearWarmup. Defaults to 0.
        warmup_start_lr (float, optional): Start learning rate within warmup. Defaults to 0.0.
        last_epoch (int, optional): Last epoch. Defaults to -1.
        by_epoch (bool, optional): Learning rate decays by epoch when by_epoch is True, else by iter. Defaults to False.

    Examples:
        >>> import ppsci
        >>> lr = ppsci.optimizer.lr_scheduler.CosineWarmRestarts(20, 1, 1e-3, 14, 2)
    """

    def __init__(
        self,
        epochs: int,
        iters_per_epoch: int,
        learning_rate: float,
        T_0: int,
        T_mult: int,
        eta_min: float = 0.0,
        warmup_epoch: int = 0,
        warmup_start_lr: float = 0.0,
        last_epoch: int = -1,
        by_epoch: bool = False,
    ):
        super().__init__(
            epochs,
            iters_per_epoch,
            learning_rate,
            warmup_epoch,
            warmup_start_lr,
            last_epoch,
            by_epoch,
        )
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        if self.by_epoch is False:
            self.T_0 = T_0 * iters_per_epoch

    def __call__(self):
        learning_rate = CosineAnnealingWarmRestarts(
            learning_rate=self.learning_rate,
            T_0=self.T_0,
            T_mult=self.T_mult,
            eta_min=self.eta_min,
            last_epoch=self.last_epoch,
            verbose=self.verbose,
        )

        if self.warmup_steps > 0:
            learning_rate = self.linear_warmup(learning_rate)

        setattr(learning_rate, "by_epoch", self.by_epoch)
        return learning_rate
