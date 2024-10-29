r"""Engine for iterative optimization-based registration."""
from __future__ import annotations

import math
from collections import OrderedDict
from timeit import default_timer as timer
from typing import Callable

import paddle

from .losses import RegistrationLoss
from .losses import RegistrationResult
from .optim import slope_of_least_squares_fit

PROFILING = False


class RegistrationEngine(object):
    r"""Minimize registration loss until convergence."""

    def __init__(
        self,
        loss: RegistrationLoss,
        optimizer: paddle.optimizer.Optimizer,
        max_steps: int = 500,
        min_delta: float = 1e-06,
        min_value: float = float("nan"),
        max_history: int = 10,
    ):
        r"""Initialize registration loop."""
        self.loss = loss
        self.optimizer = optimizer
        self.num_steps = 0
        self.max_steps = max_steps
        self.min_delta = min_delta
        self.min_value = min_value
        self.max_history = max(2, max_history)
        self.loss_values = []
        self._eval_hooks = OrderedDict()
        self._step_hooks = OrderedDict()

    @property
    def loss_value(self) -> float:
        if not self.loss_values:
            return float("inf")
        return self.loss_values[-1]

    def step(self) -> float:
        r"""Perform one registration step.

        Returns:
            Loss value prior to taking gradient step.

        """
        num_evals = 0

        def closure() -> float:
            self.optimizer.clear_gradients(set_to_zero=False)
            t_start = timer()
            result = self.loss.eval()
            if PROFILING:
                print(f"Forward pass in {timer() - t_start:.3f}s")
            loss = result["loss"]
            assert isinstance(loss, paddle.Tensor)
            t_start = timer()
            loss.backward()
            if PROFILING:
                print(f"Backward pass in {timer() - t_start:.3f}s")
            nonlocal num_evals
            num_evals += 1
            with paddle.no_grad():
                for hook in self._eval_hooks.values():
                    hook(self, self.num_steps, num_evals, result)
            return float(loss)

        loss_value = self.optimizer.step(closure)
        assert loss_value is not None
        with paddle.no_grad():
            for hook in self._step_hooks.values():
                hook(self, self.num_steps, num_evals, loss_value)
        return loss_value

    def run(self) -> float:
        r"""Perform registration steps until convergence.

        Returns:
            Loss value prior to taking last gradient step.

        """
        self.loss_values = []
        self.num_steps = 0
        while self.num_steps < self.max_steps and not self.converged():
            value = self.step()
            self.num_steps += 1
            if math.isnan(value):
                raise RuntimeError(
                    f"NaN value in registration loss at gradient step {self.num_steps}"
                )
            if math.isinf(value):
                raise RuntimeError(
                    f"Inf value in registration loss at gradient step {self.num_steps}"
                )
            self.loss_values.append(value)
            if len(self.loss_values) > self.max_history:
                self.loss_values.pop(0)
        return self.loss_value

    def converged(self) -> bool:
        r"""Check convergence criteria."""
        values = self.loss_values
        if not values:
            return False
        value = values[-1]
        if self.min_delta < 0:
            epsilon = abs(self.min_delta * value)
        else:
            epsilon = self.min_delta
        slope = slope_of_least_squares_fit(values)
        if abs(slope) < epsilon:
            return True
        if value < self.min_value:
            return True
        return False

    def register_eval_hook(
        self, hook: Callable[[RegistrationEngine, int, int, RegistrationResult], None]
    ):
        r"""Registers a evaluation hook.

        The hook will be called every time after the registration loss has been evaluated
        during a single step of the optimizer, and the backward pass was performed, but
        before the model parameters are updated by taking a gradient step.

            hook(self, num_steps: int, num_evals: int, result: RegistrationResult) -> None

        Returns:
            A handle that can be used to remove the added hook by calling ``handle.remove()``

        """
        raise NotImplementedError("register_eval_hook")
        import paddle

        handle = paddle.utils.hooks.RemovableHandle(self._eval_hooks)
        self._eval_hooks[handle.id] = hook
        return handle

    def register_step_hook(self, hook: Callable[[RegistrationEngine, int, int, float], None]):
        r"""Registers a gradient step hook.

        The hook will be called every time after a gradient step of the optimizer.

            hook(self, num_steps: int, num_evals: int, loss: float) -> None

        Returns:
            A handle that can be used to remove the added hook by calling ``handle.remove()``

        """
        raise NotImplementedError("register_step_hook")
        import paddle

        handle = paddle.utils.hooks.RemovableHandle(self._step_hooks)
        self._step_hooks[handle.id] = hook
        return handle
