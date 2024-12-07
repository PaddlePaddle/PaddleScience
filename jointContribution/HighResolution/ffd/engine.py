from __future__ import annotations

import math
import weakref
from collections import OrderedDict
from timeit import default_timer as timer
from typing import Any
from typing import Callable
from typing import Tuple

import paddle

from .losses import RegistrationLoss
from .losses import RegistrationResult
from .optim import slope_of_least_squares_fit

PROFILING = False


class RegistrationEngine:
    """Minimize registration loss until convergence."""

    def __init__(
        self,
        model: paddle.nn.Layer,
        loss: RegistrationLoss,
        optimizer: paddle.optimizer.Optimizer,
        max_steps: int = 500,
        min_delta: float = 1e-06,
        min_value: float = float("nan"),
        max_history: int = 10,
    ):
        """Initialize registration loop."""
        self.model = model
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
        """Perform one registration step.

        Returns:
            Loss value prior to taking gradient step.

        """
        num_evals = 0

        def closure() -> float:
            self.optimizer.clear_grad()
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

        loss_value = closure()
        self.optimizer.step()
        assert loss_value is not None
        with paddle.no_grad():
            for hook in self._step_hooks.values():
                hook(self, self.num_steps, num_evals, loss_value)
        return loss_value

    def run(self) -> float:
        """Perform registration steps until convergence.

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
        """Check convergence criteria."""
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
        self,
        hook: Callable[["RegistrationEngine", int, int, "RegistrationResult"], None],
    ) -> "RemovableHandle":
        r"""Registers an evaluation hook."""
        handle = RemovableHandle(self._eval_hooks)
        self._eval_hooks[handle.id] = hook
        return handle

    def register_step_hook(
        self, hook: Callable[["RegistrationEngine", int, int, float], None]
    ) -> "RemovableHandle":
        r"""Registers a gradient step hook."""
        handle = RemovableHandle(self._step_hooks)
        self._step_hooks[handle.id] = hook
        return handle


class RemovableHandle:
    r"""
    A handle which provides the capability to remove a hook.

    Args:
        hooks_dict (dict): A dictionary of hooks, indexed by hook ``id``.
        extra_dict (Union[dict, List[dict]]): An additional dictionary or list of
            dictionaries whose keys will be deleted when the same keys are
            removed from ``hooks_dict``.
    """

    id: int
    next_id: int = 0

    def __init__(self, hooks_dict: Any, *, extra_dict: Any = None) -> None:
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

        self.extra_dict_ref: Tuple = ()
        if isinstance(extra_dict, dict):
            self.extra_dict_ref = (weakref.ref(extra_dict),)
        elif isinstance(extra_dict, list):
            self.extra_dict_ref = tuple(weakref.ref(d) for d in extra_dict)

    def remove(self) -> None:
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

        for ref in self.extra_dict_ref:
            extra_dict = ref()
            if extra_dict is not None and self.id in extra_dict:
                del extra_dict[self.id]

    def __getstate__(self):
        if self.extra_dict_ref is None:
            return (self.hooks_dict_ref(), self.id)
        else:
            return (
                self.hooks_dict_ref(),
                self.id,
                tuple(ref() for ref in self.extra_dict_ref),
            )

    def __setstate__(self, state) -> None:
        if state[0] is None:
            # create a dead reference
            self.hooks_dict_ref = weakref.ref(OrderedDict())
        else:
            self.hooks_dict_ref = weakref.ref(state[0])
        self.id = state[1]
        RemovableHandle.next_id = max(RemovableHandle.next_id, self.id + 1)

        if len(state) < 3 or state[2] is None:
            self.extra_dict_ref = ()
        else:
            self.extra_dict_ref = tuple(weakref.ref(d) for d in state[2])

    def __enter__(self) -> "RemovableHandle":
        return self

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        self.remove()
