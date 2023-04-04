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

from typing import Optional
from typing import Tuple
from typing import Union

from paddle import nn
from paddle import optimizer as optim
from paddle import regularizer
from paddle.incubate import optimizer as incubate_optim
from typing_extensions import Literal

from ppsci.utils import logger

__all__ = ["SGD", "Momentum", "Adam", "RMSProp", "AdamW", "LBFGS"]


class SGD(object):
    """Stochastic Gradient Descent.

    Args:
        learning_rate (Union[float, optim.lr.LRScheduler], optional): The learning rate
            used to update parameter(s). Defaults to 0.001.
        weight_decay (Optional[Union[float, regularizer.L1Decay, regularizer.L2Decay]], optional):
            Regularization strategy. Defaults to None.
        grad_clip (Optional[Union[nn.ClipGradByNorm, nn.ClipGradByValue, nn.ClipGradByGlobalNorm]], optional):
            Gradient cliping strategy. Defaults to None.
    """

    def __init__(
        self,
        learning_rate: Union[float, optim.lr.LRScheduler] = 0.001,
        weight_decay: Optional[
            Union[float, regularizer.L1Decay, regularizer.L2Decay]
        ] = None,
        grad_clip: Optional[
            Union[nn.ClipGradByNorm, nn.ClipGradByValue, nn.ClipGradByGlobalNorm]
        ] = None,
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def __call__(self, model_list: Tuple[nn.Layer, ...]):
        # model_list is None in static graph
        parameters = (
            sum([m.parameters() for m in model_list], []) if model_list else None
        )
        opt = optim.SGD(
            learning_rate=self.learning_rate,
            parameters=parameters,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
        )
        return opt


class Momentum(object):
    """Simple Momentum optimizer with velocity state.

    Args:
        learning_rate (Union[float, optim.lr.LRScheduler]): The learning rate
            used to update parameter(s).
        momentum (float): Momentum factor.
        weight_decay (Optional[Union[float, regularizer.L1Decay, regularizer.L2Decay]], optional):
            Regularization strategy. Defaults to None.
        grad_clip (Optional[Union[nn.ClipGradByNorm, nn.ClipGradByValue, nn.ClipGradByGlobalNorm]], optional):
            Gradient cliping strategy. Defaults to None.
        use_nesterov (bool, optional): Whether to use nesterov momentum. Defaults to False.
        no_weight_decay_name (Optional[str], optional): List of names of no weight decay parameters split by white space. Defaults to None.
    """

    def __init__(
        self,
        learning_rate: Union[float, optim.lr.LRScheduler],
        momentum: float,
        weight_decay: Optional[
            Union[float, regularizer.L1Decay, regularizer.L2Decay]
        ] = None,
        grad_clip: Optional[
            Union[nn.ClipGradByNorm, nn.ClipGradByValue, nn.ClipGradByGlobalNorm]
        ] = None,
        use_nesterov: bool = False,
        no_weight_decay_name: Optional[str] = None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.use_nesterov = use_nesterov
        self.no_weight_decay_name_list = (
            no_weight_decay_name.split() if no_weight_decay_name else []
        )

    def __call__(self, model_list: Tuple[nn.Layer, ...]):
        # model_list is None in static graph
        parameters = None
        if len(self.no_weight_decay_name_list) > 0:
            params_with_decay = []
            params_without_decay = []
            for m in model_list:
                params = [
                    p
                    for n, p in m.named_parameters()
                    if not any(nd in n for nd in self.no_weight_decay_name_list)
                ]
                params_with_decay.extend(params)
                params = [
                    p
                    for n, p in m.named_parameters()
                    if any(nd in n for nd in self.no_weight_decay_name_list)
                ]
                params_without_decay.extend(params)
            parameters = [
                {"params": params_with_decay, "weight_decay": self.weight_decay},
                {"params": params_without_decay, "weight_decay": 0.0},
            ]
        else:
            parameters = (
                sum([m.parameters() for m in model_list], []) if model_list else None
            )
        opt = optim.Momentum(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            use_nesterov=self.use_nesterov,
            parameters=parameters,
        )
        if hasattr(opt, "_use_multi_tensor"):
            opt = optim.Momentum(
                learning_rate=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                grad_clip=self.grad_clip,
                parameters=parameters,
                use_nesterov=self.use_nesterov,
                use_multi_tensor=True,
            )
        return opt


class Adam(object):
    """Adam: A Method for Stochastic Optimization.

    Args:
        learning_rate (Union[float, optim.lr.LRScheduler], optional): The learning rate
            used to update parameter(s). Defaults to 0.001.
        beta1 (float, optional): The exponential decay rate for the 1st moment estimates. Defaults to 0.9.
        beta2 (float, optional): The exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
        epsilon (float, optional): A small float value for numerical stability. Defaults to 1e-08.
        weight_decay (Optional[Union[float, regularizer.L1Decay, regularizer.L2Decay]], optional): Regularization strategy. Defaults to None.
        grad_clip (Optional[Union[nn.ClipGradByNorm, nn.ClipGradByValue, nn.ClipGradByGlobalNorm]], optional): Gradient cliping strategy. Defaults to None.
        lazy_mode (bool, optional): Whether to enable lazy mode for moving-average. Defaults to False.
    """

    def __init__(
        self,
        learning_rate: Union[float, optim.lr.LRScheduler] = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-08,
        weight_decay: Optional[
            Union[float, regularizer.L1Decay, regularizer.L2Decay]
        ] = None,
        grad_clip: Optional[
            Union[nn.ClipGradByNorm, nn.ClipGradByValue, nn.ClipGradByGlobalNorm]
        ] = None,
        lazy_mode: bool = False,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.lazy_mode = lazy_mode

    def __call__(self, model_list: Tuple[nn.Layer, ...]):
        # model_list is None in static graph
        parameters = (
            sum([m.parameters() for m in model_list], []) if model_list else None
        )
        opt = optim.Adam(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            lazy_mode=self.lazy_mode,
            parameters=parameters,
        )
        return opt


class LBFGS(object):
    """The L-BFGS is a quasi-Newton method for solving an unconstrained optimization
        problem over a differentiable function. Closely related is the Newton method for minimization.

    Args:
        learning_rate (float, optional): The learning rate
            used to update parameter(s). Defaults to 1.0.
        max_iter (int, optional): Maximal number of iterations per optimization step.
            Defaults to 1.
        max_eval (Optional[int], optional): Maximal number of function evaluations per
            optimization step. Defaults to None.
        tolerance_grad (float, optional): Termination tolerance on first order optimality.
            Defaults to 1e-07.
        tolerance_change (float, optional): termination tolerance on function
            value/parameterchanges. Defaults to 1e-09.
        history_size (int, optional): Update history size. Defaults to 100.
        line_search_fn (Optional[Literal["strong_wolfe"]], optional): Either 'strong_wolfe' or None.
            Defaults to "strong_wolfe".
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        max_iter: int = 1,
        max_eval: Optional[int] = None,
        tolerance_grad: float = 1e-07,
        tolerance_change: float = 1e-09,
        history_size: int = 100,
        line_search_fn: Optional[Literal["strong_wolfe"]] = "strong_wolfe",
    ):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change
        self.history_size = history_size
        self.line_search_fn = line_search_fn

    def __call__(self, model_list: Tuple[nn.Layer, ...]):
        # model_list is None in static graph
        parameters = (
            sum([m.parameters() for m in model_list], []) if model_list else None
        )
        opt = incubate_optim.LBFGS(
            lr=self.lr,
            max_iter=self.max_iter,
            max_eval=self.max_eval,
            tolerance_grad=self.tolerance_grad,
            tolerance_change=self.tolerance_change,
            history_size=self.history_size,
            line_search_fn=self.line_search_fn,
            parameters=parameters,
        )
        return opt


class RMSProp(object):
    """Root Mean Squared Propagation (RMSProp) is an unpublished, adaptive learning rate method.

    Args:
        learning_rate (Union[float, optim.lr.LRScheduler]): The learning rate
            used to update parameter(s)
        rho (float, optional): Factor ρ in equation. Defaults to 0.95.
        epsilon (float, optional): Factor ϵ in equation as a smoothing term. Defaults to 1e-6.
        momentum (float, optional):β in equation is the momentum term. Defaults to 0.0.
        weight_decay (Optional[Union[float, regularizer.L1Decay, regularizer.L2Decay]], optional):
            Regularization strategy. Defaults to None.
        grad_clip (Optional[Union[nn.ClipGradByNorm, nn.ClipGradByValue, nn.ClipGradByGlobalNorm]], optional):
            Gradient cliping strategy. Defaults to None.
    """

    def __init__(
        self,
        learning_rate: Union[float, optim.lr.LRScheduler],
        rho: float = 0.95,
        epsilon: float = 1e-6,
        momentum: float = 0.0,
        weight_decay: Optional[
            Union[float, regularizer.L1Decay, regularizer.L2Decay]
        ] = None,
        grad_clip: Optional[
            Union[nn.ClipGradByNorm, nn.ClipGradByValue, nn.ClipGradByGlobalNorm]
        ] = None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.rho = rho
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def __call__(self, model_list: Tuple[nn.Layer, ...]):
        # model_list is None in static graph
        parameters = (
            sum([m.parameters() for m in model_list], []) if model_list else None
        )
        opt = optim.RMSProp(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            rho=self.rho,
            epsilon=self.epsilon,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            parameters=parameters,
        )
        return opt


class AdamW(object):
    """AdamW is implemented based on DECOUPLED WEIGHT DECAY REGULARIZATION.

    Args:
        learning_rate (Union[float, optim.lr.LRScheduler], optional): The learning rate
            used to update parameter(s). Defaults to 0.001.
        beta1 (float, optional): The exponential decay rate for the 1st moment estimates. Defaults to 0.9.
        beta2 (float, optional): The exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
        epsilon (float, optional): A small float value for numerical stability. Defaults to 1e-8.
        weight_decay (Optional[Union[float, regularizer.L1Decay, regularizer.L2Decay]], optional): Regularization strategy. Defaults to None.
        grad_clip (Optional[Union[nn.ClipGradByNorm, nn.ClipGradByValue, nn.ClipGradByGlobalNorm]], optional): Gradient cliping strategy. Defaults to None.
        no_weight_decay_name (Optional[str], optional): List of names of no weight decay parameters split by white space. Defaults to None.
        one_dim_param_no_weight_decay (bool, optional): Apply no weight decay on 1-D parameter(s). Defaults to False.
    """

    def __init__(
        self,
        learning_rate: Union[float, optim.lr.LRScheduler] = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: Optional[
            Union[float, regularizer.L1Decay, regularizer.L2Decay]
        ] = None,
        grad_clip: Optional[
            Union[nn.ClipGradByNorm, nn.ClipGradByValue, nn.ClipGradByGlobalNorm]
        ] = None,
        no_weight_decay_name: Optional[str] = None,
        one_dim_param_no_weight_decay: bool = False,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.no_weight_decay_name_list = (
            no_weight_decay_name.split() if no_weight_decay_name else []
        )
        self.one_dim_param_no_weight_decay = one_dim_param_no_weight_decay

    def __call__(self, model_list: Tuple[nn.Layer, ...]):
        # model_list is None in static graph
        parameters = (
            sum([m.parameters() for m in model_list], []) if model_list else None
        )

        # TODO(gaotingquan): model_list is None when in static graph, "no_weight_decay" not work.
        if model_list is None:
            if (
                self.one_dim_param_no_weight_decay
                or len(self.no_weight_decay_name_list) != 0
            ):
                msg = '"AdamW" does not support setting "no_weight_decay" in static graph. Please use dynamic graph.'
                logger.error(Exception(msg))
                raise Exception(msg)

        self.no_weight_decay_param_name_list = (
            [
                p.name
                for model in model_list
                for n, p in model.named_parameters()
                if any(nd in n for nd in self.no_weight_decay_name_list)
            ]
            if model_list
            else []
        )

        if self.one_dim_param_no_weight_decay:
            self.no_weight_decay_param_name_list += (
                [
                    p.name
                    for model in model_list
                    for n, p in model.named_parameters()
                    if len(p.shape) == 1
                ]
                if model_list
                else []
            )

        opt = optim.AdamW(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            parameters=parameters,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            apply_decay_param_fun=self._apply_decay_param_fun,
        )
        return opt

    def _apply_decay_param_fun(self, name):
        return name not in self.no_weight_decay_param_name_list
