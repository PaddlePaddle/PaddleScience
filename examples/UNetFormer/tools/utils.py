import collections
import copy
import os
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import paddle


def merge_dicts(*dicts: dict) -> dict:
    """Recursive dict merge.
    Instead of updating only top-level keys,
    ``merge_dicts`` recurses down into dicts nested
    to an arbitrary depth, updating keys.

    Args:
        *dicts: several dictionaries to merge

    Returns:
        dict: deep-merged dictionary
    """
    assert len(dicts) > 1
    dict_ = copy.deepcopy(dicts[0])
    for merge_dict in dicts[1:]:
        merge_dict = merge_dict or {}
        for k in merge_dict:
            if (
                k in dict_
                and isinstance(dict_[k], dict)
                and isinstance(merge_dict[k], collections.Mapping)
            ):
                dict_[k] = merge_dicts(dict_[k], merge_dict[k])
            else:
                dict_[k] = merge_dict[k]
    return dict_


def process_model_params(
    model: paddle.nn.Layer,
    layerwise_params: Dict[str, dict] = None,
    no_bias_weight_decay: bool = True,
    lr_scaling: float = 1.0,
) -> List[Union[paddle.base.framework.EagerParamBase.from_tensor, dict]]:
    """Gains model parameters for ``torch.optim.Optimizer``.

    Args:
        model (torch.nn.Module): Model to process
        layerwise_params (Dict): Order-sensitive dict where
            each key is regex pattern and values are layer-wise options
            for layers matching with a pattern
        no_bias_weight_decay (bool): If true, removes weight_decay
            for all ``bias`` parameters in the model
        lr_scaling (float): layer-wise learning rate scaling,
            if 1.0, learning rates will not be scaled

    Returns:
        iterable: parameters for an optimizer

    Example::

    """
    params = list(model.named_parameters())
    layerwise_params = layerwise_params or collections.OrderedDict()
    model_params = []
    for name, parameters in params:
        options = {}
        for pattern, pattern_options in layerwise_params.items():
            if re.match(pattern, name) is not None:
                options = merge_dicts(options, pattern_options)
        if no_bias_weight_decay and name.endswith("bias"):
            options["weight_decay"] = 0.0
        if "lr" in options:
            options["lr"] *= lr_scaling
        model_params.append({"params": parameters, **options})
    return model_params


class Lookahead(paddle.optimizer.Optimizer):
    """Implements Lookahead algorithm.

    It has been proposed in `Lookahead Optimizer: k steps forward,
    1 step back`_.

    Adapted from:
    https://github.com/alphadl/lookahead.pytorch (MIT License)

    .. _`Lookahead Optimizer\\: k steps forward, 1 step back`:
        https://arxiv.org/abs/1907.08610
    """

    def __init__(
        self, optimizer: paddle.optimizer.Optimizer, k: int = 5, alpha: float = 0.5
    ):
        """@TODO: Docs. Contribution is welcome."""
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer._param_groups
        self.defaults = {
            'lr': self.optimizer.get_lr(),
            'beta1': self.optimizer.get_,
            'beta2': self.optimizer._beta2,
            'eps': self.optimizer._epsilon(),
            'weight_decay': self.optimizer._weight_decay,
            'amsgrad': self.optimizer._get_amsgrad(),
            'foreach': None,        
            'maximize': False,      
            'capturable': False,    
            'differentiable': False, 
            'fused': None,        
        }
        self.defaults['__len__'] = len(self.defaults)
        self.defaults['__str__'] = lambda: f"OptimizerDefaults({self.defaults})"
        self.defaults['__repr__'] = lambda: f"<OptimizerDefaults> with keys: {list(self.defaults.keys())}"
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        """@TODO: Docs. Contribution is welcome."""
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = paddle.zeros_like(x=fast.data)
                paddle.assign(fast.data, output=param_state["slow_param"])
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        """@TODO: Docs. Contribution is welcome."""
        for group in self.param_groups:
            self.update(group)

    def step(self, closure: Optional[Callable] = None):
        """Makes optimizer step.

        Args:
            closure (callable, optional): A closure that reevaluates
                the model and returns the loss.

        Returns:
            computed loss
        """
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        """@TODO: Docs. Contribution is welcome."""
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, paddle.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        """@TODO: Docs. Contribution is welcome."""
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).set_state_dict(state_dict=slow_state_dict)
        self.optimizer.set_state_dict(state_dict=fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        """@TODO: Docs. Contribution is welcome."""
        param_group["counter"] = 0
        """Not Support auto convert *.add_param_group, please judge whether it is Pytorch API and convert by yourself"""
        self.optimizer.add_param_group(param_group)

    @classmethod
    def get_from_params(
        cls, params: Dict, base_optimizer_params: Dict = None, **kwargs
    ) -> "Lookahead":
        """@TODO: Docs. Contribution is welcome."""
        from catalyst.registry import OPTIMIZERS

        base_optimizer = OPTIMIZERS.get_from_params(
            params=params, **base_optimizer_params
        )
        optimizer = cls(optimizer=base_optimizer, **kwargs)
        return optimizer
