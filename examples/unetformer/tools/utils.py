import collections
import copy
import re
from typing import Dict
from typing import List
from typing import Union

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
