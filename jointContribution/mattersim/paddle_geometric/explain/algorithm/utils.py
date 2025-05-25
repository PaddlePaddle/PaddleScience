from typing import Dict, Union, Any

import paddle
from paddle import Tensor
from paddle.nn import Layer
from paddle_geometric.nn import MessagePassing
from paddle_geometric.typing import EdgeType


def set_masks(
    model: Layer,
    mask: Union[Tensor, Any],
    edge_index: Tensor,
    apply_sigmoid: bool = True,
):
    r"""Apply mask to every graph layer in the :obj:`model`."""
    loop_mask = edge_index[0] != edge_index[1]

    # Loop over layers and set masks on MessagePassing layers:
    for module in model.sublayers():
        if isinstance(module, MessagePassing):
            # Skip layers that have been explicitly set to `False`:
            if getattr(module, 'explain', True) is False:
                continue

            # Convert mask to a param if it was previously registered as one.
            if not isinstance(mask, paddle.create_parameter) and '_edge_mask' in module._parameters:
                mask = paddle.create_parameter(
                    shape=mask.shape,  # Assuming mask is a tensor, use its shape
                    dtype=mask.dtype,  # Assuming mask already has a dtype, you can also specify it as 'float32'
                    default_initializer=paddle.nn.initializer.Assign(mask)
                )

            module.explain = True
            module._edge_mask = mask
            module._loop_mask = loop_mask
            module._apply_sigmoid = apply_sigmoid


def set_hetero_masks(
    model: Layer,
    mask_dict: Dict[EdgeType, Union[Tensor, Any]],
    edge_index_dict: Dict[EdgeType, Tensor],
    apply_sigmoid: bool = True,
):
    r"""Apply masks to every heterogeneous graph layer in the :obj:`model`
    according to edge types.
    """
    for module in model.sublayers():
        if isinstance(module, paddle.nn.LayerDict):
            for edge_type, mask in mask_dict.items():
                if edge_type in module:
                    edge_level_module = module[edge_type]
                elif '__'.join(edge_type) in module:
                    edge_level_module = module['__'.join(edge_type)]
                else:
                    continue

                set_masks(
                    edge_level_module,
                    mask,
                    edge_index_dict[edge_type],
                    apply_sigmoid=apply_sigmoid,
                )


def clear_masks(model: Layer):
    r"""Clear all masks from the model."""
    for module in model.sublayers():
        if isinstance(module, MessagePassing):
            if getattr(module, 'explain', None) is True:
                module.explain = None
            module._edge_mask = None
            module._loop_mask = None
            module._apply_sigmoid = True
    return module
