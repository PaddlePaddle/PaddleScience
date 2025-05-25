import warnings
from typing import Any, List

import paddle
from paddle import Tensor


def get_embeddings(
    model: paddle.nn.Layer,
    *args: Any,
    **kwargs: Any,
) -> List[Tensor]:
    """Returns the output embeddings of all
    :class:`~paddle_geometric.nn.conv.MessagePassing` layers in
    :obj:`model`.

    Internally, this method registers forward hooks on all
    :class:`~paddle_geometric.nn.conv.MessagePassing` layers of a :obj:`model`,
    and runs the forward pass of the :obj:`model` by calling
    :obj:`model(*args, **kwargs)`.

    Args:
        model (paddle.nn.Layer): The message passing model.
        *args: Arguments passed to the model.
        **kwargs (optional): Additional keyword arguments passed to the model.
    """
    from paddle_geometric.nn import MessagePassing

    embeddings: List[Tensor] = []

    def hook(layer: paddle.nn.Layer, inputs: Any, outputs: Any) -> None:
        # Clone output in case it will be later modified in-place:
        outputs = outputs[0] if isinstance(outputs, tuple) else outputs
        assert isinstance(outputs, Tensor)
        embeddings.append(outputs.clone())

    hook_handles = []
    for layer in model.sublayers():  # Register forward hooks:
        if isinstance(layer, MessagePassing):
            hook_handle = layer.register_forward_post_hook(hook)
            hook_handles.append(hook_handle)

    if len(hook_handles) == 0:
        warnings.warn("The 'model' does not have any 'MessagePassing' layers")

    training = model.training
    model.eval()
    with paddle.no_grad():
        model(*args, **kwargs)
    model.train()

    for handle in hook_handles:  # Remove hooks:
        handle.remove()

    return embeddings
