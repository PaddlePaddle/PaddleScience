from typing import Any, Tuple, Type, Union

import paddle

import paddle_geometric.typing

# Placeholder for potential PaddlePaddle dynamic optimization in the future
# Currently, no equivalent exists for `torch._dynamo.OptimizedModule`.

def is_paddle_instance(obj: Any, cls: Union[Type, Tuple[Type]]) -> bool:
    r"""Checks if the :obj:`obj` is an instance of a :obj:`cls`.

    This function extends :meth:`isinstance` to be applicable for any dynamic
    optimization features PaddlePaddle may introduce in the future.

    Args:
        obj (Any): The object to check.
        cls (Union[Type, Tuple[Type]]): The class or tuple of classes to check against.

    Returns:
        bool: Whether the object is an instance of the given class or classes.
    """
    # PaddlePaddle currently does not have an equivalent to `torch._dynamo.OptimizedModule`.
    # This placeholder ensures future compatibility if such a feature is introduced.
    return isinstance(obj, cls)
