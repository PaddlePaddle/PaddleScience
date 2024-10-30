r"""Normalization layers."""

from functools import partial
from numbers import Integral
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Union

import paddle
from paddle import Tensor

from .lambd import LambdaLayer

NormFunc = Callable[[Tensor], paddle.Tensor]
NormArg = Union[NormFunc, str, Mapping[str, Any], Sequence, None]


def normalization(
    arg: NormArg,
    *args,
    spatial_dims: Optional[int] = None,
    num_features: Optional[int] = None,
    **kwargs
) -> paddle.nn.Layer:
    r"""Create normalization layer.

    Args:
        arg: Custom normalization function or module, or name of normalization layer with optional keyword arguments.
        args: Positional arguments passed to normalization layer.
        num_features: Number of input features.
        spatial_dims: Number of spatial dimensions of input tensors.
        kwargs: Additional keyword arguments for normalization layer. Overrides keyword arguments given as second
            tuple item when ``arg`` is a ``(name, kwargs)`` tuple instead of a string.

    Returns:
        Given normalization function when ``arg`` is a ``paddle.nn.Layer``, or a new normalization layer otherwise.

    """
    if isinstance(arg, paddle.nn.Layer) and not args and not kwargs:
        return arg
    if callable(arg):
        return NormLayer(arg, *args, **kwargs)
    norm_name = "identity"
    norm_args = {}
    if isinstance(arg, str):
        norm_name = arg
    elif isinstance(arg, Mapping):
        norm_name = arg.get("name")
        if not norm_name:
            raise ValueError("normalization() 'arg' map must contain 'name'")
        if not isinstance(norm_name, str):
            raise TypeError("normalization() 'name' must be str")
        norm_args = {key: value for key, value in arg.items() if key != "name"}
    elif isinstance(arg, Sequence):
        if len(arg) != 2:
            raise ValueError("normalization() 'arg' sequence must have length two")
        norm_name, norm_args = arg
        if not isinstance(norm_name, str):
            raise TypeError("normalization() first 'arg' sequence argument must be str")
        if not isinstance(norm_args, dict):
            if norm_name == "group" and isinstance(norm_args, Integral):
                norm_args = dict(num_groups=norm_args)
            else:
                raise TypeError("normalization() second 'arg' sequence argument must be dict")
        norm_args = norm_args.copy()
    elif arg is not None:
        raise TypeError("normalization() 'arg' must be str, mapping, 2-tuple, callable, or None")
    norm_name = norm_name.lower()
    norm_args.update(kwargs)
    if "affine" in norm_args:
        if not norm_args["affine"]:
            norm_args["bias_attr"] = False
            norm_args["weight_attr"] = False
        del norm_args["affine"]
    if norm_name in ("none", "identity"):
        norm = paddle.nn.Identity()
    elif norm_name in ("batch", "batchnorm"):
        if spatial_dims is None:
            raise ValueError("normalization() 'spatial_dims' required for 'batch' norm")
        if spatial_dims < 0 or spatial_dims > 3:
            raise ValueError("normalization() 'spatial_dims' must be 1, 2, or 3")
        if num_features is None:
            raise ValueError("normalization() 'num_features' required for 'batch' norm")
        norm_type = (paddle.nn.BatchNorm1D, paddle.nn.BatchNorm2D, paddle.nn.BatchNorm3D)[
            spatial_dims - 1
        ]
        norm = norm_type(num_features, *args, **norm_args)
    elif norm_name in ("group", "groupnorm"):
        num_groups = norm_args.pop("num_groups", 1)
        if num_features is None:
            if "num_channels" not in norm_args:
                raise ValueError("normalization() 'num_features' required for 'group' norm")
            num_features = norm_args.pop("num_channels")
        norm = paddle.nn.GroupNorm(num_groups, num_features, *args, **norm_args)
    elif norm_name in ("layer", "layernorm"):
        if num_features is None:
            raise ValueError("normalization() 'num_features' required for 'layer' norm")
        # See examples at https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/GroupNorm_cn.html#groupnorm
        # and Figure 2 of "Group Normalization" paper (https://arxiv.org/abs/1803.08494).
        norm = paddle.nn.GroupNorm(1, num_features, *args, **norm_args)
    elif norm_name in ("instance", "instancenorm"):
        if spatial_dims is None:
            raise ValueError("normalization() 'spatial_dims' required for 'instance' norm")
        if spatial_dims < 0 or spatial_dims > 3:
            raise ValueError("normalization() 'spatial_dims' must be 1, 2, or 3")
        if num_features is None:
            raise ValueError("normalization() 'num_features' required for 'instance' norm")
        norm_type = (paddle.nn.InstanceNorm1D, paddle.nn.InstanceNorm2D, paddle.nn.InstanceNorm3D)[
            spatial_dims - 1
        ]
        norm = norm_type(num_features, *args, **norm_args)
    else:
        raise ValueError("normalization() unknown layer type {norm_name!r}")
    return norm


def norm_layer(*args, **kwargs) -> paddle.nn.Layer:
    r"""Create normalization layer, see ``normalization``."""
    return normalization(*args, **kwargs)


def is_norm_layer(arg: Any) -> bool:
    r"""Whether given module is a normalization layer."""
    if isinstance(arg, NormLayer):
        return True
    return is_batch_norm(arg) or is_group_norm(arg) or is_instance_norm(arg)


def is_batch_norm(arg: Any) -> bool:
    r"""Whether given module is a batch normalization layer."""
    return isinstance(arg, (paddle.nn.BatchNorm1D, paddle.nn.BatchNorm2D, paddle.nn.BatchNorm3D))


def is_group_norm(arg: Any) -> bool:
    r"""Whether given module is a group normalization layer."""
    return isinstance(arg, paddle.nn.GroupNorm)


def is_instance_norm(arg: Any) -> bool:
    r"""Whether given module is an instance normalization layer."""
    return isinstance(
        arg, (paddle.nn.InstanceNorm1D, paddle.nn.InstanceNorm2D, paddle.nn.InstanceNorm3D)
    )


class NormLayer(LambdaLayer):
    r"""Normalization layer."""

    def __init__(
        self,
        arg: NormArg,
        *args,
        spatial_dims: Optional[int] = None,
        num_features: Optional[int] = None,
        **kwargs: Mapping[str, Any]
    ) -> None:
        if callable(arg):
            norm = partial(arg, *args, **kwargs) if args or kwargs else arg
        else:
            kwargs.update(dict(spatial_dims=spatial_dims, num_features=num_features))
            norm = normalization(arg, *args, **kwargs)
        super().__init__(norm)
