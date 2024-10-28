r"""Engine state output transformations."""
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import paddle
from deepali.core import ALIGN_CORNERS
from deepali.core.collections import TensorCollection
from deepali.core.collections import get_tensor
from deepali.core.image import grid_reshape
from deepali.core.tensor import as_one_hot_tensor
from ignite.engine import Engine


def get_output_transform(key: str) -> Callable[[TensorCollection], paddle.Tensor]:
    r"""Get tensor at specified nested engine output map key entry."""

    def output_transform(output: TensorCollection) -> paddle.Tensor:
        return get_tensor(output, key)

    return output_transform


def cm_binary_output_transform(
    y_pred: str = "y_pred", y: str = "y", reshape: str = "none", align_corners: bool = ALIGN_CORNERS
) -> Callable[[TensorCollection], Tuple[paddle.Tensor, paddle.Tensor]]:
    r"""Convert engine output to a vector-valued one-hot prediction.

    Engine output transformation for use with ``ignite.metrics.ConfusionMatrix()``.

    Args:
        output: Ignite engine state output.

    Returns:
        y_pred: Tensor of one-hot encoded predictions with shape ``(N, 2, ..., X)``.
        y: Tensor of target labels with shape ``(N, ..., X)``.

    """

    def output_transform(output: TensorCollection) -> Tuple[paddle.Tensor, paddle.Tensor]:
        dtype = "int32"
        y_pred_tensor = get_tensor(output, y_pred)
        y_tensor = get_tensor(output, y)
        assert y_tensor.ndim == y_pred_tensor.ndim
        assert tuple(y_tensor.shape)[1] == 1
        if tuple(y_tensor.shape)[0] == 1 and tuple(y_pred_tensor.shape)[0] > 1:
            y_tensor = y_tensor.expand(
                shape=tuple(y_pred_tensor.shape)[0:1] + tuple(y_tensor.shape)[1:]
            )
        assert tuple(y_pred_tensor.shape)[0] == tuple(y_tensor.shape)[0]
        assert tuple(y_pred_tensor.shape)[1] == 1
        grid_shape = tuple(y_tensor.shape)[2:]
        if tuple(y_pred_tensor.shape)[2:] != grid_shape and reshape == "y_pred":
            y_pred_tensor = grid_reshape(y_pred_tensor, grid_shape, align_corners=align_corners)
        y_pred_tensor = y_pred_tensor.round().astype("int64")
        y_pred_tensor = as_one_hot_tensor(y_pred_tensor, num_classes=2, dtype=dtype)
        if tuple(y_tensor.shape)[2:] != grid_shape and reshape == "y":
            y_tensor = grid_reshape(y_tensor, grid_shape, align_corners=align_corners)
        y_tensor = y_tensor.round().astype(dtype).squeeze(axis=1)
        assert tuple(y_pred_tensor.shape)[2:] == tuple(y_tensor.shape)[1:]
        return y_pred_tensor, y_tensor

    return output_transform


def cm_multilabel_output_transform(
    y_pred: str = "y_pred", y: str = "y"
) -> Callable[[TensorCollection], Tuple[paddle.Tensor, paddle.Tensor]]:
    r"""Convert engine output to a vector-valued one-hot prediction.

    Engine output transformation for use with ``ignite.metrics.ConfusionMatrix()``.

    Args:
        output: Ignite engine state output.

    Returns:
        y_pred: Tensor of multi-class probabilites with shape ``(N, C, ..., X)``.
        y: Tensor of target labels with shape ``(N, ..., X)``.

    """

    def output_transform(output: TensorCollection) -> Tuple[paddle.Tensor, paddle.Tensor]:
        y_pred_tensor = get_tensor(output, y_pred)
        y_tensor = get_tensor(output, y)
        if tuple(y_tensor.shape)[1] == 1:
            if y_tensor.is_floating_point():
                y_tensor = y_tensor.round().astype("int32")
            y_tensor = y_tensor.squeeze(axis=1)
        else:
            y_tensor = y_tensor.argmax(axis=1)
        if tuple(y_tensor.shape)[0] == 1 and tuple(y_pred_tensor.shape)[0] > 1:
            y_tensor = y_tensor.expand(
                shape=tuple(y_pred_tensor.shape)[0:1] + tuple(y_tensor.shape)[1:]
            )
        return y_pred_tensor, y_tensor

    return output_transform


def cm_output_transform(
    y_pred: str = "y_pred", y: str = "y", multilabel: bool = False
) -> Callable[[TensorCollection], Tuple[paddle.Tensor, paddle.Tensor]]:
    r"""Get engine output transformation for binary or multi-class segmentation output, respectively."""
    if multilabel:
        return cm_multilabel_output_transform(y_pred, y)
    return cm_binary_output_transform(y_pred, y)


def negative_loss_score_function(engine: Engine) -> paddle.Tensor:
    r"""Get negated loss value from ``engine.state.output``.

    This output transformation can be used as ``score_function`` argument of
    ``ignite.handlers.ModelCheckpoint``, for example.

    """
    output = engine.state.output
    if isinstance(output, dict):
        output = output["loss"]
    assert isinstance(output, paddle.Tensor)
    return -output


def y_pred_y_output_transform(
    y_pred: str = "y_pred", y: str = "y", channels: Optional[Union[int, Sequence[int]]] = None
) -> Callable[[TensorCollection], Tuple[paddle.Tensor, paddle.Tensor]]:
    r"""Get engine output transformation which returns (y_pred, y) tuples.

    Args:
        y_pred: Output dictionary key for ``y_pred`` tensor.
        y: Output dictionary key for ``y`` tensor.
        channels: Indices of image channels to extract.

    """
    if isinstance(channels, int):
        channels = (channels,)
    if channels:

        def output_transform(output: TensorCollection) -> Tuple[paddle.Tensor, paddle.Tensor]:
            a = get_tensor(output, y_pred)
            b = get_tensor(output, y)
            return a[:, channels], b[:, channels]

    else:

        def output_transform(output: TensorCollection) -> Tuple[paddle.Tensor, paddle.Tensor]:
            a = get_tensor(output, y_pred)
            b = get_tensor(output, y)
            return a, b

    return output_transform


def y_pred_y_with_weight_output_transform(
    y_pred: str = "y_pred",
    y: str = "y",
    weight: str = "weight",
    kwarg: str = "weight",
    channels: Optional[Union[int, Sequence[int]]] = None,
) -> Callable[[TensorCollection], Tuple[paddle.Tensor, paddle.Tensor]]:
    r"""Get engine output transformation which returns (y_pred, y) tuples.

    Args:
        y_pred: Output dictionary key for ``y_pred`` tensor.
        y: Output dictionary key for ``y`` tensor.
        weight: Output dictionary key for ``weight`` tensor.
        kwarg: Name of weight keyword argument.
        channels: Indices of image channels to extract.

    """
    if isinstance(channels, int):
        channels = (channels,)
    if channels:

        def output_transform(output: TensorCollection) -> Tuple[paddle.Tensor, paddle.Tensor]:
            a = get_tensor(output, y_pred)
            b = get_tensor(output, y)
            w = get_tensor(output, weight)
            return a[:, channels], b[:, channels], {kwarg: w}

    else:

        def output_transform(output: TensorCollection) -> Tuple[paddle.Tensor, paddle.Tensor]:
            a = get_tensor(output, y_pred)
            b = get_tensor(output, y)
            w = get_tensor(output, weight)
            return a, b, {kwarg: w}

    return output_transform
