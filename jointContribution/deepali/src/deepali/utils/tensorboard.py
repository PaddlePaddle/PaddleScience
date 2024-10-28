r"""Auxiliary functions for writing to `TensorBoard <https://www.tensorflow.org/tensorboard>`_ event files."""
import re
from typing import Callable
from typing import Optional
from typing import Union

import paddle
from paddle import Tensor

from ..core.image import normalize_image
from ..core.typing import TensorCollection

RE_CHANNEL_INDEX = re.compile(r"\{c(:[^}]+)?\}")


def escape_channel_index_format_string(tag: str) -> str:
    r"""Escape image channel index format before str.format() call."""
    return RE_CHANNEL_INDEX.sub(r"{{c\1}}", tag)


def add_summary_image(
    writer,
    tag: str,
    image: paddle.Tensor,
    global_step: Optional[int] = None,
    walltime: Optional[float] = None,
    image_transform: Union[bool, Callable[[str, Tensor], paddle.Tensor]] = True,
    rescale_transform: Union[bool, Callable[[str, Tensor], paddle.Tensor]] = False,
    channel_offset: int = 0,
) -> None:
    r"""Add image to TensorBoard summary.

    Args:
        writer: TensorBoard writer with open event file.
        tag: Image tag passed to ``writer.add_image``.
        image: Image data tensor.
        global_step: Global step value to record.
        walltime: Optional override default walltime seconds.
        image_transform: Callable used to extract a 2D image tensor of shape
            ``(C, Y, X)`` from ``data``. When a multi-channel tensor is returnd,
            each channel is saved as separate image to the TensorBoard event file.
            with channel index appended to the ``tag`` separated with an underscore ``_``.
            By default, the central slice of the first image in the batch is extracted.
            The first argument is the ``tag`` of the image tensor. This can be used to
            differently process different images by the same callable transform.
        rescale_transform: Image intensities must be in the closed interval ``[0, 1]``.
            When ``False``, this must already the case, e.g., as part of ``image_transform``.
            When ``True``, images with values outside this interval are rescaled unless
            the ``tag`` matches "y" or "y_pred".
        channel_offset: Offset to add to channel index in tag format string.

    """
    img = image.detach()
    if image_transform is True:
        image_transform = first_central_image_slice
    if image_transform not in (False, None):
        img = image_transform(tag, img)
    if rescale_transform is True:
        rescale_transform = normalize_summary_image
    if rescale_transform not in (False, None):
        img = rescale_transform(tag, img)
    if img.ndim != 3:
        raise AssertionError("add_summary_image() transformed tensor must have shape (C, H, W)")
    if tuple(img.shape)[0] > 1 and RE_CHANNEL_INDEX.search(tag) is None:
        tag = tag + "/{c}"
    kwargs = dict(global_step=global_step, walltime=walltime)
    for c in range(tuple(img.shape)[0]):
        start_58 = img.shape[0] + c if c < 0 else c
        writer.add_image(
            tag.format(c=c + channel_offset),
            paddle.slice(img, [0], [start_58], [start_58 + 1]),
            **kwargs
        )


def add_summary_images(
    writer,
    prefix: str,
    images: TensorCollection,
    global_step: Optional[int] = None,
    walltime: Optional[float] = None,
    image_transform: Union[bool, Callable[[str, Tensor], paddle.Tensor]] = True,
    rescale_transform: Union[bool, Callable[[str, Tensor], paddle.Tensor]] = False,
    channel_offset: int = 0,
) -> None:
    r"""Add slices of image tensors to TensorBoard summary.

    Args:
        writer: TensorBoard writer with open event file.
        prefix: Prefix string for TensorBoard tags.
        images: Possibly nested dictionary and/or sequence of image tensors.
        global_step: Global step value to record.
        walltime: Optional override default walltime seconds.
        image_transform: Callable used to extract a 2D image tensor of shape
            ``(C, Y, X)`` from each image. When a multi-channel tensor is returnd (C > 1),
            each channel is saved as separate image to the TensorBoard event file.
            By default, the central slice of the first image in the batch is extracted.
            The first argument is the name of the image tensor if applicable, or an empty
            string otherwise. This can be used to differently process different images.
        rescale_transform: Image intensities must be in the closed interval ``[0, 1]``.
            Set to ``False``, if this is already the case or when a custom
            ``image_transform`` is used which normalizes the image intensities.
        channel_offset: Offset to add to channel index in tag format string.

    """
    if prefix is None:
        prefix = ""
    if not isinstance(images, dict):
        images = {str(i): value for i, value in enumerate(images)}
    for name, value in images.items():
        if isinstance(value, paddle.Tensor):
            add_summary_image(
                writer,
                prefix + name,
                value,
                global_step=global_step,
                walltime=walltime,
                image_transform=image_transform,
                rescale_transform=rescale_transform,
                channel_offset=channel_offset,
            )
        else:
            add_summary_images(
                writer,
                prefix + name + "/",
                value,
                global_step=global_step,
                walltime=walltime,
                image_transform=image_transform,
                rescale_transform=rescale_transform,
                channel_offset=channel_offset,
            )


def central_image_slices(
    tag: str, data: paddle.Tensor, start: int = 0, length: int = -1
) -> paddle.Tensor:
    if data.ndim < 4 or tuple(data.shape)[1] != 1:
        raise AssertionError(
            "central_image_slices() expects image tensors of shape (N, 1, ..., Y, X)"
        )
    start_59 = data.shape[1] + 0 if 0 < 0 else 0
    data = paddle.slice(data, [1], [start_59], [start_59 + 1]).squeeze(axis=1)
    for dim in range(1, data.ndim - 2):
        start_60 = (
            data.shape[dim] + tuple(data.shape)[dim] // 2
            if tuple(data.shape)[dim] // 2 < 0
            else tuple(data.shape)[dim] // 2
        )
        data = paddle.slice(data, [dim], [start_60], [start_60 + 1])
    for dim in range(1, data.ndim - 2):
        data = data.squeeze(axis=dim)
    if length < 1:
        length = tuple(data.shape)[0]
    length = min(length, tuple(data.shape)[0])
    start_61 = data.shape[0] + start if start < 0 else start
    return paddle.slice(data, [0], [start_61], [start_61 + length])


def all_central_image_slices(tag: str, data: paddle.Tensor) -> paddle.Tensor:
    r"""Extract central slice from each scalar image in batch."""
    return central_image_slices(tag, data, start=0, length=-1)


def first_central_image_slice(tag: str, data: paddle.Tensor) -> paddle.Tensor:
    r"""Extract central slice of first image in batch."""
    return central_image_slices(tag, data, start=0, length=1)


def normalize_summary_image(tag: str, data: paddle.Tensor) -> paddle.Tensor:
    r"""Linearly rescale image values to [0, 1]."""
    return normalize_image(data, mode="unit")
