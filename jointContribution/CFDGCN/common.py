import math
import pathlib
import warnings
from typing import BinaryIO
from typing import List
from typing import Optional
from typing import Text
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import paddle
from PIL import Image

plt.switch_backend("agg")


def pad_sequence(
    sequences: List[paddle.Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
) -> paddle.Tensor:
    r"""Pad a list of variable length Tensors with ``padding_value``
    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.
    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.
    Example:
        >>> a = paddle.ones(25, 300)
        >>> b = paddle.ones(22, 300)
        >>> c = paddle.ones(15, 300)
        >>> pad_sequence([a, b, c]).shape
        paddle.Tensor([25, 3, 300])
    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.
    Args:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.
    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = paddle.shape(sequences[0])
    trailing_dims = (
        tuple(max_size[1:].numpy().tolist()) if sequences[0].ndim >= 2 else ()
    )
    max_len = max([s.shape[0] for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims
    out_tensor = paddle.full(out_dims, padding_value, sequences[0].dtype)
    for i, tensor in enumerate(sequences):
        length = tensor.shape[0]
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            if length != 0:
                out_tensor[i, :length] = tensor
            else:
                out_tensor[i, length] = tensor
        else:
            if length != 0:
                out_tensor[:length, i] = tensor
            else:
                out_tensor[length, i] = tensor

    return out_tensor


@paddle.no_grad()
def make_grid(
    tensor: Union[paddle.Tensor, List[paddle.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    **kwargs,
) -> paddle.Tensor:
    if not (
        isinstance(tensor, paddle.Tensor)
        or (
            isinstance(tensor, list)
            and all(isinstance(t, paddle.Tensor) for t in tensor)
        )
    ):
        raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    if "range" in kwargs:
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = paddle.stack(tensor, axis=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = paddle.concat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = paddle.concat((tensor, tensor, tensor), 1)

    if normalize is True:
        if value_range is not None:
            if not isinstance(value_range, tuple):
                raise ValueError(
                    "value_range has to be a tuple (min, max) if specified. min and max are numbers"
                )

        def norm_input_pic(img, low, high):
            img.clip(min=low, max=high)
            img = img - low
            img = img / max(high - low, 1e-5)

        def norm_range(t, value_range):
            if value_range is not None:
                norm_input_pic(t, value_range[0], value_range[1])
            else:
                norm_input_pic(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.shape[0] == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    num_channels = tensor.shape[1]
    grid = paddle.full(
        (num_channels, height * ymaps + padding, width * xmaps + padding), pad_value
    )
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[
                :,
                y * height + padding : (y + 1) * height,
                x * width + padding : (x + 1) * width,
            ] = tensor[k]
            k = k + 1
    return grid


@paddle.no_grad()
def save_image(
    tensor: Union[paddle.Tensor, List[paddle.Tensor]],
    fp: Union[Text, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs,
) -> None:
    grid = make_grid(tensor, **kwargs)
    ndarr = paddle.clip(grid * 255 + 0.5, 0, 255).cast("uint8").numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)
