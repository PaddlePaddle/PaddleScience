from typing import List

import paddle


# https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/paddlespeech/audio/utils/tensor_utils.py#L40
def pad_sequence(
    sequences: List[paddle.Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
) -> paddle.Tensor:
    r"""Pad a list of variable length Tensors with `padding_value`.

    `pad_sequence` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size `L x *` and if batch_first is False, and `T x B x *`
    otherwise.

    `B` is batch size. It is equal to the number of elements in `sequences`.
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
        This function returns a Tensor of size `T x B x *` or `B x T x *`
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Args:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in `B x T x *` if True, or in
            `T x B x *` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size `T x B x *` if :attr:`batch_first` is `False`.
        Tensor of size `B x T x *` otherwise
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
