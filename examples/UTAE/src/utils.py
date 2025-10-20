"""
Utility functions (Paddle Version)
"""
import collections.abc
import re

import numpy as np
import paddle

np_str_obj_array_pattern = re.compile(r"[SaUO]")

"""Pad tensor to target shape for all dimensions"""


def pad_tensor(x, target_shape, pad_value=0):

    if len(x.shape) != len(target_shape):
        raise ValueError(f"Shape mismatch: {x.shape} vs {target_shape}")

    # Check if padding is needed
    if tuple(x.shape) == tuple(target_shape):
        return x

    # Calculate padding for each dimension
    # Paddle padding format: [dim_n_left, dim_n_right, dim_{n-1}_left, dim_{n-1}_right, ...]
    # For 2D: [dim1_left, dim1_right, dim0_left, dim0_right]
    # For 4D: [dim3_left, dim3_right, dim2_left, dim2_right, dim1_left, dim1_right, dim0_left, dim0_right]

    pad = []
    needs_padding = False

    # Build padding list from last dimension to first
    # But we need to add pairs in the correct order for Paddle
    pad_pairs = []
    for i in range(len(x.shape) - 1, -1, -1):
        pad_size = target_shape[i] - x.shape[i]
        if pad_size < 0:
            raise ValueError(
                f"Target size {target_shape[i]} smaller than current size {x.shape[i]} in dim {i}"
            )

        if pad_size > 0:
            needs_padding = True

        # Store [left_pad, right_pad] for this dimension
        pad_pairs.append([0, pad_size])

    # Reverse the pairs to match Paddle's expected order
    pad_pairs.reverse()

    # Flatten the pairs into the final pad list
    for pair in pad_pairs:
        pad.extend(pair)

    if not needs_padding:
        return x

    # Apply padding
    result = paddle.nn.functional.pad(x, pad=pad, value=pad_value)

    # Verify result shape
    if tuple(result.shape) != tuple(target_shape):
        # Print debug info for troubleshooting
        print(
            f"Debug: input_shape={x.shape}, target_shape={target_shape}, pad={pad}, result_shape={result.shape}"
        )
        raise ValueError(f"Padding failed: expected {target_shape}, got {result.shape}")

    return result


"""Get the maximum shape across all tensors in batch"""


def get_max_shape(batch):

    if not batch:
        return None

    max_shape = list(batch[0].shape)
    for tensor in batch[1:]:
        for i, size in enumerate(tensor.shape):
            max_shape[i] = max(max_shape[i], size)

    return tuple(max_shape)


"""
Modified default_collate from the official pytorch repo for padding variable length sequences
Adapted from the original PyTorch implementation
"""


def pad_collate(batch, pad_value=0):

    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, paddle.Tensor):
        _out = None
        if len(elem.shape) > 0:
            # Check if any shapes differ
            shapes = [e.shape for e in batch]
            if not all(s == shapes[0] for s in shapes):
                # Get maximum shape across all dimensions
                max_shape = get_max_shape(batch)
                # Pad all tensors to max shape
                batch = [pad_tensor(e, max_shape, pad_value=pad_value) for e in batch]
        return paddle.stack(batch, axis=0)

    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError("Format not managed : {}".format(elem.dtype))
            return pad_collate([paddle.to_tensor(b) for b in batch], pad_value)
        elif elem.shape == ():  # scalars
            return paddle.to_tensor(batch)

    elif isinstance(elem, collections.abc.Mapping):
        return {key: pad_collate([d[key] for d in batch], pad_value) for key in elem}

    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(
            *(pad_collate(list(samples), pad_value) for samples in zip(*batch))
        )

    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = list(zip(*batch))
        return [pad_collate(samples, pad_value) for samples in transposed]

    raise TypeError("Format not managed : {}".format(elem_type))


"""
Set random seed for reproducibility
"""


def set_seed(seed):

    np.random.seed(seed)
    paddle.seed(seed)
