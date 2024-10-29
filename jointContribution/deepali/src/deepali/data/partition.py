r"""Auxiliary types and functions to work with dataset partitions."""
from __future__ import annotations  # noqa

from enum import Enum
from itertools import accumulate
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import paddle

__all__ = "Partition", "dataset_split_lengths", "random_split_indices"


class Partition(Enum):
    r"""Enumeration of dataset partitions / splits."""

    NONE = "none"
    TEST = "test"  # use for testing a trained model
    TRAIN = "train"  # use for model training backward pass
    VALID = "valid"  # use for model training evaluation (validation)

    @classmethod
    def from_arg(cls, arg: Union[Partition, str, None]) -> Partition:
        r"""Create enumeration value from function argument."""
        if arg is None:
            return cls.NONE
        return cls(arg)


def dataset_split_lengths(
    total: int, ratios: Union[float, Sequence[float]]
) -> Tuple[int, int, int]:
    r"""Split dataset in training, validation, and test subset.

    The output ``lengths`` of this function can be passed to ``paddle.io.random_split`` to obtain
    the ``paddle.io.Subset`` for each split.

    See also:
        https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/random_split_cn.html

    Args:
        total: Total number of samples in dataset.
        ratios: Fraction of samples in each split. When a float or 1-tuple is given,
            the specified fraction of samples is used for training and all remaining
            samples for validation during training. When a 2-tuple is given, the test
            set is assigned no samples. Otherwise, a 3-tuple consisting of ratios
            for training, validation, and test set, respectively should be given.
            The ratios must sum to one.

    Returns:
        lengths: Number of dataset samples in each subset.

    """
    if not isinstance(ratios, float) and len(ratios) == 1:
        ratios = ratios[0]
    if isinstance(ratios, float):
        ratios = ratios, 1.0 - ratios
    if len(ratios) == 2:
        ratios += (0.0,)
    elif len(ratios) != 3:
        raise ValueError(
            "dataset_split_lengths() 'ratios' must be float or tuple of length 1, 2, or 3"
        )
    if ratios[0] <= 0 or ratios[0] > 1:
        raise ValueError("dataset_split_lengths() training split ratio must be in (0, 1]")
    if any([(ratio < 0 or ratio > 1) for ratio in ratios]):
        raise ValueError("dataset_split_lengths() ratios must be in [0, 1]")
    if sum(ratios) != 1:
        raise ValueError("dataset_split_lengths() 'ratios' must sum to one")
    lengths = [int(round(ratio * total)) for ratio in ratios]
    lengths[2] = max(0, lengths[2] + (total - sum(lengths)))
    lengths[1] = max(0, lengths[1] + (total - sum(lengths)))
    assert sum(lengths) == total
    return tuple(lengths)


def random_split_indices(lengths: Sequence[int], generator: None) -> List[List[int]]:
    r"""Randomly split dataset indices into non-overlapping sets of given lengths.

    Args:
        lengths: Lengths of splits to be produced.
        generator: Generator used for the random permutation.

    Returns:
        Lists of specified ``lengths`` with randomly selected indices in ``[0, sum(lengths))``.

    """
    subsets = []
    indices = paddle.randperm(n=sum(lengths)).tolist()
    for offset, length in zip(accumulate(lengths), lengths):
        subsets.append(indices[offset - length : offset])
    return subsets
