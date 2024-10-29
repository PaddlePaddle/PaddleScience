r"""Custom samplers for ``paddle.utils.data.DataLoader``."""
import math
import multiprocessing as mp
from typing import Optional
from typing import Sequence

import paddle


class DistributedWeightedRandomSampler(paddle.io.Sampler):
    r"""A version of WeightedRandomSampler that can be used with DistributedDataParallel training."""

    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int = -1,
        replacement: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        r"""Initialize map-style dataset index sampler.

        Sampler that restricts data loading to a subset of the random samples.
        It is especially useful in conjunction with ``paddle.nn.parallel.DistributedDataParallel``.
        In such a case, each process can pass a ``paddle.utils.data.DistributedWeightedRandomSampler``
        instance as a ``paddle.utils.data.DataLoader`` sampler, and load a subset of the
        original dataset that is exclusive to it.

        .. note::
            Dataset is assumed to be a map-style dataset of constant size.

        Args:
            weights: Sampling weights for each of the ``len(weights)`` indices.
                These are passed on to ``paddle.multinomial()`` to sample indices.
            num_samples: Number of dataset indices to sample. If negative, use ``len(weights)``.
                Note that using all samples would only result in shuffling the dataset if
                ``replacement=False``, but no weighted sampling of a subset of it.
            replacement: Whether to sample dataset indices with replacement.
            num_replicas: Number of processes participating in distributed training.
                By default, ``world_size`` is retrieved from the current distributed group.
            rank: Rank of the current process within ``num_replicas``. By default, ``rank`` is
                retrieved from the current distributed group.
            shuffle: If ``True``, the current ``epoch`` is added to the ``seed`` value to draw
                different samples at each epoch. Moreover, when sampling without replacement,
                the sampler will further shuffle the randomly drawn indices.
            seed: Random seed used to draw and shuffle random samples. This number should be identical
                across all processes in the distributed group.
            drop_last: If ``True``, then the sampler will drop the tail of the data to make it evenly
                divisible across the number of replicas. If ``False``, the sampler will add extra indices
                to make the data evenly divisible across the replicas.

        .. warning::
            In distributed mode, calling the :meth:``set_epoch`` method at the beginning of each epoch
            **before** creating the ``DataLoader`` iterator is necessary to make shuffling work properly
            across multiple epochs. Otherwise, the same ordering will always be used. It should further
            be noted, that the ``epoch`` is stored in shared memory such that persistent worker processes
            all receive an update of the epoch number when ``set_epoch`` is called.

        """
        if num_replicas is None:
            if not paddle.distributed.is_available():
                raise RuntimeError(
                    f"{type(self).__name__}() requires distributed package to be available"
                )
            num_replicas = paddle.distributed.get_world_size()
        if rank is None:
            if not paddle.distributed.is_available():
                raise RuntimeError(
                    f"{type(self).__name__}() requires distributed package to be available"
                )
            rank = paddle.distributed.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"{type(self).__name__}() invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        if num_samples < 1:
            num_samples = len(weights)
        self.weights = paddle.to_tensor(data=weights, dtype="float32")
        self.replacement = replacement
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        if self.drop_last and num_samples % self.num_replicas != 0:
            self.num_samples = math.ceil((num_samples - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(num_samples / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        # When a DataLoader with num_workers>0 and persistent_workers=True is used,
        # each persistent worker holds a copy of the sampler object. Calls of set_epoch()
        # would thus not be reflected in the worker processes. In order to still enable
        # an update of the epoch number, a synchronized value in shared memory is used.
        self._epoch = mp.Value("Q", 0)
        # We cannot sampler more indices than available if replacement=False
        if not replacement and self.total_size > len(self.weights):
            raise ValueError(
                f"{type(self).__name__}() total number of samples is greater than number of available samples to draw without replacement; reduce 'num_samples' and/or enable 'drop_last'"
            )

    @property
    def epoch(self) -> int:
        return self._epoch.value

    def set_epoch(self, epoch: int) -> None:
        r"""Set training epoch used to adjust seed of number generator when shuffling is enabled."""
        if epoch < 0:
            raise ValueError(f"{type(self).__name__}.set_epoch() 'epoch' must be non-negative")
        self._epoch.value = epoch

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        g = paddle.framework.core.default_cpu_generator()
        g = g.manual_seed(self.seed + (self.epoch if self.shuffle else 0))
        indices = paddle.multinomial(
            x=self.weights, num_samples=self.total_size, replacement=self.replacement
        )
        indices = indices.tolist()
        if self.shuffle and not self.replacement:
            perm = paddle.randperm(n=len(indices)).tolist()
            indices = [indices[j] for j in perm]
        assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)
