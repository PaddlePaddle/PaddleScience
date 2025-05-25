from collections.abc import Mapping
from typing import Any, List, Optional, Sequence, Union

import paddle
from paddle.io import DataLoader as PaddleDataLoader

from paddle_geometric.data import Batch, Dataset
from paddle_geometric.data.data import BaseData
from paddle_geometric.data.datapipes import DatasetAdapter
from paddle_geometric.typing import TensorFrame, paddle_frame


class Collater:
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Any]) -> Any:
        elem = batch[0]
        if isinstance(elem, BaseData):
            return Batch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
        elif isinstance(elem, paddle.Tensor):
            return paddle.to_tensor(batch)
        elif isinstance(elem, TensorFrame):
            return paddle_frame.cat(batch, axis=0)
        elif isinstance(elem, float):
            return paddle.to_tensor(batch, dtype='float32')
        elif isinstance(elem, int):
            return paddle.to_tensor(batch, dtype='int64')
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")


class DataLoader(PaddleDataLoader):
    r"""A data loader which merges data objects from a
    :class:`paddle_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~paddle_geometric.data.Data` or
    :class:`~paddle_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`paddle.io.DataLoader`.
    """
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=Collater(dataset, follow_batch, exclude_keys),
            **kwargs,
        )
