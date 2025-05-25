import copy
from typing import Any, Callable, Iterator, Optional, Sequence

import paddle

from paddle_geometric.data import Batch
from paddle_geometric.utils import from_smiles

try:
    from paddle.io import DataPipe, functional_datapipe
    from paddle.io.datapipes.iter import Batcher as IterBatcher
except ImportError:
    DataPipe = IterBatcher = object  # Fallback in case of missing dependency

    def functional_datapipe(name: str) -> Callable:
        return lambda cls: cls


@functional_datapipe('batch_graphs')
class Batcher(IterBatcher):
    """
    A custom batching DataPipe to create batches of graphs.

    Args:
        dp (DataPipe): Input DataPipe.
        batch_size (int): Size of each batch.
        drop_last (bool): Whether to drop the last incomplete batch.
    """
    def __init__(
        self,
        dp: DataPipe,
        batch_size: int,
        drop_last: bool = False,
    ) -> None:
        super().__init__(
            dp,
            batch_size=batch_size,
            drop_last=drop_last,
            wrapper_class=Batch.from_data_list,
        )


@functional_datapipe('parse_smiles')
class SMILESParser(DataPipe):
    """
    A DataPipe to parse SMILES strings into graph data.

    Args:
        dp (DataPipe): Input DataPipe containing SMILES strings.
        smiles_key (str): The key for SMILES strings in input dictionaries.
        target_key (Optional[str]): The key for target values in input dictionaries.
    """
    def __init__(
        self,
        dp: DataPipe,
        smiles_key: str = 'smiles',
        target_key: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.dp = dp
        self.smiles_key = smiles_key
        self.target_key = target_key

    def __iter__(self) -> Iterator:
        for d in self.dp:
            if isinstance(d, str):
                # Parse SMILES string directly
                data = from_smiles(d)
            elif isinstance(d, dict):
                # Parse SMILES from dictionary
                data = from_smiles(d[self.smiles_key])
                if self.target_key is not None:
                    y = d.get(self.target_key, None)
                    if y is not None:
                        y = float(y) if len(y) > 0 else float('NaN')
                        data.y = paddle.to_tensor([y], dtype=paddle.float32)
            else:
                raise ValueError(
                    f"'{self.__class__.__name__}' expects either a string or "
                    f"a dictionary as input (got '{type(d)}')"
                )
            yield data


class DatasetAdapter(DataPipe):
    """
    Adapts a dataset for usage with DataPipes.

    Args:
        dataset (Sequence[Any]): The input dataset to wrap.
    """
    def __init__(self, dataset: Sequence[Any]) -> None:
        super().__init__()
        self.dataset = dataset
        self.range = range(len(self))

    def is_shardable(self) -> bool:
        """Indicates whether the dataset can be sharded."""
        return True

    def apply_sharding(self, num_shards: int, shard_idx: int) -> None:
        """Applies sharding to the dataset."""
        self.range = range(shard_idx, len(self), num_shards)

    def __iter__(self) -> Iterator:
        for i in self.range:
            yield self.dataset[i]

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.dataset)


def functional_transform(name: str) -> Callable:
    """
    A decorator to wrap classes into functional transforms for DataPipes.

    Args:
        name (str): The name to register the functional transform.

    Returns:
        Callable: The wrapper function.
    """
    def wrapper(cls: Any) -> Any:
        @functional_datapipe(name)
        class DynamicMapper(DataPipe):
            """
            Dynamically maps a transformation function onto DataPipe elements.

            Args:
                dp (DataPipe): The input DataPipe.
                *args (Any): Arguments for the transformation function.
                **kwargs (Any): Keyword arguments for the transformation function.
            """
            def __init__(
                self,
                dp: DataPipe,
                *args: Any,
                **kwargs: Any,
            ) -> None:
                super().__init__()
                self.dp = dp
                self.fn = cls(*args, **kwargs)

            def __iter__(self) -> Iterator:
                for data in self.dp:
                    yield self.fn(copy.copy(data))

        return cls

    return wrapper
