from typing import Any, Callable
import paddle
from paddle.io import DataLoader, Dataset


class DataLoaderIterator:
    r"""A data loader iterator extended by a simple post-transformation
    function :meth:`transform_fn`. While the iterator may request items from
    different sub-processes, :meth:`transform_fn` will always be executed in
    the main process.

    This iterator is used in PyG's sampler classes, and is responsible for
    feature fetching and filtering data objects after sampling has taken place
    in a sub-process. This has the following advantages:

    * We do not need to share feature matrices across processes which may
      prevent any errors due to too many open file handles.
    * We can execute any expensive post-processing commands on the main thread
      with full parallelization power (which usually executes faster).
    * It lets us naturally support data already being present on the GPU.
    """

    def __init__(self, loader: DataLoader, transform_fn: Callable):
        # In Paddle, we directly pass DataLoader and its iterable
        self.loader = loader
        self.transform_fn = transform_fn
        self.iterator = iter(self.loader)  # Create an iterator from DataLoader

    def __iter__(self) -> 'DataLoaderIterator':
        return self

    def _reset(self, loader: Any, first_iter: bool = False):
        # In Paddle, reset is handled by DataLoader itself, so we don't need to manually reset it
        self.iterator = iter(loader)

    def __len__(self) -> int:
        return len(self.loader)

    def __next__(self) -> Any:
        # Apply transformation to each batch loaded by the iterator
        data = next(self.iterator)
        return self.transform_fn(data)

    def __del__(self) -> None:
        # Clean up if necessary, although Paddle handles it internally
        del self.iterator
