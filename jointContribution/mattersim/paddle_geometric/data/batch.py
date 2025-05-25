import inspect
from collections.abc import Sequence
from typing import Any, List, Optional, Type, Union

import numpy as np
import paddle
from paddle import Tensor
from typing_extensions import Self

from paddle_geometric.data.collate import collate
from paddle_geometric.data.data import BaseData, Data
from paddle_geometric.data.dataset import IndexType
from paddle_geometric.data.separate import separate


class DynamicInheritance(type):
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        base_cls = kwargs.pop('_base_cls', Data)

        if issubclass(base_cls, Batch):
            new_cls = base_cls
        else:
            name = f'{base_cls.__name__}{cls.__name__}'

            class MetaResolver(type(cls), type(base_cls)):
                pass

            if name not in globals():
                globals()[name] = MetaResolver(name, (cls, base_cls), {})
            new_cls = globals()[name]

        params = list(inspect.signature(base_cls.__init__).parameters.items())
        for i, (k, v) in enumerate(params[1:]):
            if k == 'args' or k == 'kwargs':
                continue
            if i < len(args) or k in kwargs:
                continue
            if v.default is not inspect.Parameter.empty:
                continue
            kwargs[k] = None

        return super(DynamicInheritance, new_cls).__call__(*args, **kwargs)


class DynamicInheritanceGetter:
    def __call__(self, cls: Type, base_cls: Type) -> Self:
        return cls(_base_cls=base_cls)


class Batch(metaclass=DynamicInheritance):
    @classmethod
    def from_data_list(
        cls,
        data_list: List[BaseData],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ) -> Self:
        batch, slice_dict, inc_dict = collate(
            cls,
            data_list=data_list,
            increment=True,
            add_batch=not isinstance(data_list[0], Batch),
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

        batch._num_graphs = len(data_list)
        batch._slice_dict = slice_dict
        batch._inc_dict = inc_dict

        return batch

    def get_example(self, idx: int) -> BaseData:
        if not hasattr(self, '_slice_dict'):
            raise RuntimeError(
                "Cannot reconstruct 'Data' object from 'Batch' because "
                "'Batch' was not created via 'Batch.from_data_list()'")

        data = separate(
            cls=self.__class__.__bases__[-1],
            batch=self,
            idx=idx,
            slice_dict=getattr(self, '_slice_dict'),
            inc_dict=getattr(self, '_inc_dict'),
            decrement=True,
        )

        return data

    def index_select(self, idx: IndexType) -> List[BaseData]:
        index: Sequence[int]
        if isinstance(idx, slice):
            index = list(range(self.num_graphs)[idx])

        elif isinstance(idx, Tensor) and idx.dtype == paddle.int64:
            index = idx.flatten().tolist()

        elif isinstance(idx, Tensor) and idx.dtype == paddle.bool:
            index = idx.flatten().nonzero(as_tuple=False).flatten().tolist()

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            index = idx.flatten().tolist()

        elif isinstance(idx, np.ndarray) and idx.dtype == bool:
            index = idx.flatten().nonzero()[0].flatten().tolist()

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            index = idx

        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, paddle.Tensor and "
                f"np.ndarray of dtype int64 or bool are valid indices (got "
                f"'{type(idx).__name__}')")

        return [self.get_example(i) for i in index]

    def __getitem__(self, idx: Union[int, np.integer, str, IndexType]) -> Any:
        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, Tensor) and idx.ndim == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):
            return self.get_example(idx)
        elif isinstance(idx, str) or (isinstance(idx, tuple)
                                      and isinstance(idx[0], str)):
            return super().__getitem__(idx)
        else:
            return self.index_select(idx)

    def to_data_list(self) -> List[BaseData]:
        return [self.get_example(i) for i in range(self.num_graphs)]

    @property
    def num_graphs(self) -> int:
        if hasattr(self, '_num_graphs'):
            return self._num_graphs
        elif hasattr(self, 'ptr'):
            return self.ptr.shape[0] - 1
        elif hasattr(self, 'batch'):
            return int(self.batch.max()) + 1
        else:
            raise ValueError("Cannot infer the number of graphs")

    @property
    def batch_size(self) -> int:
        return self.num_graphs

    def __len__(self) -> int:
        return self.num_graphs

    def __reduce__(self) -> Any:
        state = self.__dict__.copy()
        return DynamicInheritanceGetter(), self.__class__.__bases__, state
