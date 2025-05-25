import copy
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import paddle

from paddle_geometric.data.data import BaseData, size_repr
from paddle_geometric.data.storage import (
    BaseStorage,
    EdgeStorage,
    GlobalStorage,
    NodeStorage,
)


class TemporalData(BaseData):
    r"""A data object composed of a stream of events describing a temporal
    graph. The :class:`~paddle_geometric.data.TemporalData` object can hold
    a list of events (that can be understood as temporal edges in a graph)
    with structured messages.
    An event is composed of a source node, a destination node, a timestamp,
    and a message. Any *Continuous-Time Dynamic Graph* (CTDG) can be
    represented with these four values.

    This object mimics the behavior of a regular Python dictionary while
    providing PyTorch/Paddle tensor functionalities and utilities.

    Args:
        src (paddle.Tensor, optional): A list of source nodes for the events
            with shape :obj:`[num_events]`. (default: :obj:`None`)
        dst (paddle.Tensor, optional): A list of destination nodes for the
            events with shape :obj:`[num_events]`. (default: :obj:`None`)
        t (paddle.Tensor, optional): The timestamps for each event with shape
            :obj:`[num_events]`. (default: :obj:`None`)
        msg (paddle.Tensor, optional): Messages feature matrix with shape
            :obj:`[num_events, num_msg_features]`. (default: :obj:`None`)
        **kwargs (optional): Additional attributes.

    .. note::
        The shape of :obj:`src`, :obj:`dst`, :obj:`t` and the first dimension
        of :obj:`msg` should be the same (:obj:`num_events`).
    """
    def __init__(
        self,
        src: Optional[paddle.Tensor] = None,
        dst: Optional[paddle.Tensor] = None,
        t: Optional[paddle.Tensor] = None,
        msg: Optional[paddle.Tensor] = None,
        **kwargs,
    ):
        super().__init__()
        self.__dict__['_store'] = GlobalStorage(_parent=self)

        self.src = src
        self.dst = dst
        self.t = t
        self.msg = msg

        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, mapping: Dict[str, Any]) -> 'TemporalData':
        """Creates a :class:`~paddle_geometric.data.TemporalData` object from
        a Python dictionary."""
        return cls(**mapping)

    def index_select(self, idx: Any) -> 'TemporalData':
        idx = prepare_idx(idx)
        data = copy.copy(self)
        for key, value in data._store.items():
            if value.shape[0] == self.num_events:
                data[key] = value[idx]
        return data

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, str):
            return self._store[idx]
        return self.index_select(idx)

    def __setitem__(self, key: str, value: Any):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        self._store[key] = value

    def __delitem__(self, key: str):
        if key in self._store:
            del self._store[key]

    def __getattr__(self, key: str) -> Any:
        if '_store' not in self.__dict__:
            raise RuntimeError(
                "The 'data' object was created by an older version. If this "
                "error occurred while loading an existing dataset, remove the "
                "'processed/' directory in the dataset's root folder and try again."
            )
        return getattr(self._store, key)

    def __setattr__(self, key: str, value: Any):
        setattr(self._store, key, value)

    def __delattr__(self, key: str):
        delattr(self._store, key)

    def __iter__(self) -> Iterable:
        for i in range(self.num_events):
            yield self[i]

    def __len__(self) -> int:
        return self.num_events

    def __call__(self, *args: List[str]) -> Iterable:
        yield from self._store.items(*args)

    def __copy__(self):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out.__dict__['_store'] = copy.copy(self._store)
        out._store._parent = out
        return out

    def __deepcopy__(self, memo):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = copy.deepcopy(value, memo)
        out._store._parent = out
        return out

    def stores_as(self, data: 'TemporalData'):
        return self

    @property
    def stores(self) -> List[BaseStorage]:
        return [self._store]

    @property
    def node_stores(self) -> List[NodeStorage]:
        return [self._store]

    @property
    def edge_stores(self) -> List[EdgeStorage]:
        return [self._store]

    def to_dict(self) -> Dict[str, Any]:
        return self._store.to_dict()

    def to_namedtuple(self) -> NamedTuple:
        return self._store.to_namedtuple()

    @property
    def num_nodes(self) -> int:
        """Returns the number of nodes in the graph."""
        return max(int(self.src.max()), int(self.dst.max())) + 1

    @property
    def num_events(self) -> int:
        """Returns the number of events loaded."""
        return self.src.shape[0]

    @property
    def num_edges(self) -> int:
        """Alias for :meth:`~paddle_geometric.data.TemporalData.num_events`."""
        return self.num_events

    @property
    def edge_index(self) -> paddle.Tensor:
        """Returns the edge indices of the graph."""
        if 'edge_index' in self:
            return self._store['edge_index']
        if self.src is not None and self.dst is not None:
            return paddle.stack([self.src, self.dst], axis=0)
        raise ValueError(f"{self.__class__.__name__} does not contain "
                         f"'edge_index' information")

    def size(
        self, dim: Optional[int] = None
    ) -> Union[Tuple[Optional[int], Optional[int]], Optional[int]]:
        """Returns the size of the adjacency matrix induced by the graph."""
        size = (int(self.src.max()), int(self.dst.max()))
        return size if dim is None else size[dim]

    def train_val_test_split(self, val_ratio: float = 0.15,
                             test_ratio: float = 0.15):
        """Splits the data into training, validation, and test sets based on
        time."""
        val_time, test_time = np.quantile(
            self.t.numpy(),
            [1. - val_ratio - test_ratio, 1. - test_ratio])

        val_idx = int((self.t <= val_time).sum().item())
        test_idx = int((self.t <= test_time).sum().item())

        return self[:val_idx], self[val_idx:test_idx], self[test_idx:]

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = ', '.join([size_repr(k, v) for k, v in self._store.items()])
        return f'{cls}({info})'


###############################################################################


def prepare_idx(idx):
    if isinstance(idx, int):
        return slice(idx, idx + 1)
    if isinstance(idx, (list, tuple)):
        return paddle.to_tensor(idx)
    elif isinstance(idx, slice):
        return idx
    elif isinstance(idx, paddle.Tensor) and idx.dtype == paddle.int64:
        return idx
    elif isinstance(idx, paddle.Tensor) and idx.dtype == paddle.bool:
        return idx

    raise IndexError(
        f"Only strings, integers, slices (`:`), list, tuples, and long or "
        f"bool tensors are valid indices (got '{type(idx).__name__}')")
