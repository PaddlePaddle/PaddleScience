from typing import Optional, Union

import paddle
from paddle import sparse as paddle_sparse
from paddle import Tensor

import paddle_geometric.typing
from paddle_geometric.data import Data, HeteroData
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.transforms import BaseTransform
from paddle_geometric.typing import SparseTensor
from paddle_geometric.utils import (
    sort_edge_index,
    to_paddle_coo_tensor,
    to_paddle_csr_tensor,
)


@functional_transform('to_sparse_tensor')
class ToSparseTensor(BaseTransform):
    r"""Converts the :obj:`edge_index` attributes of a homogeneous or
    heterogeneous data object into a **transposed**
    sparse tensor with key :obj:`adj_t`
    (functional name: :obj:`to_sparse_tensor`).

    Args:
        attr (str, optional): The name of the attribute to add as a value to
            the sparse tensor (if present).
            (default: :obj:`edge_weight`)
        remove_edge_index (bool, optional): If set to :obj:`False`, the
            :obj:`edge_index` tensor will not be removed.
            (default: :obj:`True`)
        fill_cache (bool, optional): If set to :obj:`True`, will fill the
            underlying sparse tensor cache (if used).
            (default: :obj:`True`)
        layout (optional): Specifies the layout of the returned
            sparse tensor (:obj:`None`, :obj:`paddle.sparse_coo` or
            :obj:`paddle.sparse_csr`).
    """
    def __init__(
        self,
        attr: Optional[str] = 'edge_weight',
        remove_edge_index: bool = True,
        fill_cache: bool = True,
        layout: Optional[int] = None,
    ) -> None:
        if layout not in {None, paddle_sparse.sparse_coo, paddle_sparse.sparse_csr}:
            raise ValueError(f"Unexpected sparse tensor layout (got '{layout}')")

        self.attr = attr
        self.remove_edge_index = remove_edge_index
        self.fill_cache = fill_cache
        self.layout = layout

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:

        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue

            keys, values = [], []
            for key, value in store.items():
                if key in {'edge_index', 'edge_label', 'edge_label_index'}:
                    continue

                if store.is_edge_attr(key):
                    keys.append(key)
                    values.append(value)

            store.edge_index, values = sort_edge_index(
                store.edge_index,
                values,
                sort_by_row=False,
            )

            for key, value in zip(keys, values):
                store[key] = value

            layout = self.layout
            size = store.size()[::-1]
            edge_weight: Optional[Tensor] = None
            if self.attr is not None and self.attr in store:
                edge_weight = store[self.attr]

            if layout == paddle_sparse.sparse_coo or (layout is None and not hasattr(paddle_geometric.typing, "WITH_PADDLE_SPARSE")):
                store.adj_t = to_paddle_coo_tensor(
                    store.edge_index.flip([0]),
                    edge_attr=edge_weight,
                    size=size,
                )

            elif layout == paddle_sparse.sparse_csr or layout is None:
                store.adj_t = to_paddle_csr_tensor(
                    store.edge_index.flip([0]),
                    edge_attr=edge_weight,
                    size=size,
                )

            if self.remove_edge_index:
                del store['edge_index']
                if self.attr is not None and self.attr in store:
                    del store[self.attr]

            if self.fill_cache:
                # Any caching steps if applicable for the backend
                pass

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(attr={self.attr}, layout={self.layout})')
