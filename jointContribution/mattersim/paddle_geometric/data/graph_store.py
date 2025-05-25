r"""This class defines the abstraction for a backend-agnostic graph store. The
goal of the graph store is to abstract away all graph edge index memory
management so that varying implementations can allow for independent scale-out.

This particular graph store abstraction makes a few key assumptions:
* The edge indices we care about storing are represented either in COO, CSC,
  or CSR format. They can be uniquely identified by an edge type (in PyG,
  this is a tuple of the source node, relation type, and destination node).
* Edge indices are static once they are stored in the graph. That is, we do not
  support dynamic modification of edge indices once they have been inserted
  into the graph store.

It is the job of a graph store implementor class to handle these assumptions
properly. For example, a simple in-memory graph store implementation may
concatenate all metadata values with an edge index and use this as a unique
index in a KV store. More complicated implementations may choose to partition
the graph in interesting manners based on the provided metadata.
"""

import copy
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from paddle import Tensor
from paddle_geometric.index import index2ptr, ptr2index
from paddle_geometric.typing import EdgeTensorType, EdgeType, OptTensor
from paddle_geometric.utils import index_sort
from paddle_geometric.utils.mixin import CastMixin

ConversionOutputType = Tuple[Dict[EdgeType, Tensor], Dict[EdgeType, Tensor], Dict[EdgeType, OptTensor]]


class EdgeLayout(Enum):
    COO = 'coo'
    CSC = 'csc'
    CSR = 'csr'


@dataclass
class EdgeAttr(CastMixin):
    r"""Defines the attributes of a :obj:`GraphStore` edge.
    It holds all the parameters necessary to uniquely identify an edge from
    the :class:`GraphStore`.

    Note that the order of the attributes is important; this is the order in
    which attributes must be provided for indexing calls. :class:`GraphStore`
    implementations can define a different ordering by overriding
    :meth:`EdgeAttr.__init__`.
    """

    edge_type: EdgeType
    layout: EdgeLayout
    is_sorted: bool = False
    size: Optional[Tuple[int, int]] = None

    def __init__(self, edge_type: EdgeType, layout: EdgeLayout, is_sorted: bool = False, size: Optional[Tuple[int, int]] = None):
        layout = EdgeLayout(layout)

        if layout == EdgeLayout.CSR and is_sorted:
            raise ValueError("Cannot create a 'CSR' edge attribute with option 'is_sorted=True'")

        if layout == EdgeLayout.CSC:
            is_sorted = True

        self.edge_type = edge_type
        self.layout = layout
        self.is_sorted = is_sorted
        self.size = size


class GraphStore(ABC):
    r"""An abstract base class to access edges from a remote graph store.

    Args:
        edge_attr_cls (EdgeAttr, optional): A user-defined
            :class:`EdgeAttr` class to customize the required attributes and
            their ordering to uniquely identify edges. (default: :obj:`None`)
    """
    def __init__(self, edge_attr_cls: Optional[Any] = None):
        super().__init__()
        self.__dict__['_edge_attr_cls'] = edge_attr_cls or EdgeAttr

    @abstractmethod
    def _put_edge_index(self, edge_index: EdgeTensorType, edge_attr: EdgeAttr) -> bool:
        r"""To be implemented by :class:`GraphStore` subclasses."""

    def put_edge_index(self, edge_index: EdgeTensorType, *args, **kwargs) -> bool:
        edge_attr = self._edge_attr_cls.cast(*args, **kwargs)
        return self._put_edge_index(edge_index, edge_attr)

    @abstractmethod
    def _get_edge_index(self, edge_attr: EdgeAttr) -> Optional[EdgeTensorType]:
        r"""To be implemented by :class:`GraphStore` subclasses."""

    def get_edge_index(self, *args, **kwargs) -> EdgeTensorType:
        edge_attr = self._edge_attr_cls.cast(*args, **kwargs)
        edge_index = self._get_edge_index(edge_attr)
        if edge_index is None:
            raise KeyError(f"'edge_index' for '{edge_attr}' not found")
        return edge_index

    @abstractmethod
    def _remove_edge_index(self, edge_attr: EdgeAttr) -> bool:
        r"""To be implemented by :class:`GraphStore` subclasses."""

    def remove_edge_index(self, *args, **kwargs) -> bool:
        edge_attr = self._edge_attr_cls.cast(*args, **kwargs)
        return self._remove_edge_index(edge_attr)

    @abstractmethod
    def get_all_edge_attrs(self) -> List[EdgeAttr]:
        r"""Returns all registered edge attributes."""

    def coo(self, edge_types: Optional[List[Any]] = None, store: bool = False) -> ConversionOutputType:
        return self._edges_to_layout(EdgeLayout.COO, edge_types, store)

    def csr(self, edge_types: Optional[List[Any]] = None, store: bool = False) -> ConversionOutputType:
        return self._edges_to_layout(EdgeLayout.CSR, edge_types, store)

    def csc(self, edge_types: Optional[List[Any]] = None, store: bool = False) -> ConversionOutputType:
        return self._edges_to_layout(EdgeLayout.CSC, edge_types, store)

    def __setitem__(self, key: EdgeAttr, value: EdgeTensorType):
        self.put_edge_index(value, key)

    def __getitem__(self, key: EdgeAttr) -> Optional[EdgeTensorType]:
        return self.get_edge_index(key)

    def __delitem__(self, key: EdgeAttr):
        return self.remove_edge_index(key)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def _edge_to_layout(self, attr: EdgeAttr, layout: EdgeLayout, store: bool = False) -> Tuple[Tensor, Tensor, OptTensor]:
        (row, col), perm = self.get_edge_index(attr), None

        if layout == EdgeLayout.COO:
            if attr.layout == EdgeLayout.CSR:
                row = ptr2index(row)
            elif attr.layout == EdgeLayout.CSC:
                col = ptr2index(col)

        elif layout == EdgeLayout.CSR:
            if attr.layout == EdgeLayout.CSC:
                col = ptr2index(col)

            if attr.layout != EdgeLayout.CSR:
                num_rows = attr.size[0] if attr.size else int(row.max()) + 1
                row, perm = index_sort(row, max_value=num_rows)
                col = col[perm]
                row = index2ptr(row, num_rows)

        else:
            if attr.layout == EdgeLayout.CSR:
                row = ptr2index(row)

            if attr.layout != EdgeLayout.CSC:
                if hasattr(self, 'meta') and self.meta.get('is_hetero', False):
                    num_cols = int(col.max()) + 1
                elif attr.size is not None:
                    num_cols = attr.size[1]
                else:
                    num_cols = int(col.max()) + 1

                if not attr.is_sorted:
                    col, perm = index_sort(col, max_value=num_cols)
                    row = row[perm]
                col = index2ptr(col, num_cols)

        if attr.layout != layout and store:
            attr = copy.copy(attr)
            attr.layout = layout
            if perm is not None:
                attr.is_sorted = False
            self.put_edge_index((row, col), attr)

        return row, col, perm

    def _edges_to_layout(self, layout: EdgeLayout, edge_types: Optional[List[Any]] = None, store: bool = False) -> ConversionOutputType:
        edge_attrs: List[EdgeAttr] = self.get_all_edge_attrs()

        if hasattr(self, 'meta'):
            is_hetero = self.meta.get('is_hetero', False)
        else:
            is_hetero = all(attr.edge_type is not None for attr in edge_attrs)

        if not is_hetero:
            return self._edge_to_layout(edge_attrs[0], layout, store)

        edge_type_attrs: Dict[EdgeType, List[EdgeAttr]] = defaultdict(list)
        for attr in self.get_all_edge_attrs():
            edge_type_attrs[attr.edge_type].append(attr)

        if edge_types is not None:
            for edge_type in edge_types:
                if edge_type not in edge_type_attrs:
                    raise ValueError(f"The 'edge_index' of type '{edge_type}' was not found in the graph store.")

            edge_type_attrs = {key: attr for key, attr in edge_type_attrs.items() if key in edge_types}

        row_dict, col_dict, perm_dict = {}, {}, {}
        for edge_type, attrs in edge_type_attrs.items():
            layouts = [attr.layout for attr in attrs]

            if layout in layouts:
                attr = attrs[layouts.index(layout)]
            elif EdgeLayout.COO in layouts:
                attr = attrs[layouts.index(EdgeLayout.COO)]
            elif EdgeLayout.CSC in layouts:
                attr = attrs[layouts.index(EdgeLayout.CSC)]
            elif EdgeLayout.CSR in layouts:
                attr = attrs[layouts.index(EdgeLayout.CSR)]

            row_dict[edge_type], col_dict[edge_type], perm_dict[edge_type] = self._edge_to_layout(attr, layout, store)

        return row_dict, col_dict, perm_dict
