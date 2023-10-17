from paddle import Tensor
from typing import Optional, Tuple, Union
import paddle

from scatter import scatter, scatter_add
OptTensor = Optional[Tensor]

import pgl

class Data():
    def __init__(self, x, y, edge_index):
        self.y = y
        self.x = x
        self.edge_index = edge_index
    def __call__(self):
        return pgl.Graph(edges=self.edge_index,
                         num_nodes=self.x.shape[0],
                         node_feat=self.x)
    
def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))

def remove_self_loops(edge_index: Tensor,
                      edge_attr: OptTensor = None) -> Tuple[Tensor, OptTensor]:
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    mask = edge_index[0] != edge_index[1]
    for _ in range(edge_index.dim()):
      edge_index[_] = paddle.masked_select(edge_index[_], mask)
    if edge_attr is None:
        return edge_index, None
    else:
        return edge_index, edge_attr[mask]

def add_self_loops(
        edge_index: Tensor, edge_attr: OptTensor = None,
        fill_value: Union[float, Tensor, str] = None,
        num_nodes: Optional[int] = None) -> Tuple[Tensor, OptTensor]:
    r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted or has multi-dimensional edge features
    (:obj:`edge_attr != None`), edge features of self-loops will be added
    according to :obj:`fill_value`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_attr != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`1.`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    N = maybe_num_nodes(edge_index, num_nodes)

    loop_index = paddle.arange(0, N, dtype=paddle.int64)
    loop_index = paddle.tile(loop_index.unsqueeze(0), repeat_times=[2, 1])
    if edge_attr is not None:
        if fill_value is None:
            loop_attr = edge_attr.new_full((N, ) + edge_attr.size()[1:], 1.)

        elif isinstance(fill_value, (int, float)):
            loop_attr = paddle.full((N, ), fill_value, dtype=edge_attr.dtype)
        elif isinstance(fill_value, Tensor):
            loop_attr = fill_value.to(edge_attr.device, edge_attr.dtype)
            if edge_attr.dim() != loop_attr.dim():
                loop_attr = loop_attr.unsqueeze(0)
            sizes = [N] + [1] * (loop_attr.dim() - 1)
            loop_attr = loop_attr.repeat(*sizes)

        elif isinstance(fill_value, str):
            loop_attr = scatter(edge_attr, edge_index[1], dim=0, dim_size=N,
                                reduce=fill_value)
        else:
            raise AttributeError("No valid 'fill_value' provided")
         
        edge_attr = paddle.concat([edge_attr, loop_attr], axis=0)

    edge_index = paddle.concat([edge_index, loop_index], axis=1)
    return edge_index, edge_attr
    
def get_laplacian(edge_index, edge_weight: Optional[paddle.Tensor] = None,
                  normalization: Optional[str] = None,
                  dtype: Optional[int] = None,
                  num_nodes: Optional[int] = None):
    r""" Computes the graph Laplacian of the graph given by :obj:`edge_index`
    and optional :obj:`edge_weight`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`None`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`
        dtype (torch.dtype, optional): The desired data type of returned tensor
            in case :obj:`edge_weight=None`. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    """

    if normalization is not None:
        assert normalization in ['sym', 'rw']  # 'Invalid normalization'

    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = paddle.ones(edge_index.shape[1], dtype=dtype)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    if normalization is None:
        # L = D - A.
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_weight = paddle.concat([-edge_weight, deg], dim=0)
    elif normalization == 'sym':
        # Compute A_norm = -D^{-1/2} A D^{-1/2}.
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt = masked_fill(deg_inv_sqrt,deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        # L = I - A_norm.
        edge_index, tmp = add_self_loops(edge_index, -edge_weight,
                                         fill_value=1., num_nodes=num_nodes)
        assert tmp is not None
        edge_weight = tmp
    else:
        # Compute A_norm = -D^{-1} A.
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight

        # L = I - A_norm.
        edge_index, tmp = add_self_loops(edge_index, -edge_weight,
                                         fill_value=1., num_nodes=num_nodes)
        assert tmp is not None
        edge_weight = tmp

    return edge_index, edge_weight

def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)

def add_remaining_self_loops(
        edge_index: Tensor, edge_attr: OptTensor = None,
        fill_value: Union[float, Tensor, str] = None,
        num_nodes: Optional[int] = None) -> Tuple[Tensor, OptTensor]:
    r"""Adds remaining self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted or has multi-dimensional edge features
    (:obj:`edge_attr != None`), edge features of non-existing self-loops will
    be added according to :obj:`fill_value`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_attr != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`1.`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    N = maybe_num_nodes(edge_index, num_nodes)
    mask = edge_index[0] != edge_index[1]

    loop_index = paddle.arange(0, N, dtype=paddle.int32)
    loop_index = paddle.tile(loop_index.unsqueeze(0), repeat_times=[2, 1])

    if edge_attr is not None:
        if fill_value is None:
            loop_attr = edge_attr.new_full((N, ) + edge_attr.size()[1:], 1.)

        elif isinstance(fill_value, (int, float)):
            loop_attr = paddle.full((N, ), fill_value, dtype=edge_attr.dtype)
        elif isinstance(fill_value, Tensor):
            loop_attr = fill_value.to(edge_attr.device, edge_attr.dtype)
            if edge_attr.dim() != loop_attr.dim():
                loop_attr = loop_attr.unsqueeze(0)
            sizes = [N] + [1] * (loop_attr.dim() - 1)
            loop_attr = loop_attr.repeat(*sizes)

        elif isinstance(fill_value, str):
            loop_attr = scatter(edge_attr, edge_index[1], dim=0, dim_size=N,
                                reduce=fill_value)
        else:
            raise AttributeError("No valid 'fill_value' provided")

        inv_mask = ~mask
        # loop_attr[edge_index[0][inv_mask]] = edge_attr[inv_mask]

        edge_attr = paddle.concat([edge_attr, loop_attr], axis=0)
    # for _ in range(edge_index.dim()):
    #   edge_index[_] = paddle.masked_select(edge_index[_], mask)
    edge_index = paddle.concat([edge_index, loop_index], axis=1)
    return edge_index, edge_attr