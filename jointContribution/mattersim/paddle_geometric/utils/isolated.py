from typing import Optional, Tuple

import paddle
from paddle import Tensor

from paddle_geometric.utils import remove_self_loops, segregate_self_loops
from paddle_geometric.utils.num_nodes import maybe_num_nodes


def contains_isolated_nodes(
    edge_index: Tensor,
    num_nodes: Optional[int] = None,
) -> bool:
    r"""Returns :obj:`True` if the graph given by :attr:`edge_index` contains
    isolated nodes.

    Args:
        edge_index (Tensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: bool

    Examples:
        # >>> edge_index = paddle.to_tensor([[0, 1, 0],
        # ...                                [1, 0, 0]])
        # >>> contains_isolated_nodes(edge_index)
        False

        # >>> contains_isolated_nodes(edge_index, num_nodes=3)
        True
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    return paddle.unique(edge_index.flatten()).shape[0] < num_nodes


def remove_isolated_nodes(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor], Tensor]:
    r"""Removes the isolated nodes from the graph given by :attr:`edge_index`
    with optional edge attributes :attr:`edge_attr`.
    In addition, returns a mask of shape :obj:`[num_nodes]` to manually filter
    out isolated node features later on.
    Self-loops are preserved for non-isolated nodes.

    Args:
        edge_index (Tensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (Tensor, Tensor, Tensor)

    Examples:
        # >>> edge_index = paddle.to_tensor([[0, 1, 0],
        # ...                                [1, 0, 0]])
        # >>> edge_index, edge_attr, mask = remove_isolated_nodes(edge_index)
        # >>> mask # node mask (2 nodes)
        tensor([True, True])

        # >>> edge_index, edge_attr, mask = remove_isolated_nodes(edge_index,
        # ...                                                     num_nodes=3)
        # >>> mask # node mask (3 nodes)
        tensor([True, True, False])
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    out = segregate_self_loops(edge_index, edge_attr)
    edge_index, edge_attr, loop_edge_index, loop_edge_attr = out

    # Create a mask tensor filled with False (paddle.zeros is used)
    mask = paddle.zeros([num_nodes], dtype=paddle.bool, place=edge_index.place)
    mask[edge_index.flatten()] = 1

    # Create the assoc tensor initialized to -1
    assoc = paddle.full([num_nodes], -1, dtype=paddle.int64)
    assoc[mask] = paddle.arange(mask.sum().item(), dtype=paddle.int64)

    # Modify edge_index with assoc
    edge_index = assoc[edge_index]

    # Create loop_mask with the same shape as mask
    loop_mask = paddle.zeros_like(mask)
    loop_mask[loop_edge_index[0]] = 1
    loop_mask = loop_mask & mask

    # Create loop_assoc initialized to -1
    loop_assoc = paddle.full_like(assoc, -1, dtype=paddle.int64)
    loop_assoc[loop_edge_index[0]] = paddle.arange(loop_edge_index.shape[1], dtype=paddle.int64)

    # Get loop_idx
    loop_idx = loop_assoc[loop_mask]

    # Update loop_edge_index
    loop_edge_index = assoc[loop_edge_index[:, loop_idx]]

    # Concatenate edge_index and loop_edge_index along dim=1
    edge_index = paddle.concat([edge_index, loop_edge_index], axis=1)

    if edge_attr is not None:
        assert loop_edge_attr is not None
        loop_edge_attr = paddle.index_select(loop_edge_attr, loop_idx)
        edge_attr = paddle.concat([edge_attr, loop_edge_attr], axis=0)

    return edge_index, edge_attr, mask
