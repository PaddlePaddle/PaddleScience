from typing import List, Optional, Tuple

import paddle
from paddle import Tensor

from paddle_geometric.data import Data
from paddle_geometric.datasets.graph_generator import GraphGenerator
from paddle_geometric.utils import to_undirected


def tree(
    depth: int,
    branch: int = 2,
    undirected: bool = False,
    device: Optional[str] = None,
) -> Tuple[Tensor, Tensor]:
    """Generates a tree graph with the given depth and branch size, along with
    node-level depth indicators.

    Args:
        depth (int): The depth of the tree.
        branch (int, optional): The branch size of the tree.
            (default: :obj:`2`)
        undirected (bool, optional): If set to :obj:`True`, the tree graph will
            be undirected. (default: :obj:`False`)
        device (paddle.device, optional): The desired device of the returned
            tensors. (default: :obj:`None`)
    """
    edges: List[Tuple[int, int]] = []
    depths: List[int] = [0]

    def add_edges(node: int, current_depth: int) -> None:
        node_count = len(depths)

        if current_depth < depth:
            for i in range(branch):
                edges.append((node, node_count + i))
                depths.append(current_depth + 1)

            for i in range(branch):
                add_edges(node=node_count + i, current_depth=current_depth + 1)

    add_edges(node=0, current_depth=0)

    edge_index = paddle.to_tensor(edges, place=device).t().astype('int64')
    if undirected:
        edge_index = to_undirected(edge_index, num_nodes=len(depths))

    return edge_index, paddle.to_tensor(depths, place=device).astype('int64')


class TreeGraph(GraphGenerator):
    r"""Generates tree graphs.

    Args:
        depth (int): The depth of the tree.
        branch (int, optional): The branch size of the tree.
            (default: :obj:`2`)
        undirected (bool, optional): If set to :obj:`True`, the tree graph will
            be undirected. (default: :obj:`False`)
    """
    def __init__(
        self,
        depth: int,
        branch: int = 2,
        undirected: bool = False,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.branch = branch
        self.undirected = undirected

    def __call__(self) -> Data:
        edge_index, depth = tree(self.depth, self.branch, self.undirected)
        num_nodes = depth.shape[0]
        return Data(edge_index=edge_index, depth=depth, num_nodes=num_nodes)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(depth={self.depth}, '
                f'branch={self.branch}, undirected={self.undirected})')
