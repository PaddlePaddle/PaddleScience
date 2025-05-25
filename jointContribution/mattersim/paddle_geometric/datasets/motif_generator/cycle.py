import paddle

from paddle_geometric.data import Data
from paddle_geometric.datasets.motif_generator import CustomMotif


class CycleMotif(CustomMotif):
    r"""Generates the cycle motif from the `"GNNExplainer:
    Generating Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`__ paper.

    Args:
        num_nodes (int): The number of nodes in the cycle.
    """
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes

        row = paddle.arange(num_nodes).reshape([-1, 1]).tile([1, 2]).reshape([-1])
        col1 = paddle.arange(-1, num_nodes - 1) % num_nodes
        col2 = paddle.arange(1, num_nodes + 1) % num_nodes
        col = paddle.stack([col1, col2], axis=1).sort(axis=-1).reshape([-1])

        structure = Data(
            num_nodes=num_nodes,
            edge_index=paddle.stack([row, col], axis=0),
        )
        super().__init__(structure)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num_nodes})'
