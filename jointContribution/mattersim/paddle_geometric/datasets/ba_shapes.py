from typing import Callable, Optional, Tuple

import paddle
from paddle import Tensor

from paddle_geometric.data import Data, InMemoryDataset
from paddle_geometric.utils import barabasi_albert_graph
from paddle_geometric.deprecation import deprecated


def house() -> Tuple[Tensor, Tensor]:
    edge_index = paddle.to_tensor([[0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4],
                                   [1, 3, 4, 4, 2, 0, 1, 3, 2, 0, 0, 1]])
    label = paddle.to_tensor([1, 1, 2, 2, 3])
    return edge_index, label

@deprecated("use 'datasets.ExplainerDataset' in combination with "
            "'datasets.graph_generator.BAGraph' instead")
class BAShapes(InMemoryDataset):
    r"""The BA-Shapes dataset from the `"GNNExplainer: Generating Explanations
    for Graph Neural Networks" <https://arxiv.org/abs/1903.03894>`__ paper,
    containing a Barabasi-Albert (BA) graph with 300 nodes and a set of 80
    "house"-structured graphs connected to it.

    Args:
        connection_distribution: Specifies how the houses and the BA graph get
            connected. Valid inputs are :obj:`"random"`
            (random BA graph nodes are selected for connection to the houses),
            and :obj:`"uniform"` (uniformly distributed BA graph nodes are
            selected for connection to the houses).
        transform: A function/transform that takes in a
            :class:`paddle_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
    """
    def __init__(
        self,
        connection_distribution: str = "random",
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(None, transform)
        assert connection_distribution in ['random', 'uniform']

        # Build the Barabasi-Albert graph:
        num_nodes = 300
        edge_index = barabasi_albert_graph(num_nodes, num_edges=5)
        edge_label = paddle.zeros([edge_index.shape[1]], dtype='int64')
        node_label = paddle.zeros([num_nodes], dtype='int64')

        # Select nodes to connect shapes:
        num_houses = 80
        if connection_distribution == 'random':
            connecting_nodes = paddle.randperm(num_nodes)[:num_houses]
        else:
            step = num_nodes // num_houses
            connecting_nodes = paddle.arange(0, num_nodes, step)

        # Connect houses to Barabasi-Albert graph:
        edge_indices = [edge_index]
        edge_labels = [edge_label]
        node_labels = [node_label]
        for i in range(num_houses):
            house_edge_index, house_label = house()

            edge_indices.append(house_edge_index + num_nodes)
            edge_indices.append(
                paddle.to_tensor([[int(connecting_nodes[i]), num_nodes],
                                  [num_nodes, int(connecting_nodes[i])]]))

            edge_labels.append(
                paddle.ones([house_edge_index.shape[1]], dtype='int64'))
            edge_labels.append(paddle.zeros([2], dtype='int64'))

            node_labels.append(house_label)

            num_nodes += 5

        edge_index = paddle.concat(edge_indices, axis=1)
        edge_label = paddle.concat(edge_labels, axis=0)
        node_label = paddle.concat(node_labels, axis=0)

        x = paddle.ones([num_nodes, 10], dtype='float32')
        expl_mask = paddle.zeros([num_nodes], dtype='bool')
        expl_mask[paddle.arange(400, num_nodes, 5)] = True

        data = Data(x=x, edge_index=edge_index, y=node_label,
                    expl_mask=expl_mask, edge_label=edge_label)

        self.data, self.slices = self.collate([data])
