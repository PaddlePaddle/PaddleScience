from typing import Any, Callable, Dict, Optional, Union

import paddle
from paddle import Tensor

from paddle_geometric.data import InMemoryDataset
from paddle_geometric.datasets.graph_generator import GraphGenerator
from paddle_geometric.datasets.motif_generator import MotifGenerator
from paddle_geometric.explain import Explanation


class ExplainerDataset(InMemoryDataset):
    r"""Generates a synthetic dataset for evaluating explainabilty algorithms,
    adapted for Paddle Geometric.

    Args:
        graph_generator (GraphGenerator or str): The graph generator to be
            used, e.g.,
            :class:`paddle_geometric.datasets.graph_generator.BAGraph`
            (or any string that automatically resolves to it).
        motif_generator (MotifGenerator): The motif generator to be used,
            e.g.,
            :class:`paddle_geometric.datasets.motif_generator.HouseMotif`
            (or any string that automatically resolves to it).
        num_motifs (int): The number of motifs to attach to the graph.
        num_graphs (int, optional): The number of graphs to generate.
            (default: :obj:`1`)
        graph_generator_kwargs (Dict[str, Any], optional): Arguments passed to
            the respective graph generator module in case it gets automatically
            resolved. (default: :obj:`None`)
        motif_generator_kwargs (Dict[str, Any], optional): Arguments passed to
            the respective motif generator module in case it gets automatically
            resolved. (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`paddle_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        graph_generator: Union[GraphGenerator, str],
        motif_generator: Union[MotifGenerator, str],
        num_motifs: int,
        num_graphs: int = 1,
        graph_generator_kwargs: Optional[Dict[str, Any]] = None,
        motif_generator_kwargs: Optional[Dict[str, Any]] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__(root=None, transform=transform)

        if num_motifs <= 0:
            raise ValueError(f"At least one motif needs to be attached to the "
                             f"graph (got {num_motifs})")

        self.graph_generator = GraphGenerator.resolve(
            graph_generator,
            **(graph_generator_kwargs or {}),
        )
        self.motif_generator = MotifGenerator.resolve(
            motif_generator,
            **(motif_generator_kwargs or {}),
        )
        self.num_motifs = num_motifs

        # Generate synthetic graphs and collate them:
        data_list = [self.get_graph() for _ in range(num_graphs)]
        self.data, self.slices = self.collate(data_list)

    def get_graph(self) -> Explanation:
        data = self.graph_generator()
        assert data.num_nodes is not None
        assert data.edge_index is not None

        edge_indices = [data.edge_index]
        num_nodes = data.num_nodes
        node_masks = [paddle.zeros([data.num_nodes])]
        edge_masks = [paddle.zeros([data.num_edges])]
        ys = [paddle.zeros([num_nodes], dtype="int64")]

        connecting_nodes = paddle.randperm(num_nodes)[:self.num_motifs]
        for i in connecting_nodes.numpy().tolist():
            motif = self.motif_generator()
            assert motif.num_nodes is not None
            assert motif.edge_index is not None

            # Add motif to the graph:
            edge_indices.append(motif.edge_index + num_nodes)
            node_masks.append(paddle.ones([motif.num_nodes]))
            edge_masks.append(paddle.ones([motif.num_edges]))

            # Add random motif connection to the graph:
            j = int(paddle.randint(0, motif.num_nodes, shape=[1])) + num_nodes
            edge_indices.append(paddle.to_tensor([[i, j], [j, i]], dtype="int64"))
            edge_masks.append(paddle.zeros([2]))

            if isinstance(motif.y, Tensor):
                ys.append(motif.y + 1 if motif.y.min() == 0 else motif.y)
            else:
                ys.append(paddle.ones([motif.num_nodes], dtype="int64"))

            num_nodes += motif.num_nodes

        return Explanation(
            edge_index=paddle.concat(edge_indices, axis=1),
            y=paddle.concat(ys, axis=0),
            edge_mask=paddle.concat(edge_masks, axis=0),
            node_mask=paddle.concat(node_masks, axis=0),
        )

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'graph_generator={self.graph_generator}, '
                f'motif_generator={self.motif_generator}, '
                f'num_motifs={self.num_motifs})')
