from typing import Any, Callable, Dict, List, Optional, Union

import paddle

from paddle_geometric.data import InMemoryDataset
from paddle_geometric.datasets.graph_generator import GraphGenerator
from paddle_geometric.explain import Explanation
from paddle_geometric.utils import k_hop_subgraph


class InfectionDataset(InMemoryDataset):
    r"""Generates a synthetic infection dataset for evaluating explainability
    algorithms, as described in the `"Explainability Techniques for Graph
    Convolutional Networks" <https://arxiv.org/abs/1905.13686>`__ paper.
    """

    def __init__(
        self,
        graph_generator: Union[GraphGenerator, str],
        num_infected_nodes: Union[int, List[int]],
        max_path_length: Union[int, List[int]],
        num_graphs: Optional[int] = None,
        graph_generator_kwargs: Optional[Dict[str, Any]] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__(root=None, transform=transform)

        assert isinstance(num_infected_nodes, (int, list))
        assert isinstance(max_path_length, (int, list))

        if (num_graphs is None and isinstance(num_infected_nodes, int)
                and isinstance(max_path_length, int)):
            num_graphs = 1

        if num_graphs is None and isinstance(num_infected_nodes, list):
            num_graphs = len(num_infected_nodes)

        if num_graphs is None and isinstance(max_path_length, list):
            num_graphs = len(max_path_length)

        assert num_graphs is not None

        self.graph_generator = GraphGenerator.resolve(
            graph_generator,
            **(graph_generator_kwargs or {}),
        )
        self.num_infected_nodes = num_infected_nodes
        self.max_path_length = max_path_length
        self.num_graphs = num_graphs

        if isinstance(num_infected_nodes, int):
            num_infected_nodes = [num_infected_nodes] * num_graphs

        if isinstance(max_path_length, int):
            max_path_length = [max_path_length] * num_graphs

        if len(num_infected_nodes) != num_graphs:
            raise ValueError(f"The length of 'num_infected_nodes' "
                             f"(got {len(num_infected_nodes)}) does not match "
                             f"the number of graphs (got {num_graphs})")

        if len(max_path_length) != num_graphs:
            raise ValueError(f"The length of 'max_path_length' "
                             f"(got {len(max_path_length)}) does not match "
                             f"the number of graphs (got {num_graphs})")

        if any(n <= 0 for n in num_infected_nodes):
            raise ValueError(f"'num_infected_nodes' must be positive "
                             f"(got {min(num_infected_nodes)})")

        if any(l <= 0 for l in max_path_length):
            raise ValueError(f"'max_path_length' must be positive "
                             f"(got {min(max_path_length)})")

        data_list: List[Explanation] = []
        for N, L in zip(num_infected_nodes, max_path_length):
            data_list.append(self.get_graph(N, L))

        self.data, self.slices = self.collate(data_list)

    def get_graph(self, num_infected_nodes: int,
                  max_path_length: int) -> Explanation:
        data = self.graph_generator()

        assert data.num_nodes is not None
        perm = paddle.randperm(data.num_nodes)
        x = paddle.zeros([data.num_nodes, 2])
        x[perm[:num_infected_nodes], 1] = 1  # Infected
        x[perm[num_infected_nodes:], 0] = 1  # Healthy

        y = paddle.full([data.num_nodes], fill_value=max_path_length + 1, dtype='int64')
        y[perm[:num_infected_nodes]] = 0  # Infected nodes have label `0`.

        assert data.edge_index is not None
        edge_mask = paddle.zeros([data.num_edges], dtype='bool')
        for num_hops in range(1, max_path_length + 1):
            sub_node_index, _, _, sub_edge_mask = k_hop_subgraph(
                perm[:num_infected_nodes], num_hops, data.edge_index,
                num_nodes=data.num_nodes, flow='target_to_source',
                directed=True)

            value = paddle.full_like(sub_node_index, fill_value=num_hops)
            y[sub_node_index] = paddle.minimum(y[sub_node_index], value)
            edge_mask |= sub_edge_mask

        return Explanation(
            x=x,
            edge_index=data.edge_index,
            y=y,
            edge_mask=edge_mask.astype('float32'),
        )

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'graph_generator={self.graph_generator}, '
                f'num_infected_nodes={self.num_infected_nodes}, '
                f'max_path_length={self.max_path_length})')
