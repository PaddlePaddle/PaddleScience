import os.path as osp
from typing import Optional
import paddle
from paddle.io import DataLoader
from paddle import Tensor
from paddle_geometric.data import Data
from paddle_geometric.typing import SparseTensor
from tqdm import tqdm
import paddle_geometric.io.fs as fs

class GraphSAINTSampler(DataLoader):
    r"""The GraphSAINT sampler base class from the `"GraphSAINT: Graph
    Sampling Based Inductive Learning Method"
    <https://arxiv.org/abs/1907.04931>`_ paper.
    """
    def __init__(self, data, batch_size: int, num_steps: int = 1,
                 sample_coverage: int = 0, save_dir: Optional[str] = None,
                 log: bool = True, **kwargs):

        assert data.edge_index is not None
        assert 'node_norm' not in data
        assert 'edge_norm' not in data

        self.num_steps = num_steps
        self._batch_size = batch_size
        self.sample_coverage = sample_coverage
        self.save_dir = save_dir
        self.log = log

        self.N = N = data.num_nodes
        self.E = data.num_edges

        self.adj = SparseTensor(
            row=data.edge_index[0], col=data.edge_index[1],
            value=paddle.arange(self.E, device=data.edge_index.device),
            sparse_sizes=(N, N))

        self.data = data

        super().__init__(self, batch_size=1, collate_fn=self._collate,
                         **kwargs)

        if self.sample_coverage > 0:
            path = osp.join(save_dir or '', self._filename)
            if save_dir is not None and osp.exists(path):
                self.node_norm, self.edge_norm = fs.torch_load(path)
            else:
                self.node_norm, self.edge_norm = self._compute_norm()
                if save_dir is not None:
                    paddle.save((self.node_norm, self.edge_norm), path)

    @property
    def _filename(self):
        return f'{self.__class__.__name__.lower()}_{self.sample_coverage}.pt'

    def __len__(self):
        return self.num_steps

    def _sample_nodes(self, batch_size):
        raise NotImplementedError

    def __getitem__(self, idx):
        node_idx = self._sample_nodes(self._batch_size).unique()
        adj, _ = self.adj.saint_subgraph(node_idx)
        return node_idx, adj

    def _collate(self, data_list):
        assert len(data_list) == 1
        node_idx, adj = data_list[0]

        data = self.data.__class__()
        data.num_nodes = node_idx.size(0)
        row, col, edge_idx = adj.coo()
        data.edge_index = paddle.stack([row, col], axis=0)

        for key, item in self.data.items():
            if key in ['edge_index', 'num_nodes']:
                continue
            if isinstance(item, paddle.Tensor) and item.size(0) == self.N:
                data[key] = item[node_idx]
            elif isinstance(item, paddle.Tensor) and item.size(0) == self.E:
                data[key] = item[edge_idx]
            else:
                data[key] = item

        if self.sample_coverage > 0:
            data.node_norm = self.node_norm[node_idx]
            data.edge_norm = self.edge_norm[edge_idx]

        return data

    def _compute_norm(self):
        node_count = paddle.zeros(self.N, dtype=paddle.float32)
        edge_count = paddle.zeros(self.E, dtype=paddle.float32)

        loader = DataLoader(self, batch_size=200,
                            collate_fn=lambda x: x,
                            num_workers=self.num_workers)

        if self.log:
            pbar = tqdm(total=self.N * self.sample_coverage)
            pbar.set_description('Compute GraphSAINT normalization')

        num_samples = total_sampled_nodes = 0
        while total_sampled_nodes < self.N * self.sample_coverage:
            for data in loader:
                for node_idx, adj in data:
                    edge_idx = adj.storage.value()
                    node_count[node_idx] += 1
                    edge_count[edge_idx] += 1
                    total_sampled_nodes += node_idx.size(0)

                    if self.log:
                        pbar.update(node_idx.size(0))
            num_samples += self.num_steps

        if self.log:
            pbar.close()

        row, _, edge_idx = self.adj.coo()
        t = paddle.empty_like(edge_count).scatter_(0, edge_idx, node_count[row])
        edge_norm = (t / edge_count).clip(min=0, max=1e4)
        edge_norm[paddle.isnan(edge_norm)] = 0.1

        node_count[node_count == 0] = 0.1
        node_norm = num_samples / node_count / self.N

        return node_norm, edge_norm


class GraphSAINTNodeSampler(GraphSAINTSampler):
    r"""The GraphSAINT node sampler class (see
    :class:`~torch_geometric.loader.GraphSAINTSampler`).
    """
    def _sample_nodes(self, batch_size):
        edge_sample = paddle.randint(0, self.E, (batch_size, self.batch_size),
                                     dtype=paddle.long)

        return self.adj.storage.row()[edge_sample]


class GraphSAINTEdgeSampler(GraphSAINTSampler):
    r"""The GraphSAINT edge sampler class (see
    :class:`~torch_geometric.loader.GraphSAINTSampler`).
    """
    def _sample_nodes(self, batch_size):
        row, col, _ = self.adj.coo()

        deg_in = 1. / self.adj.storage.colcount()
        deg_out = 1. / self.adj.storage.rowcount()
        prob = (1. / deg_in[row]) + (1. / deg_out[col])

        rand = paddle.rand((batch_size, self.E)).log() / (prob + 1e-10)
        edge_sample = rand.topk(self.batch_size, axis=-1).indices

        source_node_sample = col[edge_sample]
        target_node_sample = row[edge_sample]

        return paddle.concat([source_node_sample, target_node_sample], -1)


class GraphSAINTRandomWalkSampler(GraphSAINTSampler):
    r"""The GraphSAINT random walk sampler class (see
    :class:`~torch_geometric.loader.GraphSAINTSampler`).

    Args:
        walk_length (int): The length of each random walk.
    """
    def __init__(self, data, batch_size: int, walk_length: int,
                 num_steps: int = 1, sample_coverage: int = 0,
                 save_dir: Optional[str] = None, log: bool = True, **kwargs):
        self.walk_length = walk_length
        super().__init__(data, batch_size, num_steps, sample_coverage,
                         save_dir, log, **kwargs)

    @property
    def _filename(self):
        return (f'{self.__class__.__name__.lower()}_{self.walk_length}_'
                f'{self.sample_coverage}.pt')

    def _sample_nodes(self, batch_size):
        start = paddle.randint(0, self.N, (batch_size,), dtype=paddle.long)
        node_idx = self.adj.random_walk(start.flatten(), self.walk_length)
        return node_idx.view(-1)
