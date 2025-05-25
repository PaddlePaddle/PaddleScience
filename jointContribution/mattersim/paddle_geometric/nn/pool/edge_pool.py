from typing import Callable, List, NamedTuple, Optional, Tuple


import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle_geometric.utils import coalesce, scatter, softmax


class UnpoolInfo(NamedTuple):
    edge_index: Tensor
    cluster: Tensor
    batch: Tensor
    new_edge_score: Tensor


class EdgePooling(paddle.nn.Layer):
    def __init__(
        self,
        in_channels: int,
        edge_score_method: Optional[Callable] = None,
        dropout: float = 0.0,
        add_to_edge_score: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        if edge_score_method is None:
            edge_score_method = self.compute_edge_score_softmax
        self.compute_edge_score = edge_score_method
        self.add_to_edge_score = add_to_edge_score
        self.dropout = dropout

        self.lin = paddle.nn.Linear(2 * in_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    @staticmethod
    def compute_edge_score_softmax(
        raw_edge_score: Tensor,
        edge_index: Tensor,
        num_nodes: int,
    ) -> Tensor:
        return softmax(raw_edge_score, edge_index[1], num_nodes=num_nodes)

    @staticmethod
    def compute_edge_score_tanh(
        raw_edge_score: Tensor,
        edge_index: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        return paddle.tanh(raw_edge_score)

    @staticmethod
    def compute_edge_score_sigmoid(
        raw_edge_score: Tensor,
        edge_index: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        return torch.sigmoid(raw_edge_score)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, UnpoolInfo]:
        e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        e = self.lin(e).view(-1)
        e = F.dropout(e, p=self.dropout, training=self.training)
        e = self.compute_edge_score(e, edge_index, x.size(0))
        e = e + self.add_to_edge_score

        x, edge_index, batch, unpool_info = self._merge_edges(
            x, edge_index, batch, e)

        return x, edge_index, batch, unpool_info

    def _merge_edges(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_score: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, UnpoolInfo]:
        cluster = torch.empty_like(batch)
        perm: List[int] = torch.argsort(edge_score, descending=True).tolist()

        mask = torch.ones(x.size(0), dtype=torch.bool)

        i = 0
        new_edge_indices: List[int] = []
        edge_index_cpu = edge_index.cpu()
        for edge_idx in perm:
            source = int(edge_index_cpu[0, edge_idx])
            if not bool(mask[source]):
                continue

            target = int(edge_index_cpu[1, edge_idx])
            if not bool(mask[target]):
                continue

            new_edge_indices.append(edge_idx)

            cluster[source] = i
            mask[source] = False

            if source != target:
                cluster[target] = i
                mask[target] = False

            i += 1

        j = int(mask.sum())
        cluster[mask] = torch.arange(i, i + j, device=x.device)
        i += j

        new_x = scatter(x, cluster, dim=0, dim_size=i, reduce='sum')
        new_edge_score = edge_score[new_edge_indices]
        if int(mask.sum()) > 0:
            remaining_score = x.new_ones(
                (new_x.size(0) - len(new_edge_indices), ))
            new_edge_score = torch.cat([new_edge_score, remaining_score])
        new_x = new_x * new_edge_score.view(-1, 1)

        new_edge_index = coalesce(cluster[edge_index], num_nodes=new_x.size(0))
        new_batch = x.new_empty(new_x.size(0), dtype=torch.long)
        new_batch = new_batch.scatter_(0, cluster, batch)

        unpool_info = UnpoolInfo(edge_index=edge_index, cluster=cluster,
                                 batch=batch, new_edge_score=new_edge_score)

        return new_x, new_edge_index, new_batch, unpool_info

    def unpool(
        self,
        x: Tensor,
        unpool_info: UnpoolInfo,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        new_x = x / unpool_info.new_edge_score.view(-1, 1)
        new_x = new_x[unpool_info.cluster]
        return new_x, unpool_info.edge_index, unpool_info.batch

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels})'
