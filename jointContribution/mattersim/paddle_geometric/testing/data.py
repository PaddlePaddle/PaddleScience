from typing import Callable, Optional

import paddle
from paddle import Tensor

from paddle_geometric.data import HeteroData, InMemoryDataset
from paddle_geometric.typing import TensorFrame, paddle_frame
from paddle_geometric.utils import coalesce as coalesce_fn


def get_random_edge_index(
    num_src_nodes: int,
    num_dst_nodes: int,
    num_edges: int,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[str] = None,
    coalesce: bool = False,
) -> Tensor:
    row = paddle.randint(0, num_src_nodes, shape=(num_edges,), dtype=dtype)
    col = paddle.randint(0, num_dst_nodes, shape=(num_edges,), dtype=dtype)
    edge_index = paddle.stack([row, col], axis=0)

    if coalesce:
        edge_index = coalesce_fn(edge_index)

    return edge_index


def get_random_tensor_frame(
    num_rows: int,
    device: Optional[str] = None,
) -> TensorFrame:

    feat_dict = {
        paddle_frame.categorical:
        paddle.randint(0, 3, shape=(num_rows, 3), device=device),
        paddle_frame.numerical:
        paddle.randn(shape=(num_rows, 2), device=device),
    }
    col_names_dict = {
        paddle_frame.categorical: ['a', 'b', 'c'],
        paddle_frame.numerical: ['x', 'y'],
    }
    y = paddle.randn([num_rows], device=device)

    return paddle_frame.TensorFrame(
        feat_dict=feat_dict,
        col_names_dict=col_names_dict,
        y=y,
    )


class FakeHeteroDataset(InMemoryDataset):
    def __init__(self, transform: Optional[Callable] = None):
        super().__init__(transform=transform)

        data = HeteroData()

        num_papers = 100
        num_authors = 10

        data['paper'].x = paddle.randn([num_papers, 16])
        data['author'].x = paddle.randn([num_authors, 8])

        edge_index = get_random_edge_index(
            num_src_nodes=num_papers,
            num_dst_nodes=num_authors,
            num_edges=300,
        )
        data['paper', 'author'].edge_index = edge_index
        data['author', 'paper'].edge_index = edge_index.flip([0])

        data['paper'].y = paddle.randint(0, 4, shape=[num_papers])

        perm = paddle.randperm(num_papers)
        data['paper'].train_mask = paddle.zeros([num_papers], dtype=paddle.bool)
        data['paper'].train_mask[perm[:60]] = True
        data['paper'].val_mask = paddle.zeros([num_papers], dtype=paddle.bool)
        data['paper'].val_mask[perm[60:80]] = True
        data['paper'].test_mask = paddle.zeros([num_papers], dtype=paddle.bool)
        data['paper'].test_mask[perm[80:100]] = True

        self.data, self.slices = self.collate([data])
