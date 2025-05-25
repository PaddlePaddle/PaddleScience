from typing import List, Tuple

import paddle
from paddle import Tensor
from paddle.io import DataLoader

class KGTripletLoader(DataLoader):
    def __init__(self, head_index: Tensor, rel_type: Tensor,
                 tail_index: Tensor, **kwargs):
        self.head_index = head_index
        self.rel_type = rel_type
        self.tail_index = tail_index

        super().__init__(dataset=range(head_index.shape[0]), batch_sampler=None, collate_fn=self.sample, **kwargs)

    def sample(self, index: List[int]) -> Tuple[Tensor, Tensor, Tensor]:
        index = paddle.to_tensor(index, place=self.head_index.place)

        head_index = paddle.index_select(self.head_index, index)
        rel_type = paddle.index_select(self.rel_type, index)
        tail_index = paddle.index_select(self.tail_index, index)

        return head_index, rel_type, tail_index
