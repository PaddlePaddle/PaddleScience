import math
from typing import List, Optional, Union

import paddle
from paddle import Tensor
import paddle.nn as nn
import paddle.nn.functional as F

from paddle_geometric.nn.aggr import Aggregation
from paddle_geometric.nn.aggr.utils import MultiheadAttentionBlock
from paddle_geometric.nn.encoding import PositionalEncoding
from paddle_geometric.utils import scatter


class PatchTransformerAggregation(Aggregation):
    r"""Performs patch transformer aggregation in which the elements to
    aggregate are processed by multi-head attention blocks across patches, as
    described in the `"Simplifying Temporal Heterogeneous Network for
    Continuous-Time Link Prediction"
    <https://dl.acm.org/doi/pdf/10.1145/3583780.3615059>`_ paper.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        patch_size (int): Number of elements in a patch.
        hidden_channels (int): Intermediate size of each sample.
        num_transformer_blocks (int, optional): Number of transformer blocks
            (default: :obj:`1`).
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        dropout (float, optional): Dropout probability of attention weights.
            (default: :obj:`0.0`)
        aggr (str or list[str], optional): The aggregation module, *e.g.*,
            :obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"var"`, :obj:`"std"`. (default: :obj:`"mean"`)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        hidden_channels: int,
        num_transformer_blocks: int = 1,
        heads: int = 1,
        dropout: float = 0.0,
        aggr: Union[str, List[str]] = 'mean',
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.aggrs = [aggr] if isinstance(aggr, str) else aggr

        assert len(self.aggrs) > 0
        for aggr in self.aggrs:
            assert aggr in ['sum', 'mean', 'min', 'max', 'var', 'std']

        self.lin = nn.Linear(in_channels, hidden_channels)
        self.pad_projector = nn.Linear(
            patch_size * hidden_channels,
            hidden_channels,
        )
        self.pe = PositionalEncoding(hidden_channels)

        self.blocks = nn.LayerList([
            MultiheadAttentionBlock(
                channels=hidden_channels,
                heads=heads,
                layer_norm=True,
                dropout=dropout,
            ) for _ in range(num_transformer_blocks)
        ])

        self.fc = nn.Linear(
            hidden_channels * len(self.aggrs),
            out_channels,
        )

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()
        self.pad_projector.reset_parameters()
        self.pe.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()
        self.fc.reset_parameters()

    def forward(
        self,
        x: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
    ) -> Tensor:

        if max_num_elements is None:
            if ptr is not None:
                count = ptr[1:] - ptr[:-1]
            else:
                count = scatter(paddle.ones_like(index), index, dim=0,
                                dim_size=dim_size, reduce='sum')
            max_num_elements = int(count.max().item()) + 1

        # Set `max_num_elements` to a multiple of `patch_size`:
        max_num_elements = (math.floor(max_num_elements / self.patch_size) *
                            self.patch_size)

        x = self.lin(x)

        x, _ = self.to_dense_batch(x, index, ptr, dim_size, dim,
                                   max_num_elements=max_num_elements)

        # [batch_size, num_patches, patch_size * hidden_channels]
        x = x.reshape([x.shape[0], max_num_elements // self.patch_size,
                       self.patch_size * x.shape[-1]])

        # [batch_size, num_patches, hidden_channels]
        x = self.pad_projector(x)

        x = x + self.pe(paddle.arange(x.shape[1], dtype=x.dtype))

        # [batch_size, num_patches, hidden_channels]
        for block in self.blocks:
            x = block(x, x)

        # [batch_size, hidden_channels]
        outs: List[Tensor] = []
        for aggr in self.aggrs:
            out = getattr(paddle, aggr)(x, axis=1)
            outs.append(out[0] if isinstance(out, tuple) else out)
        out = paddle.concat(outs, axis=1) if len(outs) > 1 else outs[0]

        # [batch_size, out_channels]
        return self.fc(out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, patch_size={self.patch_size})')
