from typing import Any, Dict, List, Optional, Union

import paddle
from paddle import Tensor

from paddle_geometric.nn.aggr import Aggregation, MultiAggregation
from paddle_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from paddle_geometric.utils import degree


class DegreeScalerAggregation(Aggregation):
    r"""Combines one or more aggregators and transforms its output with one or
    more scalers as introduced in the `"Principal Neighbourhood Aggregation for
    Graph Nets" <https://arxiv.org/abs/2004.05718>`_ paper.
    The scalers are normalised by the in-degree of the training set and so must
    be provided at time of construction.
    See :class:`paddle_geometric.nn.conv.PNAConv` for more information.

    Args:
        aggr (str or [str] or Aggregation): The aggregation scheme to use.
        scaler (str or list): Set of scaling function identifiers, namely one
            or more of :obj:`"identity"`, :obj:`"amplification"`,
            :obj:`"attenuation"`, :obj:`"linear"` and :obj:`"inverse_linear"`.
        deg (Tensor): Histogram of in-degrees of nodes in the training set,
            used by scalers to normalize.
        train_norm (bool, optional): Whether normalization parameters
            are trainable. (default: :obj:`False`)
        aggr_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective aggregation function in case it gets automatically
            resolved. (default: :obj:`None`)
    """
    def __init__(
        self,
        aggr: Union[str, List[str], Aggregation],
        scaler: Union[str, List[str]],
        deg: Tensor,
        train_norm: bool = False,
        aggr_kwargs: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__()

        if isinstance(aggr, (str, Aggregation)):
            self.aggr = aggr_resolver(aggr, **(aggr_kwargs or {}))
        elif isinstance(aggr, (tuple, list)):
            self.aggr = MultiAggregation(aggr, aggr_kwargs)
        else:
            raise ValueError(f"Only strings, list, tuples and instances of"
                             f"`paddle_geometric.nn.aggr.Aggregation` are "
                             f"valid aggregation schemes (got '{type(aggr)}')")

        self.scaler = [scaler] if isinstance(aggr, str) else scaler

        deg = deg.astype('float32')
        N = int(deg.sum().item())
        bin_degree = paddle.arange(deg.shape[0], dtype='float32')

        self.init_avg_deg_lin = float((bin_degree * deg).sum().item()) / N
        self.init_avg_deg_log = float(((bin_degree + 1).log() * deg).sum().item()) / N

        if train_norm:
            self.avg_deg_lin = self.create_parameter(
                shape=[1], default_initializer=paddle.nn.initializer.Constant(self.init_avg_deg_lin))
            self.avg_deg_log = self.create_parameter(
                shape=[1], default_initializer=paddle.nn.initializer.Constant(self.init_avg_deg_log))
        else:
            self.register_buffer('avg_deg_lin', paddle.to_tensor([self.init_avg_deg_lin]))
            self.register_buffer('avg_deg_log', paddle.to_tensor([self.init_avg_deg_log]))

    def reset_parameters(self):
        self.avg_deg_lin.set_value(paddle.full([1], self.init_avg_deg_lin))
        self.avg_deg_log.set_value(paddle.full([1], self.init_avg_deg_log))

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        # Ensure `index` is provided
        self.assert_index_present(index)

        out = self.aggr(x, index, ptr, dim_size, dim)

        assert index is not None
        deg = degree(index, num_nodes=dim_size, dtype=out.dtype)
        size = [1] * len(out.shape)
        size[dim] = -1
        deg = deg.reshape(size)

        outs = []
        for scaler in self.scaler:
            if scaler == 'identity':
                out_scaler = out
            elif scaler == 'amplification':
                out_scaler = out * (paddle.log(deg + 1) / self.avg_deg_log)
            elif scaler == 'attenuation':
                out_scaler = out * (self.avg_deg_log / paddle.log(deg.clip(min=1) + 1))
            elif scaler == 'linear':
                out_scaler = out * (deg / self.avg_deg_lin)
            elif scaler == 'inverse_linear':
                out_scaler = out * (self.avg_deg_lin / deg.clip(min=1))
            else:
                raise ValueError(f"Unknown scaler '{scaler}'")
            outs.append(out_scaler)

        return paddle.concat(outs, axis=-1) if len(outs) > 1 else outs[0]
