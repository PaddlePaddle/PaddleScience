import copy
import inspect
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

import paddle
from paddle import Tensor
from paddle.nn import Linear, LayerList
from tqdm import tqdm

from paddle_geometric.data import Data
from paddle_geometric.loader import CachedLoader, NeighborLoader
from paddle_geometric.nn.conv import (
    EdgeConv,
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    MessagePassing,
    PNAConv,
    SAGEConv,
)
from paddle_geometric.nn.models import MLP
from paddle_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from paddle_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from paddle_geometric.typing import Adj, OptTensor
from paddle_geometric.utils._trim_to_layer import TrimToLayer


class BasicGNN(paddle.nn.Layer):
    r"""An abstract class for implementing basic GNN models."""

    supports_edge_weight: Final[bool]
    supports_edge_attr: Final[bool]
    supports_norm_batch: Final[bool]

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.dropout = paddle.nn.Dropout(p=dropout)
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.jk_mode = jk
        self.act_first = act_first
        self.norm = norm if isinstance(norm, str) else None
        self.norm_kwargs = norm_kwargs

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels

        self.convs = LayerList()
        if num_layers > 1:
            self.convs.append(self.init_conv(in_channels, hidden_channels, **kwargs))
            if isinstance(in_channels, (tuple, list)):
                in_channels = (hidden_channels, hidden_channels)
            else:
                in_channels = hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(self.init_conv(in_channels, hidden_channels, **kwargs))
            if isinstance(in_channels, (tuple, list)):
                in_channels = (hidden_channels, hidden_channels)
            else:
                in_channels = hidden_channels
        if out_channels is not None and jk is None:
            self._is_conv_to_out = True
            self.convs.append(self.init_conv(in_channels, out_channels, **kwargs))
        else:
            self.convs.append(self.init_conv(in_channels, hidden_channels, **kwargs))

        self.norms = LayerList()
        norm_layer = normalization_resolver(norm, hidden_channels, **(norm_kwargs or {}))
        if norm_layer is None:
            norm_layer = paddle.nn.Identity()

        self.supports_norm_batch = False
        if hasattr(norm_layer, 'forward'):
            norm_params = inspect.signature(norm_layer.forward).parameters
            self.supports_norm_batch = 'batch' in norm_params

        for _ in range(num_layers - 1):
            self.norms.append(copy.deepcopy(norm_layer))

        if jk is not None:
            self.norms.append(copy.deepcopy(norm_layer))
        else:
            self.norms.append(paddle.nn.Identity())

        if jk is not None and jk != 'last':
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        if jk is not None:
            if jk == 'cat':
                in_channels = num_layers * hidden_channels
            else:
                in_channels = hidden_channels
            self.lin = Linear(in_channels, self.out_channels)

        self._trim = TrimToLayer()

    def init_conv(self, in_channels: Union[int, Tuple[int, int]], out_channels: int, **kwargs) -> MessagePassing:
        raise NotImplementedError

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()
        if hasattr(self, 'jk'):
            self.jk.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
    ) -> Tensor:
        xs: List[Tensor] = []
        assert len(self.convs) == len(self.norms)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if (not paddle.jit.is_scripting()
                    and num_sampled_nodes_per_hop is not None):
                x, edge_index, value = self._trim(
                    i,
                    num_sampled_nodes_per_hop,
                    num_sampled_edges_per_hop,
                    x,
                    edge_index,
                    edge_weight if edge_weight is not None else edge_attr,
                )
                if edge_weight is not None:
                    edge_weight = value
                else:
                    edge_attr = value

            if self.supports_edge_weight and self.supports_edge_attr:
                x = conv(x, edge_index, edge_weight=edge_weight, edge_attr=edge_attr)
            elif self.supports_edge_weight:
                x = conv(x, edge_index, edge_weight=edge_weight)
            elif self.supports_edge_attr:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)

            if i < self.num_layers - 1 or self.jk_mode is not None:
                if self.act is not None and self.act_first:
                    x = self.act(x)
                if self.supports_norm_batch:
                    x = norm(x, batch, batch_size)
                else:
                    x = norm(x)
                if self.act is not None and not self.act_first:
                    x = self.act(x)
                x = self.dropout(x)
                if hasattr(self, 'jk'):
                    xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x

        return x

    @paddle.no_grad()
    def inference_per_layer(
        self,
        layer: int,
        x: Tensor,
        edge_index: Adj,
        batch_size: int,
    ) -> Tensor:
        x = self.convs[layer](x, edge_index)[:batch_size]

        if layer == self.num_layers - 1 and self.jk_mode is None:
            return x

        if self.act is not None and self.act_first:
            x = self.act(x)
        if self.norms is not None:
            x = self.norms[layer](x)
        if self.act is not None and not self.act_first:
            x = self.act(x)
        if layer == self.num_layers - 1 and hasattr(self, 'lin'):
            x = self.lin(x)

        return x

    @paddle.no_grad()
    def inference(
        self,
        loader: NeighborLoader,
        device: Optional[Union[str, str]] = None,
        embedding_device: Union[str, str] = 'cpu',
        progress_bar: bool = False,
        cache: bool = False,
    ) -> Tensor:
        assert self.jk_mode is None or self.jk_mode == 'last'
        assert isinstance(loader, NeighborLoader)
        assert len(loader.dataset) == loader.data.num_nodes
        assert len(loader.node_sampler.num_neighbors) == 1
        assert not self.training

        if progress_bar:
            pbar = tqdm(total=len(self.convs) * len(loader))
            pbar.set_description('Inference')

        x_all = loader.data.x.to(embedding_device)

        if cache:
            def transform(data: Data) -> Data:
                kwargs = dict(n_id=data.n_id, batch_size=data.batch_size)
                if hasattr(data, 'adj_t'):
                    kwargs['adj_t'] = data.adj_t
                else:
                    kwargs['edge_index'] = data.edge_index

                return Data.from_dict(kwargs)

            loader = CachedLoader(loader, device=device, transform=transform)

        for i in range(self.num_layers):
            xs: List[Tensor] = []
            for batch in loader:
                x = x_all[batch.n_id].to(device)
                batch_size = batch.batch_size
                if hasattr(batch, 'adj_t'):
                    edge_index = batch.adj_t.to(device)
                else:
                    edge_index = batch.edge_index.to(device)

                x = self.inference_per_layer(i, x, edge_index, batch_size)
                xs.append(x.to(embedding_device))

                if progress_bar:
                    pbar.update(1)

            x_all = paddle.concat(xs, axis=0)

        if progress_bar:
            pbar.close()

        return x_all

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')


class GCN(BasicGNN):
    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> MessagePassing:
        return GCNConv(in_channels, out_channels, **kwargs)


class GraphSAGE(BasicGNN):
    def init_conv(self, in_channels: Union[int, Tuple[int, int]], out_channels: int, **kwargs) -> MessagePassing:
        return SAGEConv(in_channels, out_channels, **kwargs)


class GIN(BasicGNN):
    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> MessagePassing:
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return GINConv(mlp, **kwargs)


class GAT(BasicGNN):
    def init_conv(self, in_channels: Union[int, Tuple[int, int]], out_channels: int, **kwargs) -> MessagePassing:
        v2 = kwargs.pop('v2', False)
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)

        if getattr(self, '_is_conv_to_out', False):
            concat = False

        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'GATConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        Conv = GATConv if not v2 else GATv2Conv
        return Conv(in_channels, out_channels, heads=heads, concat=concat, dropout=self.dropout.p, **kwargs)


class PNA(BasicGNN):
    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> MessagePassing:
        return PNAConv(in_channels, out_channels, **kwargs)


class EdgeCNN(BasicGNN):
    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> MessagePassing:
        mlp = MLP(
            [2 * in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return EdgeConv(mlp, **kwargs)


__all__ = [
    'GCN',
    'GraphSAGE',
    'GIN',
    'GAT',
    'PNA',
    'EdgeCNN',
]
