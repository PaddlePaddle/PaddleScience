from typing import List, Optional, Tuple, Union

import paddle.nn.functional as F
from paddle import Tensor

from paddle_geometric.nn.aggr import Aggregation, MultiAggregation
from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.dense.linear import Linear
from paddle_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from paddle_geometric.utils import spmm



class SAGEConv(MessagePassing):
    r"""
    The GraphSAGE operator from the "Inductive Representation Learning on
    Large Graphs" (https://arxiv.org/abs/1706.02216) paper.

    This operator computes the node embeddings using the following equation:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    If `project = True`, the equation is modified to apply a linear transformation
    to node features before aggregation as described in Eq. (3) of the paper.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:-1 to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        aggr (str or Aggregation, optional): The aggregation scheme to use.
            Any aggregation of :obj:torch_geometric.nn.aggr can be used,
            *e.g.*, :obj:"mean", :obj:"max", or :obj:"lstm". (default: :obj:"mean")
        normalize (bool, optional): If set to :obj:True, output features will be
            :math:\ell_2-normalized. (default: :obj:False)
        root_weight (bool, optional): If set to :obj:False, the layer will not
            add transformed root node features to the output. (default: :obj:True)
        project (bool, optional): If set to :obj:True, the layer will apply a
            linear transformation followed by an activation function before aggregation.
            (default: :obj:False)
        bias (bool, optional): If set to :obj:False, the layer will not learn
            an additive bias. (default: :obj:True)
        **kwargs (optional): Additional arguments of :class:torch_geometric.nn.conv.MessagePassing.

    Shapes:
        - **inputs:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **outputs:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V_t}|, F_{out})` if bipartite
    """

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
            normalize: bool = False,
            root_weight: bool = True,
            project: bool = False,
            bias: bool = True,
            **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        # If `in_channels` is an integer, we use the same value for both source and target
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        # Handle aggregation options for lstm
        if aggr == 'lstm':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_channels', in_channels[0])
            kwargs['aggr_kwargs'].setdefault('out_channels', in_channels[0])

        # Call super class constructor
        super().__init__(aggr=aggr, **kwargs)

        if self.project:
            if in_channels[0] <= 0:
                raise ValueError(f"'{self.__class__.__name__}' does not "
                                 f"support lazy initialization with "
                                 f"project=True")
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the weights."""
        super().reset_parameters()
        if self.project:
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, size: Size = None) -> Tensor:
        """Forward pass of the SAGEConv layer."""
        if isinstance(x, Tensor):
            x = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        # If root_weight is enabled, we add the transformed root features
        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        # Normalize the output if required
        if self.normalize:
            out = F.normalize(out, p=2., axis=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        """Message function to aggregate the neighbors' features."""
        return x_j

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        """Helper function for aggregation in the propagation step."""
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        """String representation of the layer."""
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')
