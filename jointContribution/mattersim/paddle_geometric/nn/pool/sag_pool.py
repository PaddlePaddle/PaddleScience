from typing import Union, Optional, Callable, Tuple

import paddle
from paddle import nn
from paddle import Tensor
from paddle_geometric.nn import GraphConv
from paddle_geometric.nn.pool.connect import FilterEdges
from paddle_geometric.nn.pool.select import SelectTopK
from paddle_geometric.typing import OptTensor


class SAGPooling(nn.Layer):
    r"""The self-attention pooling operator from the "Self-Attention Graph
    Pooling" <https://arxiv.org/abs/1904.08082>_ and "Understanding
    Attention and Generalization in Graph Neural Networks"
    <https://arxiv.org/abs/1905.02850>_ papers.

    If :obj:min_score :math:\tilde{\alpha} is :obj:None, computes:

        .. math::
            \mathbf{y} &= \textrm{GNN}(\mathbf{X}, \mathbf{A})

            \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot
            \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}

    If :obj:min_score :math:\tilde{\alpha} is a value in :obj:[0, 1],
    computes:

        .. math::
            \mathbf{y} &= \mathrm{softmax}(\textrm{GNN}(\mathbf{X},\mathbf{A}))

            \mathbf{i} &= \mathbf{y}_i > \tilde{\alpha}

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot \mathbf{y})_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}.

    Projections scores are learned based on a graph neural network layer.

    Args:
        in_channels (int): Size of each input sample.
        ratio (float or int): Graph pooling ratio, which is used to compute
            :math:k = \lceil \mathrm{ratio} \cdot N \rceil, or the value
            of :math:k itself, depending on whether the type of :obj:ratio
            is :obj:float or :obj:int.
            This value is ignored if :obj:min_score is not :obj:None.
            (default: :obj:0.5)
        GNN (paddle.nn.Layer, optional): A graph neural network layer for
            calculating projection scores (one of
            :class:paddle_geometric.nn.conv.GraphConv,
            :class:paddle_geometric.nn.conv.GCNConv,
            :class:paddle_geometric.nn.conv.GATConv or
            :class:paddle_geometric.nn.conv.SAGEConv). (default:
            :class:paddle_geometric.nn.conv.GraphConv)
        min_score (float, optional): Minimal node score :math:\tilde{\alpha}
            which is used to compute indices of pooled nodes
            :math:\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}.
            When this value is not :obj:None, the :obj:ratio argument is
            ignored. (default: :obj:None)
        multiplier (float, optional): Coefficient by which features gets
            multiplied after pooling. This can be useful for large graphs and
            when :obj:min_score is used. (default: :obj:1)
        nonlinearity (str or callable, optional): The non-linearity to use.
            (default: :obj:"tanh")
        **kwargs (optional): Additional parameters for initializing the graph
            neural network layer.
    """
    def __init__(
        self,
        in_channels: int,
        ratio: Union[float, int] = 0.5,
        GNN: nn.Layer = GraphConv,
        min_score: Optional[float] = None,
        multiplier: float = 1.0,
        nonlinearity: Union[str, Callable] = 'tanh',
        **kwargs,
    ):
        super(SAGPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier

        self.gnn = GNN(in_channels, 1, **kwargs)
        self.select = SelectTopK(1, ratio, min_score, nonlinearity)
        self.connect = FilterEdges()

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.gnn.reset_parameters()
        self.select.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        attn: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, OptTensor, OptTensor, Tensor, Tensor]:
        r"""Forward pass.

        Args:
            x (paddle.Tensor): The node feature matrix.
            edge_index (paddle.Tensor): The edge indices.
            edge_attr (paddle.Tensor, optional): The edge features.
                (default: :obj:None)
            batch (paddle.Tensor, optional): The batch vector
                :math:\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N, which assigns
                each node to a specific example. (default: :obj:None)
            attn (paddle.Tensor, optional): Optional node-level matrix to use
                for computing attention scores instead of using the node
                feature matrix :obj:x. (default: :obj:None)
        """
        if batch is None:
            batch = paddle.zeros([x.shape[0]], dtype='int64')

        attn = x if attn is None else attn
        attn = attn.reshape([-1, 1]) if attn.ndim == 1 else attn
        attn = self.gnn(attn, edge_index)

        select_out = self.select(attn, batch)

        perm = select_out.node_index
        score = select_out.weight
        assert score is not None

        x = x[perm] * score.reshape([-1, 1])
        x = self.multiplier * x if self.multiplier != 1 else x

        connect_out = self.connect(select_out, edge_index, edge_attr, batch)

        return (x, connect_out.edge_index, connect_out.edge_attr,
                connect_out.batch, perm, score)

    def __repr__(self) -> str:
        if self.min_score is None:
            ratio = f'ratio={self.ratio}'
        else:
            ratio = f'min_score={self.min_score}'

        return (f'{self.__class__.__name__}({self.gnn.__class__.__name__}, '
                f'{self.in_channels}, {ratio}, multiplier={self.multiplier})')
