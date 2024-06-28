from typing import Optional

import paddle
from init import zeros
from linear import Linear
from message_passing import MessagePassing
from paddle import Tensor
from paddle.nn import LayerList
from utils import add_self_loops
from utils import get_laplacian
from utils import masked_fill
from utils import remove_self_loops

OptTensor = Optional[Tensor]


class ChebConv(MessagePassing):
    r"""The chebyshev spectral graph convolutional operator from the
    `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering" <https://arxiv.org/abs/1606.09375>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \sum_{k=1}^{K} \mathbf{Z}^{(k)} \cdot
        \mathbf{\Theta}^{(k)}

    where :math:`\mathbf{Z}^{(k)}` is computed recursively by

    .. math::
        \mathbf{Z}^{(1)} &= \mathbf{X}

        \mathbf{Z}^{(2)} &= \mathbf{\hat{L}} \cdot \mathbf{X}

        \mathbf{Z}^{(k)} &= 2 \cdot \mathbf{\hat{L}} \cdot
        \mathbf{Z}^{(k-1)} - \mathbf{Z}^{(k-2)}

    and :math:`\mathbf{\hat{L}}` denotes the scaled and normalized Laplacian
    :math:`\frac{2\mathbf{L}}{\lambda_{\max}} - \mathbf{I}`.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`paddle.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*,
          batch vector :math:`(|\mathcal{V}|)` *(optional)*,
          maximum :obj:`lambda` value :math:`(|\mathcal{G}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        normalization: Optional[str] = "sym",
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        assert K > 0
        assert normalization in [None, "sym", "rw"], "Invalid normalization"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.lins = LayerList(
            [
                Linear(
                    in_channels, out_channels, bias=False, weight_initializer="glorot"
                )
                for _ in range(K)
            ]
        )
        if bias:
            self.bias = paddle.create_parameter([out_channels], paddle.float32)
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    def __norm__(
        self,
        edge_index,
        num_nodes: Optional[int],
        edge_weight: OptTensor,
        normalization: Optional[str],
        lambda_max,
        dtype: Optional[int] = None,
        batch: OptTensor = None,
    ):

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = get_laplacian(
            edge_index, edge_weight, normalization, dtype, num_nodes
        )

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight = masked_fill(
            edge_weight, edge_weight == float("inf"), 0
        )  ###########

        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=-1.0, num_nodes=num_nodes
        )
        assert edge_weight is not None
        return edge_index, edge_weight

    def forward(
        self,
        x,
        edge_index,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
        lambda_max: OptTensor = None,
    ):
        """"""
        if self.normalization != "sym" and lambda_max is None:
            raise ValueError(
                "You need to pass `lambda_max` to `forward() in`"
                "case the normalization is non-symmetric."
            )

        if lambda_max is None:
            lambda_max = paddle.to_tensor(2.0, dtype=x.dtype)
        if not isinstance(lambda_max, paddle.Tensor):
            lambda_max = paddle.to_tensor(lambda_max, dtype=x.dtype)
        assert lambda_max is not None

        edge_index, norm = self.__norm__(
            edge_index,
            x.shape[self.node_dim],
            edge_weight,
            self.normalization,
            lambda_max,
            dtype=x.dtype,
            batch=batch,
        )
        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = self.lins[0](Tx_0)
        # propagate_type: (x: Tensor, norm: Tensor)
        if len(self.lins) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm, size=None)
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
            Tx_2 = 2.0 * Tx_2 - Tx_0
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, norm):
        return norm.reshape([-1, 1]) * x_j

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, K={len(self.lins)}, "
            f"normalization={self.normalization})"
        )
