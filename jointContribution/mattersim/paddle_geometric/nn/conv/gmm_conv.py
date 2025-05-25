from typing import Tuple, Union

import paddle
from paddle import Tensor
from paddle.nn import Layer, Linear
from paddle.nn.initializer import XavierNormal, Constant

from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.typing import Adj, OptPairTensor, OptTensor, Size


class GMMConv(MessagePassing):
    r"""The gaussian mixture model convolutional operator from the `"Geometric
    Deep Learning on Graphs and Manifolds using Mixture Model CNNs"
    <https://arxiv.org/abs/1611.08402>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \frac{1}{|\mathcal{N}(i)|}
        \sum_{j \in \mathcal{N}(i)} \frac{1}{K} \sum_{k=1}^K
        \mathbf{w}_k(\mathbf{e}_{i,j}) \odot \mathbf{\Theta}_k \mathbf{x}_j,

    where

    .. math::
        \mathbf{w}_k(\mathbf{e}) = \exp \left( -\frac{1}{2} {\left(
        \mathbf{e} - \mathbf{\mu}_k \right)}^{\top} \Sigma_k^{-1}
        \left( \mathbf{e} - \mathbf{\mu}_k \right) \right)

    denotes a weighting function based on trainable mean vector
    :math:`\mathbf{\mu}_k` and diagonal covariance matrix
    :math:`\mathbf{\Sigma}_k`.

    .. note::

        The edge attribute :math:`\mathbf{e}_{ij}` is usually given by
        :math:`\mathbf{e}_{ij} = \mathbf{p}_j - \mathbf{p}_i`, where
        :math:`\mathbf{p}_i` denotes the position of node :math:`i` (see
        :class:`paddle_geometric.transform.Cartesian`).

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        dim (int): Pseudo-coordinate dimensionality.
        kernel_size (int): Number of kernels :math:`K`.
        separate_gaussians (bool, optional): If set to :obj:`True`, will
            learn separate GMMs for every pair of input and output channel,
            inspired by traditional CNNs. (default: :obj:`False`)
        aggr (str, optional): The aggregation operator to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"mean"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`paddle_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V_t}|, F_{out})` if bipartite
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, dim: int, kernel_size: int,
                 separate_gaussians: bool = False, aggr: str = 'mean',
                 root_weight: bool = True, bias: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size
        self.separate_gaussians = separate_gaussians
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.rel_in_channels = in_channels[0]

        if in_channels[0] > 0:
            self.g = self.create_parameter(
                [in_channels[0], out_channels * kernel_size],
                default_initializer=XavierNormal())
            if not self.separate_gaussians:
                self.mu = self.create_parameter([kernel_size, dim], default_initializer=XavierNormal())
                self.sigma = self.create_parameter([kernel_size, dim], default_initializer=XavierNormal())
            else:
                self.mu = self.create_parameter([in_channels[0], out_channels, kernel_size, dim],
                                                default_initializer=XavierNormal())
                self.sigma = self.create_parameter([in_channels[0], out_channels, kernel_size, dim],
                                                   default_initializer=XavierNormal())
        else:
            self.g = None
            self.mu = None
            self.sigma = None
            self._hook = self.register_forward_pre_hook(self.initialize_parameters)

        if root_weight:
            self.root = Linear(in_channels[1], out_channels, bias_attr=False)

        if bias:
            self.bias = self.create_parameter([out_channels], default_initializer=zeros)
        else:
            self.bias = None

    def initialize_parameters(self, layer, input):
        if self.g is None:
            x = input[0][0] if isinstance(input, tuple) else input[0]
            in_channels = x.shape[-1]
            self.g = self.create_parameter([in_channels, self.out_channels * self.kernel_size],
                                           default_initializer=XavierNormal())
            if not self.separate_gaussians:
                self.mu = self.create_parameter([self.kernel_size, self.dim], default_initializer=XavierNormal())
                self.sigma = self.create_parameter([self.kernel_size, self.dim], default_initializer=XavierNormal())
            else:
                self.mu = self.create_parameter([in_channels, self.out_channels, self.kernel_size, self.dim],
                                                default_initializer=XavierNormal())
                self.sigma = self.create_parameter([in_channels, self.out_channels, self.kernel_size, self.dim],
                                                   default_initializer=XavierNormal())
            layer._hook.remove()
            del layer._hook

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        if not self.separate_gaussians:
            out: OptPairTensor = (paddle.matmul(x[0], self.g), x[1])
            out = self.propagate(edge_index, x=out, edge_attr=edge_attr, size=size)
        else:
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None and self.root is not None:
            out = out + self.root(x_r)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        EPS = 1e-15
        F, M = self.rel_in_channels, self.out_channels
        (E, D), K = edge_attr.shape, self.kernel_size

        if not self.separate_gaussians:
            gaussian = -0.5 * (edge_attr.reshape([E, 1, D]) - self.mu.reshape([1, K, D])) ** 2
            gaussian = gaussian / (EPS + self.sigma.reshape([1, K, D]) ** 2)
            gaussian = paddle.exp(gaussian.sum(axis=-1))
            return (x_j.reshape([E, K, M]) * gaussian.reshape([E, K, 1])).sum(axis=-2)
        else:
            gaussian = -0.5 * (edge_attr.reshape([E, 1, 1, 1, D]) - self.mu.reshape([1, F, M, K, D])) ** 2
            gaussian = gaussian / (EPS + self.sigma.reshape([1, F, M, K, D]) ** 2)
            gaussian = paddle.exp(gaussian.sum(axis=-1))
            gaussian = gaussian * self.g.reshape([1, F, M, K])
            gaussian = gaussian.sum(axis=-1)
            return (x_j.reshape([E, F, 1]) * gaussian).sum(axis=-2)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, dim={self.dim})')
