from __future__ import annotations

import itertools
from collections.abc import Sequence

import paddle
from paddle import nn


def aggregate(
    data: paddle.Tensor, owners: paddle.Tensor, *, average=True, num_owner=None
) -> paddle.Tensor:
    """Aggregate rows in data by specifying the owners.

    Args:
        data (Tensor): data tensor to aggregate [n_row, feature_dim]
        owners (Tensor): specify the owner of each row [n_row, 1]
        average (bool): if True, average the rows, if False, sum the rows.
            Default = True
        num_owner (int, optional): the number of owners, this is needed if the
            max idx of owner is not presented in owners tensor
            Default = None

    Returns:
        output (Tensor): [num_owner, feature_dim]
    """

    bin_count = paddle.bincount(x=owners.cast("int32"))
    # bin_count = (bin_count!=0).cast(dtype=bin_count.dtype)
    bin_count = paddle.where(
        bin_count != 0, bin_count, paddle.ones([1], dtype=bin_count.dtype)
    )
    # .where(bin_count != 0, y=paddle.ones(
    #    shape=[1], dtype=bin_count.dtype))
    # bin_count = bin_count.where(bin_count != 0, bin_count.new_ones(1))
    if num_owner is not None and tuple(bin_count.shape)[0] != num_owner:
        difference = num_owner - tuple(bin_count.shape)[0]
        bin_count = paddle.concat(
            x=[bin_count, paddle.ones(shape=difference, dtype=bin_count.dtype)]
        )

    output0 = paddle.zeros(
        shape=[tuple(bin_count.shape)[0], tuple(data.shape)[1]], dtype=data.dtype
    )
    output0.stop_gradient = False
    output = output0.index_add(axis=0, index=owners.cast("int32"), value=data)

    if average:
        output = (output.T / bin_count).T
    return output


class MLP(paddle.nn.Layer):
    """Multi-Layer Perceptron used for non-linear regression."""

    def __init__(
        self,
        input_dim: int,
        *,
        output_dim: int = 1,
        hidden_dim: (int | Sequence[int] | None) = (64, 64),
        dropout: float = 0,
        activation: str = "silu",
        bias: bool = True,
    ) -> None:
        """Initialize the MLP.

        Args:
            input_dim (int): the input dimension
            output_dim (int): the output dimension
            hidden_dim (list[int] | int]): a list of integers or a single integer
                representing the number of hidden units in each layer of the MLP.
                Default = [64, 64]
            dropout (float): the dropout rate before each linear layer. Default: 0
            activation (str, optional): The name of the activation function to use
                in the gated MLP. Must be one of "relu", "silu", "tanh", or "gelu".
                Default = "silu"
            bias (bool): whether to use bias in each Linear layers.
                Default = True
        """
        super().__init__()
        if hidden_dim is None or hidden_dim == 0:
            layers = [
                paddle.nn.Dropout(p=dropout),
                paddle.nn.Linear(
                    in_features=input_dim, out_features=output_dim, bias_attr=bias
                ),
            ]
        elif isinstance(hidden_dim, int):
            layers = [
                paddle.nn.Linear(
                    in_features=input_dim, out_features=hidden_dim, bias_attr=bias
                ),
                find_activation(activation),
                paddle.nn.Dropout(p=dropout),
                paddle.nn.Linear(
                    in_features=hidden_dim, out_features=output_dim, bias_attr=bias
                ),
            ]
        elif isinstance(hidden_dim, Sequence):
            layers = [
                paddle.nn.Linear(
                    in_features=input_dim, out_features=hidden_dim[0], bias_attr=bias
                ),
                find_activation(activation),
            ]
            if len(hidden_dim) != 1:
                for h_in, h_out in itertools.pairwise(hidden_dim):
                    layers.append(
                        paddle.nn.Linear(
                            in_features=h_in, out_features=h_out, bias_attr=bias
                        )
                    )
                    layers.append(find_activation(activation))
            layers.append(paddle.nn.Dropout(p=dropout))
            layers.append(
                paddle.nn.Linear(
                    in_features=hidden_dim[-1], out_features=output_dim, bias_attr=bias
                )
            )
        else:
            raise TypeError(
                f"hidden_dim={hidden_dim!r} must be an integer, a list of integers, or None."
            )
        self.layers = paddle.nn.Sequential(*layers)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Performs a forward pass through the MLP.

        Args:
            x (Tensor): a tensor of shape (batch_size, input_dim)

        Returns:
            Tensor: a tensor of shape (batch_size, output_dim)
        """
        return self.layers(x)


class GatedMLP(paddle.nn.Layer):
    """Gated MLP
    similar model structure is used in CGCNN and M3GNet.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: (int | list[int] | None) = None,
        dropout: float = 0,
        activation: str = "silu",
        norm: str = "batch",
        bias: bool = True,
    ) -> None:
        """Initialize a gated MLP.

        Args:
            input_dim (int): the input dimension
            output_dim (int): the output dimension
            hidden_dim (list[int] | int]): a list of integers or a single integer
                representing the number of hidden units in each layer of the MLP.
                Default = None
            dropout (float): the dropout rate before each linear layer.
                Default: 0
            activation (str, optional): The name of the activation function to use in
                the gated MLP. Must be one of "relu", "silu", "tanh", or "gelu".
                Default = "silu"
            norm (str, optional): The name of the normalization layer to use on the
                updated atom features. Must be one of "batch", "layer", or None.
                Default = "batch"
            bias (bool): whether to use bias in each Linear layers.
                Default = True
        """
        super().__init__()
        self.mlp_core = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
            bias=bias,
        )
        self.mlp_gate = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
            bias=bias,
        )
        self.activation = find_activation(activation)
        self.sigmoid = paddle.nn.Sigmoid()
        self.norm = norm
        self.bn1 = find_normalization(name=norm, dim=output_dim)
        self.bn2 = find_normalization(name=norm, dim=output_dim)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Performs a forward pass through the MLP.

        Args:
            x (Tensor): a tensor of shape (batch_size, input_dim)

        Returns:
            Tensor: a tensor of shape (batch_size, output_dim)
        """
        if self.norm is None:
            core = self.activation(self.mlp_core(x))
            gate = self.sigmoid(self.mlp_gate(x))
        else:
            core = self.activation(self.bn1(self.mlp_core(x)))
            gate = self.sigmoid(self.bn2(self.mlp_gate(x)))
        return core * gate


class ScaledSiLU(paddle.nn.Layer):
    """Scaled Sigmoid Linear Unit."""

    def __init__(self) -> None:
        """Initialize a scaled SiLU."""
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = paddle.nn.Silu()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Forward pass."""
        return self._activation(x) * self.scale_factor


def find_activation(name: str) -> paddle.nn.Layer:
    """Return an activation function using name."""
    try:
        return {
            "relu": paddle.nn.ReLU,
            "silu": paddle.nn.Silu,
            "scaledsilu": ScaledSiLU,
            "gelu": paddle.nn.GELU,
            "softplus": paddle.nn.Softplus,
            "sigmoid": paddle.nn.Sigmoid,
            "tanh": paddle.nn.Tanh,
        }[name.lower()]()
    except KeyError as exc:
        raise NotImplementedError from exc


def find_normalization(name: str, dim: (int | None) = None) -> (paddle.nn.Layer | None):
    """Return an normalization function using name."""
    if name is None:
        return None
    return {"batch": nn.BatchNorm1D(dim), "layer": nn.LayerNorm(dim)}.get(
        name.lower(), None
    )
