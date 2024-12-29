from __future__ import annotations

import paddle
from chgnet.model.functions import MLP
from chgnet.model.functions import GatedMLP
from chgnet.model.functions import aggregate
from chgnet.model.functions import find_activation
from chgnet.model.functions import find_normalization


class AtomConv(paddle.nn.Layer):
    """A convolution Layer to update atom features."""

    def __init__(
        self,
        *,
        atom_fea_dim: int,
        bond_fea_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0,
        activation: str = "silu",
        norm: (str | None) = None,
        use_mlp_out: bool = True,
        mlp_out_bias: bool = False,
        resnet: bool = True,
        gMLP_norm: (str | None) = None,
    ) -> None:
        """Initialize the AtomConv layer.

        Args:
            atom_fea_dim (int): The dimensionality of the input atom features.
            bond_fea_dim (int): The dimensionality of the input bond features.
            hidden_dim (int, optional): The dimensionality of the hidden layers in the
                gated MLP.
                Default = 64
            dropout (float, optional): The dropout rate to apply to the gated MLP.
                Default = 0.
            activation (str, optional): The name of the activation function to use in
                the gated MLP. Must be one of "relu", "silu", "tanh", or "gelu".
                Default = "silu"
            norm (str, optional): The name of the normalization layer to use on the
                updated atom features. Must be one of "batch", "layer", or None.
                Default = None
            use_mlp_out (bool, optional): Whether to apply an MLP output layer to the
                updated atom features.
                Default = True
            mlp_out_bias (bool): whether to use bias in the output MLP Linear layer.
                Default = False
            resnet (bool, optional): Whether to apply a residual connection to the
                updated atom features.
                Default = True
            gMLP_norm (str, optional): The name of the normalization layer to use on the
                gated MLP. Must be one of "batch", "layer", or None.
                Default = None
        """
        super().__init__()
        self.use_mlp_out = use_mlp_out
        self.resnet = resnet
        self.activation = find_activation(activation)
        self.twoBody_atom = GatedMLP(
            input_dim=2 * atom_fea_dim + bond_fea_dim,
            output_dim=atom_fea_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            norm=gMLP_norm,
            activation=activation,
        )
        if self.use_mlp_out:
            self.mlp_out = MLP(
                input_dim=atom_fea_dim,
                output_dim=atom_fea_dim,
                hidden_dim=0,
                bias=mlp_out_bias,
            )
        self.atom_norm = find_normalization(name=norm, dim=atom_fea_dim)

    def forward(
        self,
        atom_feas: paddle.Tensor,
        bond_feas: paddle.Tensor,
        bond_weights: paddle.Tensor,
        atom_graph: paddle.Tensor,
        directed2undirected: paddle.Tensor,
    ) -> paddle.Tensor:
        """Forward pass of AtomConv module that updates the atom features and
            optionally bond features.

        Args:
            atom_feas (Tensor): Input tensor with shape
                [num_batch_atoms, atom_fea_dim]
            bond_feas (Tensor): Input tensor with shape
                [num_undirected_bonds, bond_fea_dim]
            bond_weights (Tensor): AtomGraph bond weights with shape
                [num_undirected_bonds, bond_fea_dim]
            atom_graph (Tensor): Directed AtomGraph adjacency list with shape
                [num_directed_bonds, 2]
            directed2undirected (Tensor): Index tensor that maps directed bonds to
                undirected bonds.with shape
                [num_undirected_bonds]

        Returns:
            Tensor: the updated atom features tensor with shape
            [num_batch_atom, atom_fea_dim]

        Notes:
            - num_batch_atoms = sum(num_atoms) in batch
        """
        center_atoms = paddle.index_select(x=atom_feas, axis=0, index=atom_graph[:, 0])
        nbr_atoms = paddle.index_select(x=atom_feas, axis=0, index=atom_graph[:, 1])
        bonds = paddle.index_select(x=bond_feas, axis=0, index=directed2undirected)
        messages = paddle.concat(x=[center_atoms, bonds, nbr_atoms], axis=1)
        messages = self.twoBody_atom(messages)
        bond_weight = paddle.index_select(
            x=bond_weights, axis=0, index=directed2undirected
        )
        messages *= bond_weight
        new_atom_feas = aggregate(
            messages, atom_graph[:, 0], average=False, num_owner=len(atom_feas)
        )
        if self.use_mlp_out:
            new_atom_feas = self.mlp_out(new_atom_feas)
        if self.resnet:
            new_atom_feas += atom_feas
        if self.atom_norm is not None:
            new_atom_feas = self.atom_norm(new_atom_feas)
        return new_atom_feas


class BondConv(paddle.nn.Layer):
    """A convolution Layer to update bond features."""

    def __init__(
        self,
        atom_fea_dim: int,
        bond_fea_dim: int,
        angle_fea_dim: int,
        *,
        hidden_dim: int = 64,
        dropout: float = 0,
        activation: str = "silu",
        norm: (str | None) = None,
        use_mlp_out: bool = True,
        mlp_out_bias: bool = False,
        resnet=True,
        gMLP_norm: (str | None) = None,
    ) -> None:
        """Initialize the BondConv layer.

        Args:
            atom_fea_dim (int): The dimensionality of the input atom features.
            bond_fea_dim (int): The dimensionality of the input bond features.
            angle_fea_dim (int): The dimensionality of the input angle features.
            hidden_dim (int, optional): The dimensionality of the hidden layers
                in the gated MLP.
                Default = 64
            dropout (float, optional): The dropout rate to apply to the gated MLP.
                Default = 0.
            activation (str, optional): The name of the activation function to use
                in the gated MLP. Must be one of "relu", "silu", "tanh", or "gelu".
                Default = "silu"
            norm (str, optional): The name of the normalization layer to use on the
                updated atom features. Must be one of "batch", "layer", or None.
                Default = None
            use_mlp_out (bool, optional): Whether to apply an MLP output layer to the
                updated atom features.
                Default = True
            mlp_out_bias (bool): whether to use bias in the output MLP Linear layer.
                Default = False
            resnet (bool, optional): Whether to apply a residual connection to the
                updated atom features.
                Default = True
            gMLP_norm (str, optional): The name of the normalization layer to use on the
                gated MLP. Must be one of "batch", "layer", or None.
                Default = None
        """
        super().__init__()
        self.use_mlp_out = use_mlp_out
        self.resnet = resnet
        self.activation = find_activation(activation)
        self.twoBody_bond = GatedMLP(
            input_dim=atom_fea_dim + 2 * bond_fea_dim + angle_fea_dim,
            output_dim=bond_fea_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            norm=gMLP_norm,
            activation=activation,
        )
        if self.use_mlp_out:
            self.mlp_out = MLP(
                input_dim=bond_fea_dim,
                output_dim=bond_fea_dim,
                hidden_dim=0,
                bias=mlp_out_bias,
            )
        self.bond_norm = find_normalization(name=norm, dim=bond_fea_dim)

    def forward(
        self,
        atom_feas: paddle.Tensor,
        bond_feas: paddle.Tensor,
        bond_weights: paddle.Tensor,
        angle_feas: paddle.Tensor,
        bond_graph: paddle.Tensor,
    ) -> paddle.Tensor:
        """Update the bond features.

        Args:
            atom_feas (Tensor): atom features tensor with shape
                [num_batch_atoms, atom_fea_dim]
            bond_feas (Tensor): bond features tensor with shape
                [num_undirected_bonds, bond_fea_dim]
            bond_weights (Tensor): BondGraph bond weights with shape
                [num_undirected_bonds, bond_fea_dim]
            angle_feas (Tensor): angle features tensor with shape
                [num_batch_angles, angle_fea_dim]
            bond_graph (Tensor): Directed BondGraph tensor with shape
                [num_batched_angles, 3]

        Returns:
            new_bond_feas (Tensor): bond feature tensor with shape
                [num_undirected_bonds, bond_fea_dim]

        Notes:
            - num_batch_atoms = sum(num_atoms) in batch
        """
        center_atoms = paddle.index_select(
            x=atom_feas, axis=0, index=bond_graph[:, 0].cast("int32")
        )
        bond_feas_i = paddle.index_select(
            x=bond_feas, axis=0, index=bond_graph[:, 1].cast("int32")
        )
        bond_feas_j = paddle.index_select(
            x=bond_feas, axis=0, index=bond_graph[:, 2].cast("int32")
        )
        total_fea = paddle.concat(
            x=[bond_feas_i, bond_feas_j, angle_feas, center_atoms], axis=1
        )
        bond_update = self.twoBody_bond(total_fea)
        bond_weights_i = paddle.index_select(
            x=bond_weights, axis=0, index=bond_graph[:, 1].cast("int32")
        )
        bond_weights_j = paddle.index_select(
            x=bond_weights, axis=0, index=bond_graph[:, 2].cast("int32")
        )
        bond_update = bond_update * bond_weights_i * bond_weights_j
        new_bond_feas = aggregate(
            bond_update, bond_graph[:, 1], average=False, num_owner=len(bond_feas)
        )
        if self.use_mlp_out:
            new_bond_feas = self.mlp_out(new_bond_feas)
        if self.resnet:
            new_bond_feas += bond_feas
        if self.bond_norm is not None:
            new_bond_feas = self.bond_norm(new_bond_feas)
        return new_bond_feas


class AngleUpdate(paddle.nn.Layer):
    """Update angle features."""

    def __init__(
        self,
        atom_fea_dim: int,
        bond_fea_dim: int,
        angle_fea_dim: int,
        *,
        hidden_dim: int = 0,
        dropout: float = 0,
        activation: str = "silu",
        norm: (str | None) = None,
        resnet: bool = True,
        gMLP_norm: (str | None) = None,
    ) -> None:
        """Initialize the AngleUpdate layer.

        Args:
            atom_fea_dim (int): The dimensionality of the input atom features.
            bond_fea_dim (int): The dimensionality of the input bond features.
            angle_fea_dim (int): The dimensionality of the input angle features.
            hidden_dim (int, optional): The dimensionality of the hidden layers
                in the gated MLP.
                Default = 0
            dropout (float, optional): The dropout rate to apply to the gated MLP.
                Default = 0.
            activation (str, optional): The name of the activation function to use
                in the gated MLP. Must be one of "relu", "silu", "tanh", or "gelu".
                Default = "silu"
            norm (str, optional): The name of the normalization layer to use on the
                updated atom features. Must be one of "batch", "layer", or None.
                Default = None
            resnet (bool, optional): Whether to apply a residual connection to the
                updated atom features.
                Default = True
            gMLP_norm (str, optional): The name of the normalization layer to use on the
                gated MLP. Must be one of "batch", "layer", or None.
                Default = None
        """
        super().__init__()
        self.resnet = resnet
        self.activation = find_activation(activation)
        self.twoBody_bond = GatedMLP(
            input_dim=atom_fea_dim + 2 * bond_fea_dim + angle_fea_dim,
            output_dim=angle_fea_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            norm=gMLP_norm,
            activation=activation,
        )
        self.angle_norm = find_normalization(norm, dim=angle_fea_dim)

    def forward(
        self,
        atom_feas: paddle.Tensor,
        bond_feas: paddle.Tensor,
        angle_feas: paddle.Tensor,
        bond_graph: paddle.Tensor,
    ) -> paddle.Tensor:
        """Update the angle features using bond graph.

        Args:
            atom_feas (Tensor): atom features tensor with shape
                [num_batch_atoms, atom_fea_dim]
            bond_feas (Tensor): bond features tensor with shape
                [num_undirected_bonds, bond_fea_dim]
            angle_feas (Tensor): angle features tensor with shape
                [num_batch_angles, angle_fea_dim]
            bond_graph (Tensor): Directed BondGraph tensor with shape
                [num_batched_angles, 3]

        Returns:
            new_angle_feas (Tensor): angle features tensor with shape
                [num_batch_angles, angle_fea_dim]

        Notes:
            - num_batch_atoms = sum(num_atoms) in batch
        """
        bond_graph = bond_graph.astype("int64")
        center_atoms = paddle.index_select(x=atom_feas, axis=0, index=bond_graph[:, 0])
        bond_feas_i = paddle.index_select(x=bond_feas, axis=0, index=bond_graph[:, 1])
        bond_feas_j = paddle.index_select(x=bond_feas, axis=0, index=bond_graph[:, 2])
        total_fea = paddle.concat(
            x=[bond_feas_i, bond_feas_j, angle_feas, center_atoms], axis=1
        )
        new_angle_feas = self.twoBody_bond(total_fea)

        if self.resnet:
            new_angle_feas += angle_feas
        if self.angle_norm is not None:
            new_angle_feas = self.angle_norm(new_angle_feas)
        return new_angle_feas


class GraphPooling(paddle.nn.Layer):
    """Pooling the sub-graphs in the batched graph."""

    def __init__(self, *, average: bool = False) -> None:
        """Args:
        average (bool): whether to average the features.
        """
        super().__init__()
        self.average = average

    def forward(
        self, atom_feas: paddle.Tensor, atom_owner: paddle.Tensor
    ) -> paddle.Tensor:
        """Merge the atom features that belong to same graph in a batched graph.

        Args:
            atom_feas (Tensor): batched atom features after convolution layers.
                [num_batch_atoms, atom_fea_dim or 1]
            atom_owner (Tensor): graph indices for each atom.
                [num_batch_atoms]

        Returns:
            crystal_feas (Tensor): crystal feature matrix.
                [n_crystals, atom_fea_dim or 1]
        """
        return aggregate(atom_feas, atom_owner, average=self.average)


class GraphAttentionReadOut(paddle.nn.Layer):
    """Multi Head Attention Read Out Layer
    merge the information from atom_feas to crystal_fea.
    """

    def __init__(
        self,
        atom_fea_dim: int,
        num_head: int = 3,
        hidden_dim: int = 32,
        *,
        average=False,
    ) -> None:
        """Initialize the layer.

        Args:
            atom_fea_dim (int): atom feature dimension
            num_head (int): number of attention heads used
            hidden_dim (int): dimension of hidden layer
            average (bool): whether to average the features
        """
        super().__init__()
        self.key = MLP(
            input_dim=atom_fea_dim, output_dim=num_head, hidden_dim=hidden_dim
        )
        self.softmax = paddle.nn.Softmax(axis=0)
        self.average = average

    def forward(
        self, atom_feas: paddle.Tensor, atom_owner: paddle.Tensor
    ) -> paddle.Tensor:
        """Merge the atom features that belong to same graph in a batched graph.

        Args:
            atom_feas (Tensor): batched atom features after convolution layers.
                [num_batch_atoms, atom_fea_dim]
            atom_owner (Tensor): graph indices for each atom.
                [num_batch_atoms]

        Returns:
            crystal_feas (Tensor): crystal feature matrix.
                [n_crystals, atom_fea_dim]
        """
        crystal_feas = []
        weights = self.key(atom_feas)
        bin_count = paddle.bincount(x=atom_owner)
        start_index = 0
        for n_atom in bin_count:
            atom_fea = atom_feas[start_index : start_index + n_atom, :]
            weight = self.softmax(weights[start_index : start_index + n_atom, :])
            crystal_fea = (atom_fea.T @ weight).reshape([-1])
            if self.average:
                crystal_fea /= n_atom
            crystal_feas.append(crystal_fea)
            start_index += n_atom
        return paddle.stack(x=crystal_feas, axis=0)
