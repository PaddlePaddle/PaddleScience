import paddle
import paddle.nn as nn

from ppsci.arch import base


class ConvLayer(nn.Layer):
    def __init__(self, atom_fea_len, nbr_fea_len):
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(
            2 * self.atom_fea_len + self.nbr_fea_len, 2 * self.atom_fea_len
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1D(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1D(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = paddle.concat(
            [
                paddle.expand(
                    atom_in_fea.unsqueeze(1), shape=[N, M, self.atom_fea_len]
                ),
                atom_nbr_fea,
                nbr_fea,
            ],
            axis=2,
        )
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = paddle.reshape(
            self.bn1(paddle.reshape(total_gated_fea, [-1, self.atom_fea_len * 2])),
            [N, M, self.atom_fea_len * 2],
        )
        nbr_filter, nbr_core = paddle.chunk(total_gated_fea, chunks=2, axis=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = paddle.sum(nbr_filter * nbr_core, axis=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(base.Arch):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.

    Args:
        orig_atom_fea_len (int): Number of atom features in the input.
        nbr_fea_len (int): Number of bond features.
        atom_fea_len (int): Number of hidden atom features in the convolutional layers.
        n_conv (int): Number of convolutional layers.
        h_fea_len (int): Number of hidden features after pooling.
        n_h (int): Number of hidden layers after pooling.

    Examples:
         >>> import paddle
         >>> import ppsci
         >>> model = ppsci.arch.CrystalGraphConvNet(
         ...     orig_atom_fea_len=92,
         ...     nbr_fea_len=41,
         ...     atom_fea_len=64,
         ...     n_conv=3,
         ...     h_fea_len=128,
         ...     n_h=1,
         ... )
         >>> input_dict = {
         ...     "i": [
         ...         paddle.rand(shape=[45, 92]), paddle.rand(shape=[45, 12, 41]),
         ...         paddle.randint(high=45, shape=[45, 12]),
         ...         [
         ...             paddle.randint(high=32, shape=[32]), paddle.randint(high=8, shape=[8]),
         ...             paddle.randint(high=2, shape=[2]), paddle.randint(high=3, shape=[3])
         ...         ]
         ...     ]
         ... }
         >>> output_dict = model(input_dict)
         >>> print(output_dict["out"].shape)
         [4, 1]
    """

    def __init__(
        self,
        orig_atom_fea_len: int,
        nbr_fea_len: int,
        atom_fea_len: int,
        n_conv: int,
        h_fea_len: int,
        n_h: int,
    ):

        super().__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.LayerList(
            [
                ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
                for _ in range(n_conv)
            ]
        )
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.LayerList(
                [nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)]
            )
            self.softpluses = nn.LayerList([nn.Softplus() for _ in range(n_h - 1)])

        self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, input) -> paddle.Tensor:
        """
        Forward pass.

        N: Total number of atoms in the batch.
        M: Max number of neighbors.
        N0: Total number of crystals in the batch.

        Args:
            input (list): List of input, which includes the following elements:
                atom_fea (paddle.Tensor): Shape (N, orig_atom_fea_len). Atom features from atom type.
                nbr_fea (paddle.Tensor): Shape (N, M, nbr_fea_len). Bond features of each atom's M neighbors.
                nbr_fea_idx (paddle.Tensor): Shape (N, M). Indices of M neighbors of each atom.
                crystal_atom_idx (list): List of paddle.Tensor of length N0. Mapping from the crystal idx to atom idx.

        Returns:
            paddle.Tensor: Shape (N,). Atom hidden features after convolution.
        """
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input["i"]
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if hasattr(self, "fcs") and hasattr(self, "softpluses"):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        out_dict = {"out": out}
        return out_dict

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Args:
            atom_fea (paddle.Tensor): Shape (N, atom_fea_len). Atom feature vectors of the batch.
            crystal_atom_idx (List[paddle.Tensor]): Length N0. Mapping from the crystal idx to atom idx
        """
        assert (
            sum([len(idx_map) for idx_map in crystal_atom_idx])
            == atom_fea.data.shape[0]
        )
        summed_fea = [
            paddle.mean(atom_fea[idx_map], axis=0, keepdim=True)
            for idx_map in crystal_atom_idx
        ]
        return paddle.concat(summed_fea, axis=0)
