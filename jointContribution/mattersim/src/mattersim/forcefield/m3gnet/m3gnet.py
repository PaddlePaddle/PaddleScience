from typing import Dict

import paddle
from mattersim.utils.paddle_utils import scatter

from .modules import MLP
from .modules import GatedMLP
from .modules import MainBlock
from .modules import SmoothBesselBasis
from .modules import SphericalBasisLayer
from .scaling import AtomScaling


class M3Gnet(paddle.nn.Layer):
    """
    M3Gnet
    """

    def __init__(
        self,
        num_blocks: int = 4,
        units: int = 128,
        max_l: int = 4,
        max_n: int = 4,
        cutoff: float = 5.0,
        max_z: int = 94,
        threebody_cutoff: float = 4.0,
        **kwargs,
    ):
        super().__init__()
        self.rbf = SmoothBesselBasis(r_max=cutoff, max_n=max_n)
        self.sbf = SphericalBasisLayer(max_n=max_n, max_l=max_l, cutoff=cutoff)
        self.edge_encoder = MLP(
            in_dim=max_n, out_dims=[units], activation="swish", use_bias=False
        )
        module_list = [
            MainBlock(max_n, max_l, cutoff, units, max_n, threebody_cutoff)
            for i in range(num_blocks)
        ]
        self.graph_conv = paddle.nn.LayerList(sublayers=module_list)
        self.final = GatedMLP(
            in_dim=units,
            out_dims=[units, units, 1],
            activation=["swish", "swish", None],
        )
        self.apply(self.init_weights)
        self.atom_embedding = MLP(
            in_dim=max_z + 1, out_dims=[units], activation=None, use_bias=False
        )
        self.atom_embedding.apply(self.init_weights_uniform)
        self.normalizer = AtomScaling(verbose=False, max_z=max_z)
        self.max_z = max_z
        self.model_args = {
            "num_blocks": num_blocks,
            "units": units,
            "max_l": max_l,
            "max_n": max_n,
            "cutoff": cutoff,
            "max_z": max_z,
            "threebody_cutoff": threebody_cutoff,
        }

    def forward(
        self, input: Dict[str, paddle.Tensor], dataset_idx: int = -1
    ) -> paddle.Tensor:
        # Exact data from input_dictionary
        pos = input["atom_pos"]
        cell = input["cell"]
        pbc_offsets = input["pbc_offsets"].astype(dtype="float32")
        atom_attr = input["atom_attr"]
        edge_index = input["edge_index"].astype(dtype="int64")
        three_body_indices = input["three_body_indices"].astype(dtype="int64")
        num_three_body = input["num_three_body"]
        num_bonds = input["num_bonds"]
        num_triple_ij = input["num_triple_ij"]
        num_atoms = input["num_atoms"]
        num_graphs = input["num_graphs"]
        batch = input["batch"]

        # -------------------------------------------------------------#
        cumsum = paddle.cumsum(x=num_bonds, axis=0) - num_bonds
        index_bias = paddle.repeat_interleave(
            x=cumsum, repeats=num_three_body, axis=0
        ).unsqueeze(axis=-1)
        three_body_indices = three_body_indices + index_bias

        # === Refer to the implementation of M3GNet,        ===
        # === we should re-compute the following attributes ===
        # edge_length, edge_vector(optional), triple_edge_length, theta_jik
        atoms_batch = paddle.repeat_interleave(
            paddle.arange(0, num_atoms.numel()), repeats=num_atoms
        )
        edge_batch = atoms_batch[edge_index[0]]
        edge_vector = pos[edge_index[0]] - (
            pos[edge_index[1]]
            + paddle.einsum("bi, bij->bj", pbc_offsets, cell[edge_batch])
        )
        edge_length = paddle.linalg.norm(x=edge_vector, axis=1)
        vij = edge_vector[three_body_indices[:, 0].clone()]
        vik = edge_vector[three_body_indices[:, 1].clone()]
        rij = edge_length[three_body_indices[:, 0].clone()]
        rik = edge_length[three_body_indices[:, 1].clone()]
        cos_jik = paddle.sum(x=vij * vik, axis=1) / (rij * rik)
        # eps = 1e-7 avoid nan in paddle.acos function
        cos_jik = paddle.clip(x=cos_jik, min=-1.0 + 1e-07, max=1.0 - 1e-07)
        triple_edge_length = rik.view(-1)
        edge_length = edge_length.unsqueeze(axis=-1)
        atomic_numbers = atom_attr.squeeze(axis=1).astype(dtype="int64")

        # featurize
        atom_attr = self.atom_embedding(self.one_hot_atoms(atomic_numbers))
        edge_attr = self.rbf(edge_length.view(-1))
        edge_attr_zero = edge_attr  # e_ij^0
        edge_attr = self.edge_encoder(edge_attr)
        three_basis = self.sbf(triple_edge_length, paddle.acos(x=cos_jik))

        # Main Loop
        for idx, conv in enumerate(self.graph_conv):
            atom_attr, edge_attr = conv(
                atom_attr,
                edge_attr,
                edge_attr_zero,
                edge_index,
                three_basis,
                three_body_indices,
                edge_length,
                num_bonds,
                num_triple_ij,
                num_atoms,
            )

        energies_i = self.final(atom_attr).view(-1)  # [batch_size*num_atoms]
        energies_i = self.normalizer(energies_i, atomic_numbers)
        energies = scatter(energies_i, batch, dim=0, dim_size=num_graphs)

        return energies  # [batch_size]

    def init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            init_XavierUniform = paddle.nn.initializer.XavierUniform()
            init_XavierUniform(m.weight)

    def init_weights_uniform(self, m):
        if isinstance(m, paddle.nn.Linear):
            init_Uniform = paddle.nn.initializer.Uniform(low=-0.05, high=0.05)
            init_Uniform(m.weight)

    def one_hot_atoms(self, species):
        return paddle.nn.functional.one_hot(
            num_classes=self.max_z + 1, x=species
        ).astype(dtype="float32")

    def print(self):
        from prettytable import PrettyTable

        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not not parameter.stop_gradient:
                continue
            params = parameter.size
            table.add_row([name, params])
            total_params += params
        print(table)
        print(f"Total Trainable Params: {total_params}")

    def set_normalizer(self, normalizer: AtomScaling):
        self.normalizer = normalizer

    def get_model_args(self):
        return self.model_args
