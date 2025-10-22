from __future__ import annotations

import collections
from typing import TYPE_CHECKING

import numpy as np
import paddle
from chgnet.model.functions import GatedMLP
from chgnet.model.functions import find_activation
from pymatgen.core import Structure

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from chgnet.graph.crystalgraph import CrystalGraph


class CompositionModel(paddle.nn.Layer):
    """A simple FC model that takes in a chemical composition (no structure info)
    and outputs energy.
    """

    def __init__(
        self,
        *,
        atom_fea_dim: int = 64,
        activation: str = "silu",
        is_intensive: bool = True,
        max_num_elements: int = 94,
    ) -> None:
        """Initialize a CompositionModel."""
        super().__init__()
        self.is_intensive = is_intensive
        self.max_num_elements = max_num_elements
        self.fc1 = paddle.nn.Linear(
            in_features=max_num_elements, out_features=atom_fea_dim
        )
        self.activation = find_activation(activation)
        self.gated_mlp = GatedMLP(
            input_dim=atom_fea_dim,
            output_dim=atom_fea_dim,
            hidden_dim=atom_fea_dim,
            activation=activation,
        )
        self.fc2 = paddle.nn.Linear(in_features=atom_fea_dim, out_features=1)

    def _get_energy(self, composition_feas: paddle.Tensor) -> paddle.Tensor:
        """Predict the energy given composition encoding.

        Args:
            composition_feas: batched atom feature matrix of shape
                [batch_size, total_num_elements].

        Returns:
            prediction associated with each composition [batchsize].
        """
        composition_feas = self.activation(self.fc1(composition_feas))
        composition_feas += self.gated_mlp(composition_feas)
        return self.fc2(composition_feas).reshape([-1])

    def forward(self, graphs: list[CrystalGraph]) -> paddle.Tensor:
        """Get the energy of a list of CrystalGraphs as Tensor."""
        composition_feas = self._assemble_graphs(graphs)
        return self._get_energy(composition_feas)

    def _assemble_graphs(self, graphs: list[CrystalGraph]) -> paddle.Tensor:
        """Assemble a list of graphs into one-hot composition encodings.

        Args:
            graphs (list[CrystalGraph]): a list of CrystalGraphs

        Returns:
            assembled batch_graph that contains all information for model.
        """
        composition_feas = []
        for graph in graphs:
            composition_fea = paddle.bincount(
                x=graph.atomic_number - 1, minlength=self.max_num_elements
            )
            if self.is_intensive:
                n_atom = graph.atomic_number.shape[0]
                composition_fea = composition_fea / n_atom
            composition_feas.append(composition_fea)
        return paddle.stack(x=composition_feas, axis=0)


class AtomRef(paddle.nn.Layer):
    """A linear regression for elemental energy.
    From: https://github.com/materialsvirtuallab/m3gnet/.
    """

    def __init__(
        self, *, is_intensive: bool = True, max_num_elements: int = 94
    ) -> None:
        """Initialize an AtomRef model."""
        super().__init__()
        self.is_intensive = is_intensive
        self.max_num_elements = max_num_elements
        self.fc = paddle.nn.Linear(
            in_features=max_num_elements, out_features=1, bias_attr=False
        )
        self.fitted = False

    def forward(self, graphs: list[CrystalGraph]) -> paddle.Tensor:
        """Get the energy of a list of CrystalGraphs.

        Args:
            graphs (List(CrystalGraph)): a list of Crystal Graph to compute

        Returns:
            energy (tensor)
        """
        if not self.fitted:
            raise ValueError("composition model needs to be fitted first!")
        composition_feas = self._assemble_graphs(graphs)
        return self._get_energy(composition_feas)

    def _get_energy(self, composition_feas: paddle.Tensor) -> paddle.Tensor:
        """Predict the energy given composition encoding.

        Args:
            composition_feas: batched atom feature matrix of shape
                [batch_size, total_num_elements].

        Returns:
            prediction associated with each composition [batchsize].
        """
        return self.fc(composition_feas).flatten()

    # .view(-1)

    def fit(
        self,
        structures_or_graphs: Sequence[Structure | CrystalGraph],
        energies: Sequence[float],
    ) -> None:
        """Fit the model to a list of crystals and energies.

        Args:
            structures_or_graphs (list[Structure  |  CrystalGraph]): Any iterable of
                pymatgen structures and/or graphs.
            energies (list[float]): Target energies.
        """
        num_data = len(energies)
        composition_feas = paddle.zeros(shape=[num_data, self.max_num_elements])
        e = paddle.zeros(shape=[num_data])
        for index, (structure, energy) in enumerate(
            zip(structures_or_graphs, energies, strict=True)
        ):

            if isinstance(structure, Structure):
                atomic_number = paddle.to_tensor(
                    [site.specie.Z for site in structure], dtype="int32"
                )
            else:
                atomic_number = structure.atomic_number
            composition_fea = paddle.bincount(
                atomic_number - 1, minlength=self.max_num_elements
            )
            if self.is_intensive:
                composition_fea = composition_fea / atomic_number.shape[0]
            composition_feas[index, :] = composition_fea
            e[index] = energy

        # Use numpy for pinv
        self.feature_matrix = composition_feas.detach().numpy()
        self.energies = e.detach().numpy()
        state_dict = collections.OrderedDict()
        weight = (
            np.linalg.pinv(self.feature_matrix.T @ self.feature_matrix)
            @ self.feature_matrix.T
            @ self.energies
        )
        state_dict["weight"] = paddle.to_tensor(data=weight).view(94, 1)
        self.fc.set_state_dict(state_dict)
        self.fitted = True

    def _assemble_graphs(self, graphs: list[CrystalGraph]) -> paddle.Tensor:
        """Assemble a list of graphs into one-hot composition encodings
        Args:
            graphs (list[Tensor]): a list of CrystalGraphs
        Returns:
            assembled batch_graph that contains all information for model.
        """
        composition_feas = []
        for graph in graphs:
            if not paddle.all(graph.atomic_number >= 0):
                raise ValueError("atomic_number should be non-negative integers.")
            composition_fea = paddle.bincount(
                graph.atomic_number - 1, minlength=self.max_num_elements
            )
            if self.is_intensive:
                n_atom = graph.atomic_number.shape[0]
                composition_fea = composition_fea / n_atom
            composition_feas.append(composition_fea)
        return paddle.stack(composition_feas, axis=0).astype("float32")

    def get_site_energies(self, graphs: list[CrystalGraph]) -> list[paddle.Tensor]:
        """Predict the site energies given a list of CrystalGraphs.

        Args:
            graphs (List(CrystalGraph)): a list of Crystal Graph to compute

        Returns:
            a list of tensors corresponding to site energies of each graph [batchsize].
        """
        return [
            self.fc.state_dict()["weight"][0, graph.atomic_number - 1]
            for graph in graphs
        ]

    def initialize_from(self, dataset: str) -> None:
        """Initialize pre-fitted weights from a dataset."""
        if dataset in {"MPtrj", "MPtrj_e"}:
            self.initialize_from_MPtrj()
        elif dataset == "MPF":
            self.initialize_from_MPF()
        else:
            raise NotImplementedError(f"dataset={dataset!r} not supported yet")

    def initialize_from_MPtrj(self) -> None:
        """Initialize pre-fitted weights from MPtrj dataset."""
        state_dict = collections.OrderedDict()
        state_dict["weight"] = paddle.to_tensor(
            data=[
                -3.4431,
                -0.1279,
                -2.83,
                -3.4737,
                -7.4946,
                -8.2354,
                -8.1611,
                -8.3861,
                -5.7498,
                -0.0236,
                -1.7406,
                -1.6788,
                -4.2833,
                -6.2002,
                -6.1315,
                -5.8405,
                -3.8795,
                -0.0703,
                -1.5668,
                -3.4451,
                -7.0549,
                -9.1465,
                -9.2594,
                -9.3514,
                -8.9843,
                -8.0228,
                -6.4955,
                -5.6057,
                -3.4002,
                -0.9217,
                -3.2499,
                -4.9164,
                -4.781,
                -5.0191,
                -3.3316,
                0.513,
                -1.4043,
                -3.2175,
                -7.4994,
                -9.3816,
                -10.4386,
                -9.9539,
                -7.9555,
                -8.544,
                -7.3245,
                -5.2771,
                -1.9014,
                -0.4034,
                -2.6002,
                -4.0054,
                -4.1156,
                -3.9928,
                -2.7003,
                2.217,
                -1.9671,
                -3.718,
                -6.8133,
                -7.3502,
                -6.0712,
                -6.1699,
                -5.1471,
                -6.1925,
                -11.5829,
                -15.8841,
                -5.9994,
                -6.0798,
                -5.9513,
                -6.04,
                -5.9773,
                -2.5091,
                -6.0767,
                -10.6666,
                -11.8761,
                -11.8491,
                -10.7397,
                -9.61,
                -8.4755,
                -6.207,
                -3.0337,
                0.4726,
                -1.6425,
                -3.1295,
                -3.3328,
                -0.1221,
                -0.3448,
                -0.4364,
                -0.1661,
                -0.368,
                -4.1869,
                -8.4233,
                -10.0467,
                -12.0953,
                -12.5228,
                -14.253,
            ]
        ).view([94, 1])
        self.fc.set_state_dict(state_dict=state_dict)
        self.is_intensive = True
        self.fitted = True

    def initialize_from_MPF(self) -> None:
        """Initialize pre-fitted weights from MPF dataset."""
        state_dict = collections.OrderedDict()
        state_dict["weight"] = paddle.to_tensor(
            data=[
                -3.4654,
                -0.62617,
                -3.4622,
                -4.7758,
                -8.0362,
                -8.4038,
                -7.7681,
                -7.3892,
                -4.9472,
                -5.4833,
                -2.4783,
                -2.0202,
                -5.1548,
                -7.9121,
                -6.9135,
                -4.6228,
                -3.0155,
                -2.1285,
                -2.3174,
                -4.7595,
                -8.1742,
                -11.421,
                -8.9229,
                -8.4901,
                -8.1664,
                -6.5826,
                -5.2614,
                -4.4841,
                -3.2737,
                -1.3498,
                -3.6264,
                -4.6727,
                -4.1316,
                -3.6755,
                -2.803,
                6.4728,
                -2.2469,
                -4.251,
                -10.245,
                -11.666,
                -11.802,
                -8.6551,
                -9.3641,
                -7.5716,
                -5.699,
                -4.9716,
                -1.8871,
                -0.67951,
                -2.7488,
                -3.7945,
                -3.3883,
                -2.5588,
                -1.9621,
                9.9793,
                -2.5566,
                -4.8803,
                -8.8604,
                -9.0537,
                -7.9431,
                -8.1259,
                -6.3212,
                -8.3025,
                -12.289,
                -17.31,
                -7.5512,
                -8.1959,
                -8.3493,
                -7.2591,
                -8.417,
                -3.3873,
                -7.6823,
                -12.63,
                -13.626,
                -9.5299,
                -11.84,
                -9.799,
                -7.5561,
                -5.469,
                -2.6508,
                0.41746,
                -2.3255,
                -3.483,
                -3.1808,
                -0.016934,
                -0.036191,
                -0.010842,
                0.01317,
                -0.065371,
                -5.4892,
                -10.335,
                -11.13,
                -14.312,
                -14.7,
                -15.473,
            ]
        ).view([94, 1])
        self.fc.set_state_dict(state_dict=state_dict)
        self.is_intensive = False
        self.fitted = True

    def initialize_from_numpy(self, file_name: (str | Path)) -> None:
        """Initialize pre-fitted weights from numpy file."""
        atom_ref_np = np.load(file_name)
        state_dict = collections.OrderedDict()
        state_dict["weight"] = paddle.to_tensor(data=atom_ref_np).view([1, 94])
        self.fc.set_state_dict(state_dict=state_dict)
        self.is_intensive = False
        self.fitted = True
