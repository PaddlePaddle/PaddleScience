from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Literal

import paddle
from chgnet.graph import CrystalGraph
from chgnet.graph import CrystalGraphConverter
from chgnet.graph.crystalgraph import DTYPE
from chgnet.model.composition_model import AtomRef
from chgnet.model.encoders import AngleEncoder
from chgnet.model.encoders import AtomEmbedding
from chgnet.model.encoders import BondEncoder
from chgnet.model.functions import MLP
from chgnet.model.functions import GatedMLP
from chgnet.model.functions import find_normalization
from chgnet.model.layers import AngleUpdate
from chgnet.model.layers import AtomConv
from chgnet.model.layers import BondConv
from chgnet.model.layers import GraphAttentionReadOut
from chgnet.model.layers import GraphPooling
from chgnet.utils import determine_device
from pymatgen.core import Structure

if TYPE_CHECKING:
    from chgnet import PredTask
    from typing_extensions import Self
module_dir = os.path.dirname(os.path.abspath(__file__))


class CHGNet(paddle.nn.Layer):
    """Crystal Hamiltonian Graph neural Network
    A model that takes in a crystal graph and output energy, force, magmom, stress.
    """

    def __init__(
        self,
        *,
        atom_fea_dim: int = 64,
        bond_fea_dim: int = 64,
        angle_fea_dim: int = 64,
        composition_model: (str | paddle.nn.Layer) = "MPtrj",
        num_radial: int = 31,
        num_angular: int = 31,
        n_conv: int = 4,
        atom_conv_hidden_dim: (Sequence[int] | int) = 64,
        update_bond: bool = True,
        bond_conv_hidden_dim: (Sequence[int] | int) = 64,
        update_angle: bool = True,
        angle_layer_hidden_dim: (Sequence[int] | int) = 0,
        conv_dropout: float = 0,
        read_out: str = "ave",
        mlp_hidden_dims: (Sequence[int] | int) = (64, 64, 64),
        mlp_dropout: float = 0,
        mlp_first: bool = True,
        is_intensive: bool = True,
        non_linearity: Literal["silu", "relu", "tanh", "gelu"] = "silu",
        atom_graph_cutoff: float = 6,
        bond_graph_cutoff: float = 3,
        graph_converter_algorithm: Literal["legacy", "fast"] = "fast",
        cutoff_coeff: int = 8,
        learnable_rbf: bool = True,
        gMLP_norm: (str | None) = "layer",
        readout_norm: (str | None) = "layer",
        version: (str | None) = None,
        **kwargs,
    ) -> None:
        """Initialize CHGNet.

        Args:
            atom_fea_dim (int): atom feature vector embedding dimension.
                Default = 64
            bond_fea_dim (int): bond feature vector embedding dimension.
                Default = 64
            angle_fea_dim (int): angle feature vector embedding dimension.
                Default = 64
            bond_fea_dim (int): angle feature vector embedding dimension.
                Default = 64
            composition_model (nn.Module, optional): attach a composition model to
                predict energy or initialize a pretrained linear regression (AtomRef).
                The default 'MPtrj' is the atom reference energy linear regression
                trained on all Materials Project relaxation trajectories
                Default = 'MPtrj'
            num_radial (int): number of radial basis used in bond basis expansion.
                Default = 9
            num_angular (int): number of angular basis used in angle basis expansion.
                Default = 9
            n_conv (int): number of interaction blocks.
                Default = 4
                Note: last interaction block contain only an atom_conv layer
            atom_conv_hidden_dim (List or int): hidden dimensions of
                atom convolution layers.
                Default = 64
            update_bond (bool): whether to use bond_conv_layer in bond graph to
                update bond embeddings
                Default = True.
            bond_conv_hidden_dim (List or int): hidden dimensions of
                bond convolution layers.
                Default = 64
            update_angle (bool): whether to use angle_update_layer to
                update angle embeddings.
                Default = True
            angle_layer_hidden_dim (List or int): hidden dimensions of angle layers.
                Default = 0
            conv_dropout (float): dropout rate in all conv_layers.
                Default = 0
            read_out (str): method for pooling layer, 'ave' for standard
                average pooling, 'attn' for multi-head attention.
                Default = "ave"
            mlp_hidden_dims (int or list): readout multilayer perceptron
                hidden dimensions.
                Default = [64, 64]
            mlp_dropout (float): dropout rate in readout MLP.
                Default = 0.
            is_intensive (bool): whether the energy training label is intensive
                i.e. energy per atom.
                Default = True
            non_linearity ('silu' | 'relu' | 'tanh' | 'gelu'): The name of the
                activation function to use in the gated MLP.
                Default = "silu".
            mlp_first (bool): whether to apply mlp first then pooling.
                if set to True, then CHGNet is essentially calculating energy for each
                atom, them sum them up, this is used for the pretrained model
                Default = True
            atom_graph_cutoff (float): cutoff radius (A) in creating atom_graph,
                this need to be consistent with the value in training dataloader
                Default = 5
            bond_graph_cutoff (float): cutoff radius (A) in creating bond_graph,
                this need to be consistent with value in training dataloader
                Default = 3
            graph_converter_algorithm ('legacy' | 'fast'): algorithm to use
                for converting pymatgen.core.Structure to CrystalGraph.
                'legacy': python implementation of graph creation
                'fast': C implementation of graph creation, this is faster,
                    but will need the cygraph.c file correctly compiled from pip install
                default = 'fast'
            cutoff_coeff (float): cutoff strength used in graph smooth cutoff function.
                the smaller this coeff is, the smoother the basis is
                Default = 5
            learnable_rbf (bool): whether to set the frequencies in rbf and Fourier
                basis functions learnable.
                Default = True
            gMLP_norm (str): normalization layer to use in gate-MLP
                Default = 'layer'
            readout_norm (str): normalization layer to use before readout layer
                Default = 'layer'
            version (str): Pretrained checkpoint version.
            **kwargs: Additional keyword arguments
        """
        self.model_args = {
            key: val
            for key, val in locals().items()
            if key not in {"self", "__class__", "kwargs"}
        }
        self.model_args.update(kwargs)
        if version:
            self.model_args["version"] = version
        super().__init__()
        self.atom_fea_dim = atom_fea_dim
        self.bond_fea_dim = bond_fea_dim
        self.is_intensive = is_intensive
        self.n_conv = n_conv
        if isinstance(composition_model, paddle.nn.Layer):
            self.composition_model = composition_model
        elif isinstance(composition_model, str):
            self.composition_model = AtomRef(is_intensive=is_intensive)
            # import pdb
            # pdb.set_trace()
            self.composition_model.initialize_from(composition_model)
        else:
            self.composition_model = None
        if self.composition_model is not None:
            for param in self.composition_model.parameters():
                param.stop_gradient = not False
        self.graph_converter = CrystalGraphConverter(
            atom_graph_cutoff=atom_graph_cutoff,
            bond_graph_cutoff=bond_graph_cutoff,
            algorithm=graph_converter_algorithm,
            verbose=kwargs.pop("converter_verbose", False),
        )
        self.atom_embedding = AtomEmbedding(atom_feature_dim=atom_fea_dim)
        self.bond_basis_expansion = BondEncoder(
            atom_graph_cutoff=atom_graph_cutoff,
            bond_graph_cutoff=bond_graph_cutoff,
            num_radial=num_radial,
            cutoff_coeff=cutoff_coeff,
            learnable=learnable_rbf,
        )
        self.bond_embedding = paddle.nn.Linear(
            in_features=num_radial, out_features=bond_fea_dim, bias_attr=False
        )
        self.bond_weights_ag = paddle.nn.Linear(
            in_features=num_radial, out_features=atom_fea_dim, bias_attr=False
        )
        self.bond_weights_bg = paddle.nn.Linear(
            in_features=num_radial, out_features=bond_fea_dim, bias_attr=False
        )
        self.angle_basis_expansion = AngleEncoder(
            num_angular=num_angular, learnable=learnable_rbf
        )
        self.angle_embedding = paddle.nn.Linear(
            in_features=num_angular, out_features=angle_fea_dim, bias_attr=False
        )
        conv_norm = kwargs.pop("conv_norm", None)
        mlp_out_bias = kwargs.pop("mlp_out_bias", False)
        atom_graph_layers = [
            AtomConv(
                atom_fea_dim=atom_fea_dim,
                bond_fea_dim=bond_fea_dim,
                hidden_dim=atom_conv_hidden_dim,
                dropout=conv_dropout,
                activation=non_linearity,
                norm=conv_norm,
                gMLP_norm=gMLP_norm,
                use_mlp_out=True,
                mlp_out_bias=mlp_out_bias,
                resnet=True,
            )
            for _ in range(n_conv)
        ]
        self.atom_conv_layers = paddle.nn.LayerList(sublayers=atom_graph_layers)
        if update_bond:
            bond_graph_layers = [
                BondConv(
                    atom_fea_dim=atom_fea_dim,
                    bond_fea_dim=bond_fea_dim,
                    angle_fea_dim=angle_fea_dim,
                    hidden_dim=bond_conv_hidden_dim,
                    dropout=conv_dropout,
                    activation=non_linearity,
                    norm=conv_norm,
                    gMLP_norm=gMLP_norm,
                    use_mlp_out=True,
                    mlp_out_bias=mlp_out_bias,
                    resnet=True,
                )
                for _ in range(n_conv - 1)
            ]
            self.bond_conv_layers = paddle.nn.LayerList(sublayers=bond_graph_layers)
        else:
            self.bond_conv_layers = [None for _ in range(n_conv - 1)]
        if update_angle:
            angle_layers = [
                AngleUpdate(
                    atom_fea_dim=atom_fea_dim,
                    bond_fea_dim=bond_fea_dim,
                    angle_fea_dim=angle_fea_dim,
                    hidden_dim=angle_layer_hidden_dim,
                    dropout=conv_dropout,
                    activation=non_linearity,
                    norm=conv_norm,
                    gMLP_norm=gMLP_norm,
                    resnet=True,
                )
                for _ in range(n_conv - 1)
            ]
            self.angle_layers = paddle.nn.LayerList(sublayers=angle_layers)
        else:
            self.angle_layers = [None for _ in range(n_conv - 1)]
        self.site_wise = paddle.nn.Linear(in_features=atom_fea_dim, out_features=1)
        self.readout_norm = find_normalization(readout_norm, dim=atom_fea_dim)
        self.mlp_first = mlp_first
        if mlp_first:
            self.read_out_type = "sum"
            input_dim = atom_fea_dim
            self.pooling = GraphPooling(average=False)
        elif read_out in {"attn", "weighted"}:
            self.read_out_type = "attn"
            num_heads = kwargs.pop("num_heads", 3)
            self.pooling = GraphAttentionReadOut(
                atom_fea_dim, num_head=num_heads, average=True
            )
            input_dim = atom_fea_dim * num_heads
        else:
            self.read_out_type = "ave"
            input_dim = atom_fea_dim
            self.pooling = GraphPooling(average=True)
        if kwargs.pop("final_mlp", "MLP") in {"normal", "MLP"}:
            self.mlp = MLP(
                input_dim=input_dim,
                hidden_dim=mlp_hidden_dims,
                output_dim=1,
                dropout=mlp_dropout,
                activation=non_linearity,
            )
        else:
            self.mlp = paddle.nn.Sequential(
                GatedMLP(
                    input_dim=input_dim,
                    hidden_dim=mlp_hidden_dims,
                    output_dim=mlp_hidden_dims[-1],
                    dropout=mlp_dropout,
                    norm=gMLP_norm,
                    activation=non_linearity,
                ),
                paddle.nn.Linear(in_features=mlp_hidden_dims[-1], out_features=1),
            )
        version_str = f" v{version}" if version else ""
        print(f"CHGNet{version_str} initialized with {self.n_params:,} parameters")

    @property
    def version(self) -> (str | None):
        """Return the version of the loaded checkpoint."""
        return self.model_args.get("version")

    @property
    def n_params(self) -> int:
        """Return the number of parameters in the model."""
        return sum(p.size for p in self.parameters())

    def forward(
        self,
        graphs: Sequence[CrystalGraph],
        *,
        task: PredTask = "e",
        return_site_energies: bool = False,
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
    ) -> dict[str, paddle.Tensor]:
        """Get prediction associated with input graphs
        Args:
            graphs (List): a list of CrystalGraphs
            task (str): the prediction task. One of 'e', 'em', 'ef', 'efs', 'efsm'.
                Default = 'e'
            return_site_energies (bool): whether to return per-site energies,
                only available if self.mlp_first == True
                Default = False
            return_atom_feas (bool): whether to return the atom features before last
                conv layer.
                Default = False
            return_crystal_feas (bool): whether to return crystal feature.
                Default = False
        Returns:
            model output (dict).
        """
        comp_energy = (
            0 if self.composition_model is None else self.composition_model(graphs)
        )
        batched_graph = BatchedGraph.from_graphs(
            graphs,
            bond_basis_expansion=self.bond_basis_expansion,
            angle_basis_expansion=self.angle_basis_expansion,
            compute_stress="s" in task,
        )
        prediction = self._compute(
            batched_graph,
            compute_force="f" in task,
            compute_stress="s" in task,
            compute_magmom="m" in task,
            return_site_energies=return_site_energies,
            return_atom_feas=return_atom_feas,
            return_crystal_feas=return_crystal_feas,
        )
        prediction["e"] += comp_energy
        if return_site_energies and self.composition_model is not None:
            site_energy_shifts = self.composition_model.get_site_energies(graphs)
            prediction["site_energies"] = [
                (i + j)
                for i, j in zip(
                    prediction["site_energies"], site_energy_shifts, strict=True
                )
            ]
        return prediction

    def _compute(
        self,
        g: BatchedGraph,
        *,
        compute_force: bool = False,
        compute_stress: bool = False,
        compute_magmom: bool = False,
        return_site_energies: bool = False,
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
    ) -> dict:
        """Get Energy, Force, Stress, Magmom associated with input graphs
        force = - d(Energy)/d(atom_positions)
        stress = 1/V * d(Energy)/d(strain).

        Args:
            g (BatchedGraph): batched graph
            compute_force (bool): whether to compute force.
                Default = False
            compute_stress (bool): whether to compute stress.
                Default = False
            compute_magmom (bool): whether to compute magmom.
                Default = False
            return_site_energies (bool): whether to return per-site energies,
                only available if self.mlp_first == True
                Default = False
            return_atom_feas (bool): whether to return atom features.
                Default = False
            return_crystal_feas (bool): whether to return crystal features.
                Default = False

        Returns:
            prediction (dict): containing the fields:
                e (Tensor) : energy of structures [batch_size, 1]
                f (Tensor) : force on atoms [num_batch_atoms, 3]
                s (Tensor) : stress of structure [3 * batch_size, 3]
                m (Tensor) : magnetic moments of sites [num_batch_atoms, 3]
        """
        prediction = {}
        atoms_per_graph = paddle.bincount(x=g.atom_owners)
        prediction["atoms_per_graph"] = atoms_per_graph
        atom_feas = self.atom_embedding(g.atomic_numbers - 1)
        bond_feas = self.bond_embedding(g.bond_bases_ag)
        bond_weights_ag = self.bond_weights_ag(g.bond_bases_ag)
        bond_weights_bg = self.bond_weights_bg(g.bond_bases_bg)
        if len(g.angle_bases) != 0:
            angle_feas = self.angle_embedding(g.angle_bases)
        for idx, (atom_layer, bond_layer, angle_layer) in enumerate(
            zip(
                self.atom_conv_layers[:-1],
                self.bond_conv_layers,
                self.angle_layers,
                strict=False,
            )
        ):
            atom_feas = atom_layer(
                atom_feas=atom_feas,
                bond_feas=bond_feas,
                bond_weights=bond_weights_ag,
                atom_graph=g.batched_atom_graph,
                directed2undirected=g.directed2undirected,
            )
            if len(g.angle_bases) != 0 and bond_layer is not None:
                bond_feas = bond_layer(
                    atom_feas=atom_feas,
                    bond_feas=bond_feas,
                    bond_weights=bond_weights_bg,
                    angle_feas=angle_feas,
                    bond_graph=g.batched_bond_graph,
                )
                if angle_layer is not None:
                    angle_feas = angle_layer(
                        atom_feas=atom_feas,
                        bond_feas=bond_feas,
                        angle_feas=angle_feas,
                        bond_graph=g.batched_bond_graph,
                    )
            if idx == self.n_conv - 2:
                if return_atom_feas:
                    prediction["atom_fea"] = paddle.split(
                        x=atom_feas, num_or_sections=atoms_per_graph.tolist()
                    )
                if compute_magmom:
                    magmom = paddle.abs(x=self.site_wise(atom_feas))
                    prediction["m"] = list(
                        paddle.split(
                            x=magmom.reshape([-1]),
                            num_or_sections=atoms_per_graph.tolist(),
                        )
                    )
        atom_feas = self.atom_conv_layers[-1](
            atom_feas=atom_feas,
            bond_feas=bond_feas,
            bond_weights=bond_weights_ag,
            atom_graph=g.batched_atom_graph,
            directed2undirected=g.directed2undirected,
        )
        if self.readout_norm is not None:
            atom_feas = self.readout_norm(atom_feas)
        if self.mlp_first:
            energies = self.mlp(atom_feas)
            energy = self.pooling(energies, g.atom_owners).reshape([-1])
            if return_site_energies:
                prediction["site_energies"] = paddle.split(
                    x=energies.squeeze(axis=1), num_or_sections=atoms_per_graph.tolist()
                )
            if return_crystal_feas:
                prediction["crystal_fea"] = self.pooling(atom_feas, g.atom_owners)
        else:
            crystal_feas = self.pooling(atom_feas, g.atom_owners)
            energy = self.mlp(crystal_feas).reshape([-1]) * atoms_per_graph
            if return_crystal_feas:
                prediction["crystal_fea"] = crystal_feas

        if compute_force:
            force = paddle.grad(
                outputs=energy.sum(),
                inputs=g.atom_positions,
                create_graph=False,
                retain_graph=True,
            )
            prediction["f"] = [(-1 * force_dim) for force_dim in force]
        if compute_stress:
            stress = paddle.grad(
                outputs=energy.sum(),
                inputs=g.strains,
                create_graph=False,
                retain_graph=True,
            )
            scale = 1 / g.volumes * 160.21766208
            stress = [(i * j) for i, j in zip(stress, scale, strict=False)]
            prediction["s"] = stress

        if self.is_intensive:
            energy /= atoms_per_graph.cast("float32")
        prediction["e"] = energy
        return prediction

    def predict_structure(
        self,
        structure: (Structure | Sequence[Structure]),
        *,
        task: PredTask = "efsm",
        return_site_energies: bool = False,
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
        batch_size: int = 16,
    ) -> (dict[str, paddle.Tensor] | list[dict[str, paddle.Tensor]]):
        """Predict from pymatgen.core.Structure.

        Args:
            structure (Structure | Sequence[Structure]): structure or a list of
                structures to predict.
            task (str): can be 'e' 'ef', 'em', 'efs', 'efsm'
                Default = "efsm"
            return_site_energies (bool): whether to return per-site energies.
                Default = False
            return_atom_feas (bool): whether to return atom features.
                Default = False
            return_crystal_feas (bool): whether to return crystal features.
                Default = False
            batch_size (int): batch_size for predict structures.
                Default = 16

        Returns:
            prediction (dict): dict or list of dict containing the fields:
                e (Tensor) : energy of structures float in eV/atom
                f (Tensor) : force on atoms [num_atoms, 3] in eV/A
                s (Tensor) : stress of structure [3, 3] in GPa
                m (Tensor) : magnetic moments of sites [num_atoms, 3] in Bohr
                    magneton mu_B
        """
        if self.graph_converter is None:
            raise ValueError("graph_converter cannot be None!")
        structures = [structure] if isinstance(structure, Structure) else structure
        graphs = [self.graph_converter(struct) for struct in structures]
        return self.predict_graph(
            graphs,
            task=task,
            return_site_energies=return_site_energies,
            return_atom_feas=return_atom_feas,
            return_crystal_feas=return_crystal_feas,
            batch_size=batch_size,
        )

    def predict_graph(
        self,
        graph: (CrystalGraph | Sequence[CrystalGraph]),
        *,
        task: PredTask = "efsm",
        return_site_energies: bool = False,
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
        batch_size: int = 16,
    ) -> (dict[str, paddle.Tensor] | list[dict[str, paddle.Tensor]]):
        """Predict from CrustalGraph.

        Args:
            graph (CrystalGraph | Sequence[CrystalGraph]): CrystalGraph(s) to predict.
            task (str): can be 'e' 'ef', 'em', 'efs', 'efsm'
                Default = "efsm"
            return_site_energies (bool): whether to return per-site energies.
                Default = False
            return_atom_feas (bool): whether to return atom features.
                Default = False
            return_crystal_feas (bool): whether to return crystal features.
                Default = False
            batch_size (int): batch_size for predict structures.
                Default = 16

        Returns:
            prediction (dict): dict or list of dict containing the fields:
                e (Tensor) : energy of structures float in eV/atom
                f (Tensor) : force on atoms [num_atoms, 3] in eV/A
                s (Tensor) : stress of structure [3, 3] in GPa
                m (Tensor) : magnetic moments of sites [num_atoms, 3] in Bohr
                    magneton mu_B
        """
        if not isinstance(graph, CrystalGraph | Sequence):
            raise TypeError(
                f"type(graph)={type(graph)!r} must be CrystalGraph or list of CrystalGraphs"
            )
        # next(iter(self.parameters())).place
        graphs = [graph] if isinstance(graph, CrystalGraph) else graph
        self.eval()
        predictions: list[dict[str, paddle.Tensor]] = [{} for _ in range(len(graphs))]
        n_steps = math.ceil(len(graphs) / batch_size)
        for step in range(n_steps):
            prediction = self.forward(
                [g for g in graphs[batch_size * step : batch_size * (step + 1)]],
                task=task,
                return_site_energies=return_site_energies,
                return_atom_feas=return_atom_feas,
                return_crystal_feas=return_crystal_feas,
            )
            for key in {
                "e",
                "f",
                "s",
                "m",
                "site_energies",
                "atom_fea",
                "crystal_fea",
            } & {*prediction}:
                for idx, tensor in enumerate(prediction[key]):
                    predictions[step * batch_size + idx][key] = (
                        tensor.cpu().detach().numpy()
                    )
        return predictions[0] if len(graphs) == 1 else predictions

    def as_dict(self) -> dict:
        """Return the CHGNet weights and args in a dictionary."""
        return {"state_dict": self.state_dict(), "model_args": self.model_args}

    def todict(self) -> dict:
        """Needed for ASE JSON serialization when saving CHGNet potential to
        trajectory file (https://github.com/CederGroupHub/chgnet/issues/48).
        """
        return {"model_name": type(self).__name__, "model_args": self.model_args}

    @classmethod
    def from_dict(cls, dct: dict, **kwargs) -> Self:
        """Build a CHGNet from a saved dictionary."""
        chgnet = cls(**dct["model_args"], **kwargs)
        chgnet.set_state_dict(state_dict=dct["state_dict"])
        return chgnet

    @classmethod
    def from_file(cls, path: str, **kwargs) -> Self:
        """Build a CHGNet from a saved file."""
        state = paddle.load(path=str(path))
        return cls.from_dict(state["model"], **kwargs)

    @classmethod
    def load(
        cls,
        *,
        model_name: str = "0.3.0",
        use_device: (str | None) = None,
        check_cuda_mem: bool = False,
        verbose: bool = True,
    ) -> Self:
        """Load pretrained CHGNet model.

        Args:
            model_name (str, optional):
                Default = "0.3.0".
            use_device (str, optional): The device to be used for predictions,
                either "cpu", "cuda", or "mps". If not specified, the default device is
                automatically selected based on the available options.
                Default = None
            check_cuda_mem (bool): Whether to use cuda with most available memory
                Default = False
            verbose (bool): whether to print model device information
                Default = True
        Raises:
            ValueError: On unknown model_name.
        """
        checkpoint_path = {
            "0.3.0": "../pretrained/0.3.0/chgnet_0.3.0_paddle.pdparams",
            "0.2.0": "../pretrained/0.2.0/chgnet_0.2.0_e30f77s348m32.pth.tar",
        }.get(model_name)
        if checkpoint_path is None:
            raise ValueError(f"Unknown model_name={model_name!r}")
        model = cls.from_file(
            os.path.join(module_dir, checkpoint_path),
            mlp_out_bias=model_name == "0.2.0",
            version=model_name,
        )
        device = determine_device(use_device=use_device, check_cuda_mem=check_cuda_mem)
        if verbose:
            print(f"CHGNet will run on {device}")
        return model


@dataclass
class BatchedGraph:
    """Batched crystal graph for parallel computing.

    Attributes:
        atomic_numbers (Tensor): atomic numbers vector
            [num_batch_atoms]
        bond_bases_ag (Tensor): bond bases vector for atom_graph
            [num_batch_bonds_ag, num_radial]
        bond_bases_bg (Tensor): bond bases vector for atom_graph
            [num_batch_bonds_bg, num_radial]
        angle_bases (Tensor): angle bases vector
            [num_batch_angles, num_angular]
        batched_atom_graph (Tensor) : batched atom graph adjacency list
            [num_batch_bonds, 2]
        batched_bond_graph (Tensor) : bond graph adjacency list
            [num_batch_angles, 3]
        atom_owners (Tensor): graph indices for each atom, used aggregate batched
            graph back to single graph
            [num_batch_atoms]
        directed2undirected (Tensor): the utility tensor used to quickly
            map directed edges to undirected edges in graph
            [num_directed]
        atom_positions (list[Tensor]): cartesian coordinates of the atoms
            from structures
            [[num_atoms_1, 3], [num_atoms_2, 3], ...]
        strains (list[Tensor]): a list of strains that's initialized to be zeros
            [[3, 3], [3, 3], ...]
        volumes (Tensor): the volume of each structure in the batch
            [batch_size]
    """

    atomic_numbers: paddle.Tensor
    bond_bases_ag: paddle.Tensor
    bond_bases_bg: paddle.Tensor
    angle_bases: paddle.Tensor
    batched_atom_graph: paddle.Tensor
    batched_bond_graph: paddle.Tensor
    atom_owners: paddle.Tensor
    directed2undirected: paddle.Tensor
    atom_positions: Sequence[paddle.Tensor]
    strains: Sequence[paddle.Tensor]
    volumes: Sequence[paddle.Tensor] | paddle.Tensor

    @classmethod
    def from_graphs(
        cls,
        graphs: Sequence[CrystalGraph],
        bond_basis_expansion: paddle.nn.Layer,
        angle_basis_expansion: paddle.nn.Layer,
        *,
        compute_stress: bool = False,
    ) -> Self:
        """Featurize and assemble a list of graphs.

        Args:
            graphs (list[Tensor]): a list of CrystalGraphs
            bond_basis_expansion (nn.Module): bond basis expansion layer in CHGNet
            angle_basis_expansion (nn.Module): angle basis expansion layer in CHGNet
            compute_stress (bool): whether to compute stress. Default = False

        Returns:
            BatchedGraph: assembled graphs ready for batched CHGNet forward pass
        """
        atomic_numbers, atom_positions = [], []
        strains, volumes = [], []
        bond_bases_ag, bond_bases_bg, angle_bases = [], [], []
        batched_atom_graph, batched_bond_graph = [], []
        directed2undirected = []
        atom_owners = []
        atom_offset_idx = n_undirected = 0
        for graph_idx, graph in enumerate(graphs):
            n_atom = graph.atomic_number.shape[0]
            atomic_numbers.append(graph.atomic_number)
            if compute_stress:
                strain = paddle.to_tensor(
                    paddle.zeros([3, 3], dtype="float32"), stop_gradient=False
                )
                lattice = paddle.matmul(
                    graph.lattice, paddle.eye(3, dtype="float32") + strain
                )
            else:
                strain = None
                lattice = graph.lattice
            volumes.append(
                paddle.dot(
                    x=lattice[0], y=paddle.cross(x=lattice[1], y=lattice[2], axis=-1)
                )
            )
            strains.append(strain)
            atom_cart_coords = graph.atom_frac_coord @ lattice
            if graph.atom_graph.dim() == 1:
                graph.atom_graph = graph.atom_graph.reshape(0, 2)
            bond_basis_ag, bond_basis_bg, bond_vectors = bond_basis_expansion(
                center=atom_cart_coords[graph.atom_graph[:, 0]],
                neighbor=atom_cart_coords[graph.atom_graph[:, 1]],
                undirected2directed=graph.undirected2directed,
                image=graph.neighbor_image,
                lattice=lattice,
            )
            atom_positions.append(atom_cart_coords)
            bond_bases_ag.append(bond_basis_ag)
            bond_bases_bg.append(bond_basis_bg)
            batched_atom_graph.append(graph.atom_graph + atom_offset_idx)
            directed2undirected.append(graph.directed2undirected + n_undirected)
            if len(graph.bond_graph) != 0:
                bond_vecs_i = paddle.gather(
                    x=bond_vectors, axis=0, index=graph.bond_graph[:, 2]
                )
                bond_vecs_j = paddle.gather(
                    x=bond_vectors, axis=0, index=graph.bond_graph[:, 4]
                )
                angle_basis = angle_basis_expansion(bond_vecs_i, bond_vecs_j)
                angle_bases.append(angle_basis)
                bond_graph = paddle.zeros([graph.bond_graph.shape[0], 3])
                # graph.bond_graph.new_zeros()
                bond_graph[:, 0] = graph.bond_graph[:, 0] + atom_offset_idx
                bond_graph[:, 1] = graph.bond_graph[:, 1] + n_undirected
                bond_graph[:, 2] = graph.bond_graph[:, 3] + n_undirected
                batched_bond_graph.append(bond_graph)
            out_0 = paddle.ones(shape=n_atom)
            out_0.stop_gradient = not False
            atom_owners.append(out_0 * graph_idx)
            atom_offset_idx += n_atom
            n_undirected += len(bond_basis_ag)
        atomic_numbers = paddle.concat(x=atomic_numbers, axis=0)
        bond_bases_ag = paddle.concat(x=bond_bases_ag, axis=0)
        bond_bases_bg = paddle.concat(x=bond_bases_bg, axis=0)
        angle_bases = (
            paddle.concat(x=angle_bases, axis=0)
            if len(angle_bases) != 0
            else paddle.to_tensor(data=[])
        )
        batched_atom_graph = paddle.concat(x=batched_atom_graph, axis=0)
        if batched_bond_graph != []:
            batched_bond_graph = paddle.concat(x=batched_bond_graph, axis=0)
        else:
            batched_bond_graph = paddle.to_tensor(data=[])
        atom_owners = paddle.concat(x=atom_owners, axis=0).astype("int32")
        directed2undirected = paddle.concat(x=directed2undirected, axis=0)
        volumes = paddle.to_tensor(
            data=volumes, dtype=DTYPE, place=atomic_numbers.place
        )
        return cls(
            atomic_numbers=atomic_numbers,
            bond_bases_ag=bond_bases_ag,
            bond_bases_bg=bond_bases_bg,
            angle_bases=angle_bases,
            batched_atom_graph=batched_atom_graph,
            batched_bond_graph=batched_bond_graph,
            atom_owners=atom_owners,
            directed2undirected=directed2undirected,
            atom_positions=atom_positions,
            strains=strains,
            volumes=volumes,
        )
