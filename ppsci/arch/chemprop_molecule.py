from __future__ import annotations

import math
from functools import reduce
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import paddle
from omegaconf import DictConfig

try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass


from ppsci.arch.chemprop_molecule_utils import BatchMolGraph
from ppsci.arch.chemprop_molecule_utils import TrainArgs
from ppsci.arch.chemprop_molecule_utils import get_activation_function
from ppsci.arch.chemprop_molecule_utils import get_atom_fdim
from ppsci.arch.chemprop_molecule_utils import get_bond_fdim
from ppsci.arch.chemprop_molecule_utils import index_select_ND


class MPNEncoder(paddle.nn.Layer):
    """Message Passing Neural Network encoder for molecules.

    Encodes atom/bond features via iterative message passing.

    Args:
        args: Model arguments in a `TrainArgs` instance.
        atom_fdim: Atom feature vector dimension.
        bond_fdim: Bond feature vector dimension.
        hidden_size: Hidden layer dimension.
        bias: Whether to add bias to linear layers.
        depth: Number of message passing steps.
    """

    def __init__(
        self,
        args: TrainArgs,
        atom_fdim: int,
        bond_fdim: int,
        hidden_size: int = None,
        bias: bool = None,
        depth: int = None,
    ):
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.atom_messages = args.atom_messages
        self.hidden_size = hidden_size or args.hidden_size
        self.bias = bias or args.bias
        self.depth = depth or args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.device = args.device
        self.aggregation = args.aggregation
        self.aggregation_norm = args.aggregation_norm

        # Dropout
        self.dropout_layer = paddle.nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.zeros(shape=self.hidden_size), trainable=False
        )

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = paddle.nn.Linear(
            in_features=input_dim, out_features=self.hidden_size, bias_attr=self.bias
        )
        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size
        self.W_h = paddle.nn.Linear(
            in_features=w_h_input_size,
            out_features=self.hidden_size,
            bias_attr=self.bias,
        )
        self.W_o = paddle.nn.Linear(
            in_features=self.atom_fdim + self.hidden_size, out_features=self.hidden_size
        )
        if args.atom_descriptors == "descriptor":
            self.atom_descriptors_size = args.atom_descriptors_size
            self.atom_descriptors_layer = paddle.nn.Linear(
                in_features=self.hidden_size + self.atom_descriptors_size,
                out_features=self.hidden_size + self.atom_descriptors_size,
            )

    def forward(
        self, mol_graph, atom_descriptors_batch: List[np.ndarray] = None
    ) -> paddle.float32:
        """Encode a batch of molecular graphs.

        Args:
            mol_graph: A `BatchMolGraph` representing a batch of molecular graphs.
            atom_descriptors_batch: Optional list of numpy arrays with additional atomic descriptors.

        Returns:
            `paddle.Tensor`: Shape `(num_molecules, hidden_size)` with the encoding of each molecule.

        Raises:
            ValueError: If descriptor count does not match the number of atoms.
        """
        if atom_descriptors_batch is not None:
            atom_descriptors_batch = [
                np.zeros([1, tuple(atom_descriptors_batch[0].shape)[1]])
            ] + atom_descriptors_batch
            atom_descriptors_batch = (
                paddle.to_tensor(data=np.concatenate(atom_descriptors_batch, axis=0))
                .astype(dtype="float32")
                .to(self.device)
            )

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph

        if self.atom_messages:
            a2a = mol_graph.get_a2a().to(self.device)
        if self.atom_messages:
            input = self.W_i(f_atoms)
        else:
            input = self.W_i(f_bonds)
        message = self.act_func(input)

        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2
            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)
                nei_f_bonds = index_select_ND(f_bonds, a2b)
                nei_message = paddle.concat(x=(nei_a_message, nei_f_bonds), axis=2)
                message = nei_message.sum(axis=1)
            else:
                nei_a_message = index_select_ND(message, a2b)
                a_message = nei_a_message.sum(axis=1)
                rev_message = message[b2revb]
                message = a_message[b2a] - rev_message
            message = self.W_h(message)
            message = self.act_func(input + message)
            message = self.dropout_layer(message)
        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)
        a_message = nei_a_message.sum(axis=1)
        a_input = paddle.concat(x=[f_atoms, a_message], axis=1)
        atom_hiddens = self.act_func(self.W_o(a_input))
        atom_hiddens = self.dropout_layer(atom_hiddens)
        if atom_descriptors_batch is not None:
            if len(atom_hiddens) != len(atom_descriptors_batch):
                raise ValueError(
                    "The number of atoms is different from the length of the extra atom features"
                )
            atom_hiddens = paddle.concat(
                x=[atom_hiddens, atom_descriptors_batch], axis=1
            )
            atom_hiddens = self.atom_descriptors_layer(atom_hiddens)
            atom_hiddens = self.dropout_layer(atom_hiddens)
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                start_0 = atom_hiddens.shape[0] + a_start if a_start < 0 else a_start
                cur_hiddens = paddle.slice(
                    atom_hiddens, [0], [start_0], [start_0 + a_size]
                )
                mol_vec = cur_hiddens
                if self.aggregation == "mean":
                    mol_vec = mol_vec.sum(axis=0) / a_size
                elif self.aggregation == "sum":
                    mol_vec = mol_vec.sum(axis=0)
                elif self.aggregation == "norm":
                    mol_vec = mol_vec.sum(axis=0) / self.aggregation_norm
                mol_vecs.append(mol_vec)
        mol_vecs = paddle.stack(x=mol_vecs, axis=0)
        return mol_vecs


class MPN(paddle.nn.Layer):
    """Wrapper around `MPNEncoder` which handles input featurization.

    Args:
        args: Model arguments in a `TrainArgs` instance.
        atom_fdim: Atom feature vector dimension.
        bond_fdim: Bond feature vector dimension.
    """

    def __init__(self, args: TrainArgs, atom_fdim: int = None, bond_fdim: int = None):
        super(MPN, self).__init__()
        self.reaction = args.reaction
        self.reaction_solvent = args.reaction_solvent
        self.atom_fdim = atom_fdim or get_atom_fdim(
            overwrite_default_atom=args.overwrite_default_atom_features,
            is_reaction=self.reaction or self.reaction_solvent,
        )
        self.bond_fdim = bond_fdim or get_bond_fdim(
            overwrite_default_atom=args.overwrite_default_atom_features,
            overwrite_default_bond=args.overwrite_default_bond_features,
            atom_messages=args.atom_messages,
            is_reaction=self.reaction or self.reaction_solvent,
        )
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.device = args.device
        self.atom_descriptors = args.atom_descriptors
        self.overwrite_default_atom_features = args.overwrite_default_atom_features
        self.overwrite_default_bond_features = args.overwrite_default_bond_features
        if self.features_only:
            return
        if not self.reaction_solvent:
            if args.mpn_shared:
                self.encoder = paddle.nn.LayerList(
                    sublayers=[MPNEncoder(args, self.atom_fdim, self.bond_fdim)]
                    * args.number_of_molecules
                )
            else:
                self.encoder = paddle.nn.LayerList(
                    sublayers=[
                        MPNEncoder(args, self.atom_fdim, self.bond_fdim)
                        for _ in range(args.number_of_molecules)
                    ]
                )
        else:
            self.encoder = MPNEncoder(args, self.atom_fdim, self.bond_fdim)
            self.atom_fdim_solvent = get_atom_fdim(
                overwrite_default_atom=args.overwrite_default_atom_features,
                is_reaction=False,
            )
            self.bond_fdim_solvent = get_bond_fdim(
                overwrite_default_atom=args.overwrite_default_atom_features,
                overwrite_default_bond=args.overwrite_default_bond_features,
                atom_messages=args.atom_messages,
                is_reaction=False,
            )
            self.encoder_solvent = MPNEncoder(
                args,
                self.atom_fdim_solvent,
                self.bond_fdim_solvent,
                args.hidden_size_solvent,
                args.bias_solvent,
                args.depth_solvent,
            )

    def forward(
        self,
        batch: Union[
            List[List[str]],
            List[List[Chem.Mol]],
            List[List[Tuple[Chem.Mol, Chem.Mol]]],
            List[BatchMolGraph],
        ],
        features_batch: List[np.ndarray] = None,
        atom_descriptors_batch: List[np.ndarray] = None,
        atom_features_batch: List[np.ndarray] = None,
        bond_features_batch: List[np.ndarray] = None,
    ) -> paddle.float32:
        """Encode a batch of molecules.

        Args:
            batch: List of lists of SMILES or RDKit molecules, or a list of `BatchMolGraph`.
                Outer list length is `num_molecules`; inner list length is `number_of_molecules`.
            features_batch: Optional list of numpy arrays with additional features.
            atom_descriptors_batch: Optional list of numpy arrays with atom descriptors.
            atom_features_batch: Optional list of numpy arrays with atom features.
            bond_features_batch: Optional list of numpy arrays with bond features.

        Returns:
            `paddle.Tensor`: Shape `(num_molecules, hidden_size)` containing encodings per molecule.
        """
        """
        if type(batch[0]) != BatchMolGraph:
            batch = [[mols[i] for mols in batch] for i in range(len(batch[0]))]
            if self.atom_descriptors == 'feature':
                if len(batch) > 1:
                    raise NotImplementedError(
                        'Atom/bond descriptors are currently only supported with one molecule per input (i.e., number_of_molecules = 1).'
                        )
                batch = [mol2graph(mols=b, atom_features_batch=
                    atom_features_batch, bond_features_batch=
                    bond_features_batch, overwrite_default_atom_features=
                    self.overwrite_default_atom_features,
                    overwrite_default_bond_features=self.
                    overwrite_default_bond_features) for b in batch]
            elif bond_features_batch is not None:
                if len(batch) > 1:
                    raise NotImplementedError(
                        'Atom/bond descriptors are currently only supported with one molecule per input (i.e., number_of_molecules = 1).'
                        )
                batch = [mol2graph(mols=b, bond_features_batch=
                    bond_features_batch, overwrite_default_atom_features=
                    self.overwrite_default_atom_features,
                    overwrite_default_bond_features=self.
                    overwrite_default_bond_features) for b in batch]
            else:
                 batch = [mol2graph(b) for b in batch]
        """

        if self.use_input_features:
            features_batch = (
                paddle.to_tensor(data=np.stack(features_batch))
                .astype(dtype="float32")
                .to(self.device)
            )
            if self.features_only:
                return features_batch
        if self.atom_descriptors == "descriptor":
            if len(batch) > 1:
                raise NotImplementedError(
                    "Atom descriptors are currently only supported with one molecule per input (i.e., number_of_molecules = 1)."
                )
            encodings = [
                enc(ba, atom_descriptors_batch) for enc, ba in zip(self.encoder, batch)
            ]
        elif not self.reaction_solvent:
            encodings = [enc(ba) for enc, ba in zip(self.encoder, batch)]
        else:
            encodings = []
            for ba in batch:
                if ba.is_reaction:
                    encodings.append(self.encoder(ba))
                else:
                    encodings.append(self.encoder_solvent(ba))
        output = reduce(lambda x, y: paddle.concat(x=(x, y), axis=1), encodings)
        if self.use_input_features:
            if len(tuple(features_batch.shape)) == 1:
                features_batch = features_batch.reshape([1, -1])
            output = paddle.concat(x=[output, features_batch], axis=1)
        return output


def split(x, num_or_sections, axis=0):
    if isinstance(num_or_sections, int):
        return paddle.split(x, x.shape[axis] // num_or_sections, axis)
    else:
        return paddle.split(x, num_or_sections, axis)


def compute_pnorm(model: paddle.nn.Layer) -> float:
    """Compute the L2 norm of model parameters.

    Args:
        model: A Paddle model.

    Returns:
        float: L2 norm of all parameters.
    """
    return math.sqrt(sum([(p.norm().item() ** 2) for p in model.parameters()]))


def compute_gnorm(model: paddle.nn.Layer) -> float:
    """Compute the L2 norm of model gradients.

    Args:
        model: A Paddle model.

    Returns:
        float: L2 norm of all available gradients.
    """
    return math.sqrt(
        sum(
            [
                (p.grad.norm().item() ** 2)
                for p in model.parameters()
                if p.grad is not None
            ]
        )
    )


def param_count(model: paddle.nn.Layer) -> int:
    """Count the number of trainable parameters.

    Args:
        model: A Paddle model.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(param.size for param in model.parameters() if not param.stop_gradient)


def param_count_all(model: paddle.nn.Layer) -> int:
    """Count the total number of parameters (trainable and frozen).

    Args:
        model: A Paddle model.

    Returns:
        int: Total number of parameters.
    """
    return sum(param.size for param in model.parameters())


def initialize_weights(model: paddle.nn.Layer) -> None:
    """Initialize model weights in place.

    Uses Constant for biases (1D) and XavierNormal for other parameters.

    Args:
        model: A Paddle model.
    """
    for param in model.parameters():
        if param.dim() == 1:
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(param)
        else:
            init_XavierNormal = paddle.nn.initializer.XavierNormal()
            init_XavierNormal(param)


class NoamLR(paddle.optimizer.lr.LRScheduler):
    """Noam learning rate scheduler with warmup and exponential decay.

    Learning rate increases linearly from `init_lr` to `max_lr` over
    `warmup_epochs * steps_per_epoch`, then decays exponentially to `final_lr`
    over the remaining steps, following Section 5.3 of "Attention is All You Need".

    Args:
        optimizer: Paddle optimizer.
        warmup_epochs: Number of epochs to linearly increase LR.
        total_epochs: Total number of epochs.
        steps_per_epoch: Number of steps (batches) per epoch.
        init_lr: Initial learning rate per param group.
        max_lr: Maximum learning rate per param group.
        final_lr: Final learning rate per param group.
    """

    def __init__(
        self,
        optimizer: paddle.optimizer.Optimizer,
        warmup_epochs: List[Union[float, int]],
        total_epochs: List[int],
        steps_per_epoch: int,
        init_lr: List[float],
        max_lr: List[float],
        final_lr: List[float],
    ):
        if (
            not len(optimizer._param_groups)
            == len(warmup_epochs)
            == len(total_epochs)
            == len(init_lr)
            == len(max_lr)
            == len(final_lr)
        ):
            raise ValueError(
                f"Number of param groups must match the number of epochs and learning rates! got: len(optimizer.param_groups)= {len(optimizer._param_groups)}, len(warmup_epochs)= {len(warmup_epochs)}, len(total_epochs)= {len(total_epochs)}, len(init_lr)= {len(init_lr)}, len(max_lr)= {len(max_lr)}, len(final_lr)= {len(final_lr)}"
            )
        self.num_lrs = len(optimizer._param_groups)
        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)
        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps
        self.exponential_gamma = (self.final_lr / self.max_lr) ** (
            1 / (self.total_steps - self.warmup_steps)
        )
        super(NoamLR, self).__init__(optimizer.get_lr())

    def get_lr(self) -> List[float]:
        """Return current learning rates per parameter group.

        Returns:
            list[float]: Current learning rates.
        """
        return list(self.lr)

    def step(self, current_step: int = None):
        """Advance scheduler and update optimizer learning rates.

        Args:
            current_step: Optional explicit step index. If None, increments by 1.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1
        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = (
                    self.init_lr[i] + self.current_step * self.linear_increment[i]
                )
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * self.exponential_gamma[i] ** (
                    self.current_step - self.warmup_steps[i]
                )
            else:
                self.lr[i] = self.final_lr[i]
            self.optimizer._param_groups[i]["learning_rate"] = self.lr[i]


def activate_dropout(module: paddle.nn.Layer, dropout_prob: float):
    """Enable dropout during inference for uncertainty estimation.

    Sets `p` on dropout layers and forces train mode.

    Args:
        module: Module to potentially modify if it is a dropout layer.
        dropout_prob: Dropout probability in (0, 1).
    """
    if isinstance(module, paddle.nn.Dropout):
        module.p = dropout_prob
        module.train()


class MoleculeModel(paddle.nn.Layer):
    """Model with an MPN encoder followed by feed-forward layers.

    Args:
        cfg: Hydra `DictConfig` used to build internal `TrainArgs`.
    """

    def __init__(self, cfg: DictConfig):
        super(MoleculeModel, self).__init__()
        args = self.build_from_cfg(cfg)
        self.classification = args.dataset_type == "classification"
        self.multiclass = args.dataset_type == "multiclass"
        self.loss_function = args.loss_function
        if hasattr(args, "train_class_sizes"):
            self.train_class_sizes = args.train_class_sizes
        else:
            self.train_class_sizes = None
        if self.classification or self.multiclass:
            self.no_training_normalization = args.loss_function in [
                "cross_entropy",
                "binary_cross_entropy",
            ]
        self.output_size = args.num_tasks
        if self.multiclass:
            self.output_size *= args.multiclass_num_classes
        if self.loss_function == "mve":
            self.output_size *= 2
        if self.loss_function == "dirichlet" and self.classification:
            self.output_size *= 2
        if self.loss_function == "evidential":
            self.output_size *= 4
        if self.classification:
            self.sigmoid = paddle.nn.Sigmoid()
        if self.multiclass:
            self.multiclass_softmax = paddle.nn.Softmax(axis=2)
        if self.loss_function in ["mve", "evidential", "dirichlet"]:
            self.softplus = paddle.nn.Softplus()
        self.create_encoder(args)
        self.create_ffn(args)
        initialize_weights(self)

    def _make_args(
        self,
        dataset_type,
        epochs,
        use_gpu,
        fingerprint_type,
        property_name,
        train_smiles=None,
        train_fingerprints=None,
    ):

        # Create args
        arg_list = [
            "--data_path",
            "foo.csv",
            "--dataset_type",
            dataset_type,
            "--save_dir",
            "foo",
            "--epochs",
            str(epochs),
            "--quiet",
        ] + ([] if use_gpu else ["--no_cuda"])

        if fingerprint_type == "morgan":
            arg_list += ["--features_generator", "morgan"]
        elif fingerprint_type == "rdkit":
            arg_list += [
                "--features_generator",
                "rdkit_2d_normalized",
                "--no_features_scaling",
            ]
        elif fingerprint_type is None:
            pass
        else:
            raise ValueError(f'Fingerprint type "{fingerprint_type}" is not supported.')

        args = TrainArgs().parse_args(arg_list)
        args.task_names = [property_name]
        if train_smiles is not None:
            args.train_data_size = len(train_smiles)

        if fingerprint_type is not None:
            args.features_size = train_fingerprints.shape[1]
        return args

    def build_from_cfg(self, cfg: DictConfig):
        args = self._make_args(
            dataset_type=cfg.DATA.dataset_type,  # "classification",
            epochs=cfg.TRAIN.epochs,  # 1,
            use_gpu=cfg.TRAIN.use_gpu,
            fingerprint_type=cfg.DATA.fingerprint_type,  # None,
            property_name=cfg.DATA.property_column,  # "antibiotic_activity"
            train_smiles=None,
            train_fingerprints=None,
        )
        return args

    def create_encoder(self, args: TrainArgs) -> None:
        """Create the message passing encoder for the model.

        Args:
            args: Model arguments in a `TrainArgs` instance.
        """
        self.encoder = MPN(args)
        if args.checkpoint_frzn is not None:
            if args.freeze_first_only:
                for param in list(self.encoder.encoder.children())[0].parameters():
                    param.stop_gradient = not False
            else:
                for param in self.encoder.parameters():
                    param.stop_gradient = not False

    def create_ffn(self, args: TrainArgs) -> None:
        """Create the feed-forward layers for the model.

        Args:
            args: Model arguments in a `TrainArgs` instance.
        """
        self.multiclass = args.dataset_type == "multiclass"
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            if args.reaction_solvent:
                first_linear_dim = args.hidden_size + args.hidden_size_solvent
            else:
                first_linear_dim = args.hidden_size * args.number_of_molecules
            if args.use_input_features:
                first_linear_dim += args.features_size
        if args.atom_descriptors == "descriptor":
            first_linear_dim += args.atom_descriptors_size
        dropout = paddle.nn.Dropout(p=args.dropout)
        activation = get_activation_function(args.activation)
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                paddle.nn.Linear(
                    in_features=first_linear_dim, out_features=self.output_size
                ),
            ]
        else:
            ffn = [
                dropout,
                paddle.nn.Linear(
                    in_features=first_linear_dim, out_features=args.ffn_hidden_size
                ),
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend(
                    [
                        activation,
                        dropout,
                        paddle.nn.Linear(
                            in_features=args.ffn_hidden_size,
                            out_features=args.ffn_hidden_size,
                        ),
                    ]
                )
            ffn.extend(
                [
                    activation,
                    dropout,
                    paddle.nn.Linear(
                        in_features=args.ffn_hidden_size, out_features=self.output_size
                    ),
                ]
            )
        if args.dataset_type == "spectra":
            if args.spectra_activation == "softplus":
                spectra_activation = paddle.nn.Softplus()
            else:

                class nn_exp(paddle.nn.Layer):
                    def __init__(self):
                        super(nn_exp, self).__init__()

                    def forward(self, x):
                        return paddle.exp(x=x)

                spectra_activation = nn_exp()
            ffn.append(spectra_activation)
        self.ffn = paddle.nn.Sequential(*ffn)
        if args.checkpoint_frzn is not None:
            if args.frzn_ffn_layers > 0:
                for param in list(self.ffn.parameters())[0 : 2 * args.frzn_ffn_layers]:
                    param.stop_gradient = not False

    def fingerprint(
        self,
        batch: Union[
            List[List[str]],
            List[List[Chem.Mol]],
            List[List[Tuple[Chem.Mol, Chem.Mol]]],
            List[BatchMolGraph],
        ],
        features_batch: List[np.ndarray] = None,
        atom_descriptors_batch: List[np.ndarray] = None,
        atom_features_batch: List[np.ndarray] = None,
        bond_features_batch: List[np.ndarray] = None,
        fingerprint_type: str = "MPN",
    ) -> paddle.Tensor:
        """Return latent representations (fingerprints) from intermediate stages.

        Args:
            batch: List of lists of SMILES or RDKit molecules, or a list of `BatchMolGraph`.
            features_batch: Optional list of numpy arrays with additional features.
            atom_descriptors_batch: Optional list of numpy arrays with atom descriptors.
            atom_features_batch: Optional list of numpy arrays with atom features.
            bond_features_batch: Optional list of numpy arrays with bond features.
            fingerprint_type: Which latent representation to return. Supported: "MPN" (output of MPN) or
                "last_FFN" (input to the final readout layer).

        Returns:
            `paddle.Tensor`: Fingerprint vectors.

        Raises:
            ValueError: If `fingerprint_type` is unsupported.
        """
        if fingerprint_type == "MPN":
            return self.encoder(
                batch,
                features_batch,
                atom_descriptors_batch,
                atom_features_batch,
                bond_features_batch,
            )
        elif fingerprint_type == "last_FFN":
            return self.ffn[:-1](
                self.encoder(
                    batch,
                    features_batch,
                    atom_descriptors_batch,
                    atom_features_batch,
                    bond_features_batch,
                )
            )
        else:
            raise ValueError(f"Unsupported fingerprint type {fingerprint_type}.")

    def forward(
        self,
        batch: Union[
            List[List[str]],
            List[List[Chem.Mol]],
            List[List[Tuple[Chem.Mol, Chem.Mol]]],
            List[BatchMolGraph],
        ],
        features_batch: List[np.ndarray] = None,
        atom_descriptors_batch: List[np.ndarray] = None,
        atom_features_batch: List[np.ndarray] = None,
        bond_features_batch: List[np.ndarray] = None,
    ) -> paddle.float32:
        """Run the model forward pass on input.

        Args:
            batch: Dictionary with keys `mol_batch`, `features_batch`,
                `atom_descriptors_batch`, `atom_features_batch`, `bond_features_batch`.
            features_batch: Unused (kept for API symmetry), see `batch`.
            atom_descriptors_batch: Unused, see `batch`.
            atom_features_batch: Unused, see `batch`.
            bond_features_batch: Unused, see `batch`.

        Returns:
            dict: `{"pred": tensor}` predictions (and transformed for special losses).
        """

        mol_batch = batch["mol_batch"]
        features_batch = batch["features_batch"]
        atom_descriptors_batch = batch["atom_descriptors_batch"]
        atom_features_batch = batch["atom_features_batch"]
        bond_features_batch = batch["bond_features_batch"]
        batch = mol_batch

        output = self.ffn(
            self.encoder(
                batch,
                features_batch,
                atom_descriptors_batch,
                atom_features_batch,
                bond_features_batch,
            )
        )
        if (
            self.classification
            and not (self.training and self.no_training_normalization)
            and self.loss_function != "dirichlet"
        ):
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((tuple(output.shape)[0], -1, self.num_classes))
            if (
                not (self.training and self.no_training_normalization)
                and self.loss_function != "dirichlet"
            ):
                output = self.multiclass_softmax(output)
        if self.loss_function == "mve":
            means, variances = split(
                x=output, num_or_sections=tuple(output.shape)[1] // 2, axis=1
            )
            variances = self.softplus(variances)
            output = paddle.concat([means, variances], axis=1)

        if self.loss_function == "evidential":
            means, lambdas, alphas, betas = split(
                x=output, num_or_sections=tuple(output.shape)[1] // 4, axis=1
            )
            lambdas = self.softplus(lambdas)
            alphas = self.softplus(alphas) + 1
            betas = self.softplus(betas)
            output = paddle.concat(x=[means, lambdas, alphas, betas], axis=1)
        if self.loss_function == "dirichlet":
            output = paddle.nn.functional.softplus(x=output) + 1
        return {"pred": output}
