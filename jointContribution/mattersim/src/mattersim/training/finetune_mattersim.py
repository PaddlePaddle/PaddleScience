import argparse
import os
import pickle as pkl
import random

import numpy as np
import paddle
import wandb
from ase.units import GPa
from mattersim.datasets.utils.build import build_dataloader
from mattersim.forcefield.m3gnet.scaling import AtomScaling
from mattersim.forcefield.potential import Potential
from mattersim.utils.atoms_utils import AtomsAdaptor
from mattersim.utils.logger_utils import get_logger
from paddle.framework import core

logger = get_logger()

EAGER_COMP_OP_BLACK_LIST = [
    "abs_grad",
    "cast_grad",
    "concat_grad",
    "cos_double_grad",
    "cos_grad",
    "cumprod_grad",
    "cumsum_grad",
    "dropout_grad",
    "erf_grad",
    "exp_grad",
    "expand_grad",
    "floor_grad",
    "gather_grad",
    "gather_nd_grad",
    "gelu_grad",
    "group_norm_grad",
    "instance_norm_grad",
    "layer_norm_grad",
    "leaky_relu_grad",
    "log_grad",
    "max_grad",
    "pad_grad",
    "pow_double_grad",
    "pow_grad",
    "prod_grad",
    "relu_grad",
    "roll_grad",
    "rsqrt_grad",
    "scatter_grad",
    "scatter_nd_add_grad",
    "sigmoid_grad",
    "silu_grad",
    "sin_double_grad",
    "sin_grad",
    "slice_grad",
    "split_grad",
    "sqrt_grad",
    "sum_grad",
    "tanh_double_grad",
    "tanh_grad",
    "topk_grad",
    "transpose_grad",
    "add_double_grad",
    "add_grad",
    "batch_norm_grad",
    "divide_grad",
    "elementwise_pow_grad",
    "maximum_grad",
    "min_grad",
    "minimum_grad",
    "multiply_grad",
    "subtract_grad",
    "tile_grad",
]

EAGER_COMP_OP_BLACK_LIST = list(set(EAGER_COMP_OP_BLACK_LIST))


def setting_eager_mode(enable=True, custom_list=None):
    core.set_prim_eager_enabled(enable)
    if enable:
        if custom_list is None:
            paddle.framework.core._set_prim_backward_blacklist(
                *EAGER_COMP_OP_BLACK_LIST
            )
        else:
            raise ValueError("custom_list must be None or 'v1'")


setting_eager_mode()


def main(args):
    paddle.device.set_device(args.device)

    args_dict = vars(args)
    if args.wandb:
        wandb_api_key = (
            args.wandb_api_key
            if args.wandb_api_key is not None
            else os.getenv("WANDB_API_KEY")
        )
        wandb.login(key=wandb_api_key)
        wandb.init(project=args.wandb_project, name=args.run_name, config=args)
    if args.wandb:
        args_dict["wandb"] = wandb

    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(seed=args.seed)

    if args.train_data_path.endswith(".pkl"):
        with open(args.train_data_path, "rb") as f:
            atoms_train = pkl.load(f)
    else:
        atoms_train = AtomsAdaptor.from_file(filename=args.train_data_path)

    energies = []
    forces = [] if args.include_forces else None
    stresses = [] if args.include_stresses else None
    logger.info("Processing training datasets...")

    for atoms in atoms_train:
        energies.append(atoms.get_potential_energy())
        if args.include_forces:
            forces.append(atoms.get_forces())
        if args.include_stresses:
            stresses.append(atoms.get_stress(voigt=False) / GPa)

    dataloader = build_dataloader(
        atoms_train,
        energies,
        forces,
        stresses,
        shuffle=True,
        is_distributed=False,
        **args_dict,
    )
    if args.re_normalize:
        scale = AtomScaling(
            atoms=atoms_train,
            total_energy=energies,
            forces=forces,
            verbose=True,
            **args_dict,
        )
    if args.valid_data_path is not None:
        if args.valid_data_path.endswith(".pkl"):
            with open(args.valid_data_path, "rb") as f:
                atoms_val = pkl.load(f)
        else:
            atoms_val = AtomsAdaptor.from_file(filename=args.train_data_path)
        energies = []
        forces = [] if args.include_forces else None
        stresses = [] if args.include_stresses else None
        logger.info("Processing validation datasets...")
        for atoms in atoms_val:
            energies.append(atoms.get_potential_energy())
            if args.include_forces:
                forces.append(atoms.get_forces())
            if args.include_stresses:
                stresses.append(atoms.get_stress(voigt=False) / GPa)
        val_dataloader = build_dataloader(
            atoms_val,
            energies,
            forces,
            stresses,
            is_distributed=False,
            **args_dict,
        )
    else:
        val_dataloader = None
    potential = Potential.from_checkpoint(
        load_path=args.load_model_path, load_training_state=False, **args_dict
    )
    if args.re_normalize:
        potential.model.set_normalizer(scale)
    # TODO: Not support distributed now.
    potential.train_model(
        dataloader,
        val_dataloader,
        loss=paddle.nn.SmoothL1Loss(delta=0.01),
        is_distributed=False,
        **args_dict,
    )
    if args.save_checkpoint and args.wandb:
        wandb.save(os.path.join(args.save_path, "best_model.pdparams"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name", type=str, default="example", help="name of the run"
    )
    parser.add_argument(
        "--train_data_path", type=str, default="./sample.xyz", help="train data path"
    )
    parser.add_argument(
        "--valid_data_path", type=str, default=None, help="valid data path"
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default="mattersim-v1.0.0-1m",
        help="path to load the model",
    )
    parser.add_argument(
        "--save_path", type=str, default="./results", help="path to save the model"
    )
    parser.add_argument(
        "--save_checkpoint",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=10,
        help="save checkpoint every ckpt_interval epochs",
    )
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--cutoff", type=float, default=5.0, help="cutoff radius")
    parser.add_argument(
        "--threebody_cutoff",
        type=float,
        default=4.0,
        help="cutoff radius for three-body term, which should be smaller than cutoff (two-body)",
    )
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument(
        "--step_size",
        type=int,
        default=10,
        help="step epoch for learning rate scheduler",
    )
    parser.add_argument(
        "--include_forces",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--include_stresses",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--force_loss_ratio", type=float, default=1.0)
    parser.add_argument("--stress_loss_ratio", type=float, default=0.1)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--re_normalize",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="re-normalize the energy and forces according to the new data",
    )
    parser.add_argument("--scale_key", type=str, default="per_species_forces_rms")
    parser.add_argument(
        "--shift_key", type=str, default="per_species_energy_mean_linear_reg"
    )
    parser.add_argument("--init_scale", type=float, default=None)
    parser.add_argument("--init_shift", type=float, default=None)
    parser.add_argument(
        "--trainable_scale",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--trainable_shift",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_api_key", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="wandb_test")

    args = parser.parse_args()

    main(args)
