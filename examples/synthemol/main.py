# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime
from pathlib import Path
from random import Random

import hydra
import numpy as np
import paddle
import pandas as pd
from chemprop_models import chemprop_predict
from chemprop_models import my_chemprop_load
from evaluation import evaluate_auto
from loss_functions import get_loss_func
from omegaconf import DictConfig
from synthemol.generate.generator import Generator
from synthemol.generate.utils import create_model_scoring_fn
from synthemol.generate.utils import save_generated_molecules
from synthemol.reactions import REACTIONS
from synthemol.reactions import load_and_set_allowed_reaction_building_blocks
from synthemol.reactions import set_all_building_blocks
from tqdm import tqdm

import ppsci
import ppsci.arch.chemprop_molecule
from ppsci.arch.chemprop_molecule_utils import TrainArgs


def get_train_loss_func(args):  #:paddle.Tensor=None):
    def train_loss_func(output_dict, label_dict, weight_dict):
        preds = output_dict["pred"]

        targets = label_dict["targets"]
        target_weights = label_dict["target_weights"]
        data_weights = label_dict["data_weights"]
        mask = label_dict["mask"]

        loss_func = get_loss_func(args)
        if args.loss_function == "bounded_mse":
            # lt_target_batch = lt_target_batch
            # gt_target_batch = gt_target_batch
            pass
        if args.loss_function == "mcc" and args.dataset_type == "classification":
            loss = loss_func(
                preds, targets, data_weights, mask
            ) * target_weights.squeeze(axis=0)
        elif args.loss_function == "mcc":
            targets = targets.astype(dtype="int64")
            target_losses = []
            for target_index in range(preds.shape[1]):
                target_loss = loss_func(
                    preds[:, target_index, :],
                    targets[:, target_index],
                    data_weights,
                    mask[:, target_index],
                ).unsqueeze(axis=0)
                target_losses.append(target_loss)
            loss = paddle.concat(x=target_losses) * target_weights.squeeze(axis=0)
        elif args.dataset_type == "multiclass":
            targets = targets.astype(dtype="int64")
            if args.loss_function == "dirichlet":
                loss = (
                    loss_func(preds, targets, args.evidential_regularization)
                    * target_weights
                    * data_weights
                    * mask
                )
            else:
                target_losses = []
                for target_index in range(preds.shape[1]):
                    target_loss = loss_func(
                        preds[:, target_index, :], targets[:, target_index]
                    ).unsqueeze(axis=1)
                    target_losses.append(target_loss)
                loss = (
                    paddle.concat(x=target_losses, axis=1)
                    * target_weights
                    * data_weights
                    * mask
                )
        elif args.dataset_type == "spectra":
            loss = (
                loss_func(preds, targets, mask) * target_weights * data_weights * mask
            )
        elif args.loss_function == "bounded_mse":
            pass
            """
            loss = (
                loss_func(preds, targets, lt_target_batch, gt_target_batch)
                * target_weights
                * data_weights
                * mask
            )
            """
        elif args.loss_function == "evidential":
            loss = (
                loss_func(preds, targets, args.evidential_regularization)
                * target_weights
                * data_weights
                * mask
            )
        elif args.loss_function == "dirichlet":
            loss = (
                loss_func(preds, targets, args.evidential_regularization)
                * target_weights
                * data_weights
                * mask
            )
        else:
            loss = loss_func(preds, targets) * target_weights * data_weights * mask
        loss = loss.sum() / mask.sum()

        return {"pred": loss.astype("float32")}

    return train_loss_func


def make_args(
    dataset_type,
    epochs,
    use_gpu,
    fingerprint_type,
    property_name,
    train_smiles,
    train_fingerprints,
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


def load_raw_data(cfg):
    data_path = cfg.DATA.data_path
    data = pd.read_csv(data_path)
    print(f"Data size = {len(data):,}")
    num_models = cfg.DATA.num_models  # 10
    num_folds = cfg.DATA.num_folds  # 10
    indices = np.tile(np.arange(num_folds), 1 + len(data) // num_folds)[: len(data)]
    random = Random(0)
    random.shuffle(indices)
    assert 1 <= num_models <= num_folds
    smiles_column = cfg.DATA.smiles_column  #'smiles'
    property_column = cfg.DATA.property_column  #'antibiotic_activity'

    model_num = 1
    test_index = model_num
    val_index = (model_num + 1) % num_folds
    test_mask = indices == test_index
    val_mask = indices == val_index
    train_mask = ~(test_mask | val_mask)
    test_data = data[test_mask]
    val_data = data[val_mask]
    train_data = data[train_mask]
    print(
        "test_data:",
        len(test_data),
        "train_data:",
        len(train_data),
        "val_data:",
        len(val_data),
    )
    train_smiles = train_data[smiles_column]
    train_fingerprints = None
    train_properties = train_data[property_column]
    return train_smiles, train_fingerprints, train_properties


def train(cfg: DictConfig):
    train_smiles, train_fingerprints, train_properties = load_raw_data(cfg)

    args = make_args(
        dataset_type=cfg.DATA.dataset_type,  # "classification",
        epochs=cfg.TRAIN.epochs,  # 1,
        use_gpu=cfg.TRAIN.use_gpu,
        fingerprint_type=cfg.DATA.fingerprint_type,  # None,
        property_name=cfg.DATA.property_column,  # "antibiotic_activity"
        train_smiles=train_smiles,
        train_fingerprints=train_fingerprints,
    )

    # set dataloader config
    train_dataloader_cfg = {
        "dataset": {
            "name": cfg.DATA.dataset_name,  # "MoleculeDatasetIter",
            "input_keys": tuple(cfg.MODEL.input_keys),
            "args": args,
            "smiles": train_smiles,
            "fingerprints": train_fingerprints,
            "properties": train_properties,
            "label_keys": tuple(cfg.MODEL.label_keys),
        },
        "num_workers": cfg.TRAIN.num_workers,
    }

    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        output_expr={"pred": lambda out: out["pred"]},
        loss=ppsci.loss.FunctionalLoss(get_train_loss_func(args)),
        name="Sup",
    )

    # wrap constraints together
    constraint = {sup_constraint.name: sup_constraint}

    # set model
    model = ppsci.arch.chemprop_molecule.MoleculeModel(cfg=cfg)

    # set optimizer
    optimizer = ppsci.optimizer.Adam(
        learning_rate=cfg.TRAIN.learning_rate,
        weight_decay=None,  # 0.001
    )(model)

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        None,
        cfg.TRAIN.epochs,
        cfg.TRAIN.iters_per_epoch,
        save_freq=cfg.TRAIN.save_freq,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_freq=cfg.TRAIN.eval_freq,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
        checkpoint_path=cfg.TRAIN.checkpoint_path,
    )

    # train model
    solver.train()


def evaluate(cfg):
    data_path = cfg.DATA.data_path
    data = pd.read_csv(data_path)
    print(f"Data size = {len(data):,}")
    num_models = cfg.DATA.num_models
    num_folds = cfg.DATA.num_folds
    indices = np.tile(np.arange(num_folds), 1 + len(data) // num_folds)[: len(data)]
    random = Random(0)
    random.shuffle(indices)
    assert 1 <= num_models <= num_folds
    smiles_column = cfg.DATA.smiles_column  #'smiles'
    property_column = cfg.DATA.property_column  #'antibiotic_activity'

    model_num = 1
    test_index = model_num
    val_index = (model_num + 1) % num_folds
    test_mask = indices == test_index
    val_mask = indices == val_index
    train_mask = ~(test_mask | val_mask)
    test_data = data[test_mask]
    val_data = data[val_mask]
    train_data = data[train_mask]
    print(
        "test_data:",
        len(test_data),
        "train_data:",
        len(train_data),
        "val_data:",
        len(val_data),
    )

    # load model
    model_path = Path(cfg.PRE_COMPUTE.model_path)
    use_gpu = cfg.PRE_COMPUTE.use_gpu
    model_type = cfg.PRE_COMPUTE.model_type  #'chemprop'

    model = ppsci.arch.chemprop_molecule.MoleculeModel(cfg=cfg)

    if model_type == "chemprop":
        if use_gpu:
            device = str("cuda").replace("cuda", "gpu")
        else:
            device = paddle.CPUPlace()
        paddle.seed(seed=0)
    m = my_chemprop_load(model, model_path=model_path, device=device)
    test_preds = chemprop_predict(
        model=m, smiles=test_data[smiles_column], fingerprints=None, num_workers=1
    )

    scores = evaluate_auto(
        true=test_data[property_column],
        preds=test_preds,
        dataset_type=cfg.DATA.dataset_type,
    )
    for score_name, score_value in scores.items():
        print(f"Test {score_name} = {score_value:.3f}")


def pre_compute(cfg):
    data_path = Path(cfg.PRE_COMPUTE.data_path)
    model_path = Path(cfg.PRE_COMPUTE.model_path)
    smiles_column = cfg.PRE_COMPUTE.smiles_column
    model_type = cfg.PRE_COMPUTE.model_type  #'chemprop'
    fingerprint_type = cfg.PRE_COMPUTE.fingerprint_type
    use_gpu = cfg.PRE_COMPUTE.use_gpu
    average_preds = cfg.PRE_COMPUTE.average_preds
    num_workers = cfg.PRE_COMPUTE.num_workers
    preds_column_prefix = cfg.PRE_COMPUTE.preds_column_prefix
    save_path = Path(cfg.PRE_COMPUTE.save_path)

    model = ppsci.arch.chemprop_molecule.MoleculeModel(cfg=cfg)

    data = pd.read_csv(data_path)
    smiles = list(data[smiles_column])
    if model_type != "chemprop" and fingerprint_type is None:
        raise ValueError("Must define fingerprint_type if using sklearn model.")
    if fingerprint_type is not None:
        # fingerprints = compute_fingerprints(smiles, fingerprint_type=
        #    fingerprint_type)
        pass
    else:
        fingerprints = None
    if model_path.is_dir():
        model_paths = list(
            model_path.glob("**/*.pt" if model_type == "chemprop" else "**/*.pkl")
        )
        if len(model_paths) == 0:
            raise ValueError(f"Could not find any models in directory {model_path}.")
    else:
        model_paths = [model_path]
    if model_type == "chemprop":
        if use_gpu:
            device = str("cuda").replace("cuda", "gpu")
        else:
            device = paddle.CPUPlace()
        paddle.seed(seed=0)

        models = [
            my_chemprop_load(model, model_path=model_path, device=device)
            for model_path in model_paths
        ]

    print(model_paths, models)

    if model_type == "chemprop":
        preds = np.array(
            [
                chemprop_predict(
                    model=m,
                    smiles=smiles,
                    fingerprints=fingerprints,
                    num_workers=num_workers,
                )
                for m in tqdm(models, desc="models")
            ]
        )

    if average_preds:
        preds = np.mean(preds, axis=0)
    model_string = (
        f"{model_type}{f'_{fingerprint_type}' if fingerprint_type is not None else ''}"
    )
    preds_string = f"{f'{preds_column_prefix}_' if preds_column_prefix is not None else ''}{model_string}"
    if average_preds:
        data[f"{preds_string}_ensemble_preds"] = preds
    else:
        for model_num, model_preds in enumerate(preds):
            data[f"{preds_string}_model_{model_num}_preds"] = model_preds
    if save_path is None:
        save_path = data_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(save_path, index=False)


def generate(cfg):
    model_path = cfg.GENERATE.model_path
    model_type = cfg.GENERATE.model_type  #'chemprop'
    save_dir = Path(cfg.GENERATE.save_dir)

    building_blocks_path = cfg.GENERATE.building_blocks_path
    fingerprint_type = cfg.GENERATE.fingerprint_type
    reaction_to_building_blocks_path = cfg.GENERATE.reaction_to_building_blocks_path
    building_blocks_id_column = cfg.GENERATE.building_blocks_id_column
    building_blocks_score_column = cfg.GENERATE.building_blocks_score_column

    building_blocks_smiles_column = cfg.GENERATE.building_blocks_smiles_column
    reactions = REACTIONS
    max_reactions = cfg.GENERATE.max_reactions
    n_rollout = cfg.GENERATE.n_rollout

    explore_weight = cfg.GENERATE.explore_weight
    num_expand_nodes = cfg.GENERATE.num_expand_nodes

    optimization = cfg.GENERATE.optimization
    rng_seed = cfg.GENERATE.rng_seed

    no_building_block_diversity = cfg.GENERATE.no_building_block_diversity
    store_nodes = cfg.GENERATE.store_nodes

    verbose = cfg.GENERATE.verbose
    replicate = cfg.GENERATE.replicate

    save_dir.mkdir(parents=True, exist_ok=True)
    print("Loading building blocks...")
    if replicate:
        building_block_data = pd.read_csv(
            building_blocks_path, dtype={building_blocks_score_column: str}
        )
        building_block_data[building_blocks_score_column] = building_block_data[
            building_blocks_score_column
        ].astype(float)
        old_reactions_order = [
            275592,
            22,
            11,
            527,
            2430,
            2708,
            240690,
            2230,
            2718,
            40,
            1458,
            271948,
            27,
        ]
        reactions = tuple(
            sorted(
                reactions, key=lambda reaction: old_reactions_order.index(reaction.id)
            )
        )
        building_block_data.drop_duplicates(
            subset=building_blocks_smiles_column, inplace=True
        )
    else:
        building_block_data = pd.read_csv(building_blocks_path)
    print(f"Loaded {len(building_block_data):,} building blocks")
    if building_block_data[building_blocks_id_column].nunique() != len(
        building_block_data
    ):
        raise ValueError("Building block IDs are not unique.")
    building_block_smiles_to_id = dict(
        zip(
            building_block_data[building_blocks_smiles_column],
            building_block_data[building_blocks_id_column],
        )
    )
    building_block_id_to_smiles = dict(
        zip(
            building_block_data[building_blocks_id_column],
            building_block_data[building_blocks_smiles_column],
        )
    )
    building_block_smiles_to_score = dict(
        zip(
            building_block_data[building_blocks_smiles_column],
            building_block_data[building_blocks_score_column],
        )
    )
    print(f"Found {len(building_block_smiles_to_id):,} unique building blocks")
    set_all_building_blocks(
        reactions=reactions, building_blocks=set(building_block_smiles_to_id)
    )
    if reaction_to_building_blocks_path is not None:
        print("Loading and setting allowed building blocks for each reaction...")
        load_and_set_allowed_reaction_building_blocks(
            reactions=reactions,
            reaction_to_reactant_to_building_blocks_path=reaction_to_building_blocks_path,
        )
    print("Loading models and creating model scoring function...")
    model_scoring_fn = create_model_scoring_fn(
        model_path=model_path,
        model_type=model_type,
        fingerprint_type=fingerprint_type,
        smiles_to_score=building_block_smiles_to_score,
    )
    print("Setting up generator...")
    generator = Generator(
        building_block_smiles_to_id=building_block_smiles_to_id,
        max_reactions=max_reactions,
        scoring_fn=model_scoring_fn,
        explore_weight=explore_weight,
        num_expand_nodes=num_expand_nodes,
        optimization=optimization,
        reactions=reactions,
        rng_seed=rng_seed,
        no_building_block_diversity=no_building_block_diversity,
        store_nodes=store_nodes,
        verbose=verbose,
        replicate=replicate,
    )
    print("Generating molecules...")
    start_time = datetime.now()
    nodes = generator.generate(n_rollout=n_rollout)
    stats = {
        "mcts_time": datetime.now() - start_time,
        "num_nonzero_reaction_molecules": len(nodes),
        "approx_num_nodes_searched": generator.approx_num_nodes_searched,
    }
    print(f"MCTS time = {stats['mcts_time']}")
    print(
        f"Number of full molecule, nonzero reaction nodes = {stats['num_nonzero_reaction_molecules']:,}"
    )
    print(
        f"Approximate total number of nodes searched = {stats['approx_num_nodes_searched']:,}"
    )
    if store_nodes:
        stats["num_nodes_searched"] = generator.num_nodes_searched
        print(f"Total number of nodes searched = {stats['num_nodes_searched']:,}")
    pd.DataFrame(data=[stats]).to_csv(save_dir / "mcts_stats.csv", index=False)
    print("Saving molecules...")
    save_generated_molecules(
        nodes=nodes,
        building_block_id_to_smiles=building_block_id_to_smiles,
        save_path=save_dir / "molecules.csv",
    )


@hydra.main(version_base=None, config_path="./conf", config_name="synthemol.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    elif cfg.mode == "pre-compute":
        pre_compute(cfg)
    elif cfg.mode == "generate":
        generate(cfg)
    else:
        raise ValueError(
            f"cfg.mode should in ['train', 'eval', 'pre-compute', 'generate'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()
