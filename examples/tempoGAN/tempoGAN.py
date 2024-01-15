# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from os import path as osp

import functions as func_module
import hydra
import numpy as np
import paddle
from omegaconf import DictConfig

import ppsci
from ppsci.utils import checker
from ppsci.utils import logger
from ppsci.utils import save_load

if not checker.dynamic_import_to_globals("hdf5storage"):
    raise ImportError(
        "Could not import hdf5storage python package. "
        "Please install it with `pip install hdf5storage`."
    )
import hdf5storage


def train(cfg: DictConfig):
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "train.log"), "info")

    gen_funcs = func_module.GenFuncs(
        cfg.WEIGHT_GEN, (cfg.WEIGHT_GEN_LAYER if cfg.USE_SPATIALDISC else None)
    )
    disc_funcs = func_module.DiscFuncs(cfg.WEIGHT_DISC)
    data_funcs = func_module.DataFuncs(cfg.TILE_RATIO)

    # load dataset
    logger.message(
        "Attention! Start loading datasets, this will take tens of seconds to several minutes, please wait patiently."
    )
    dataset_train = hdf5storage.loadmat(cfg.DATASET_PATH)
    logger.message("Finish loading training dataset.")
    dataset_valid = hdf5storage.loadmat(cfg.DATASET_PATH_VALID)
    logger.message("Finish loading validation dataset.")

    # define Generator model
    model_gen = ppsci.arch.Generator(**cfg.MODEL.gen_net)
    model_gen.register_input_transform(gen_funcs.transform_in)
    disc_funcs.model_gen = model_gen

    model_tuple = (model_gen,)
    # define Discriminators
    if cfg.USE_SPATIALDISC:
        model_disc = ppsci.arch.Discriminator(**cfg.MODEL.disc_net)
        model_disc.register_input_transform(disc_funcs.transform_in)
        model_tuple += (model_disc,)

    # define temporal Discriminators
    if cfg.USE_TEMPODISC:
        model_disc_tempo = ppsci.arch.Discriminator(**cfg.MODEL.tempo_net)
        model_disc_tempo.register_input_transform(disc_funcs.transform_in_tempo)
        model_tuple += (model_disc_tempo,)

    # define model_list
    model_list = ppsci.arch.ModelList(model_tuple)

    # initialize Adam optimizer
    lr_scheduler_gen = ppsci.optimizer.lr_scheduler.Step(
        step_size=cfg.TRAIN.epochs // 2, **cfg.TRAIN.lr_scheduler
    )()
    optimizer_gen = ppsci.optimizer.Adam(lr_scheduler_gen)(model_gen)
    if cfg.USE_SPATIALDISC:
        lr_scheduler_disc = ppsci.optimizer.lr_scheduler.Step(
            step_size=cfg.TRAIN.epochs // 2, **cfg.TRAIN.lr_scheduler
        )()
        optimizer_disc = ppsci.optimizer.Adam(lr_scheduler_disc)(model_disc)
    if cfg.USE_TEMPODISC:
        lr_scheduler_disc_tempo = ppsci.optimizer.lr_scheduler.Step(
            step_size=cfg.TRAIN.epochs // 2, **cfg.TRAIN.lr_scheduler
        )()
        optimizer_disc_tempo = ppsci.optimizer.Adam(lr_scheduler_disc_tempo)(
            (model_disc_tempo,)
        )

    # Generator
    # manually build constraint(s)
    sup_constraint_gen = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": {
                    "density_low": dataset_train["density_low"],
                    "density_high": dataset_train["density_high"],
                },
                "transforms": (
                    {
                        "FunctionalTransform": {
                            "transform_func": data_funcs.transform,
                        },
                    },
                ),
            },
            "batch_size": cfg.TRAIN.batch_size.sup_constraint,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": False,
            },
        },
        ppsci.loss.FunctionalLoss(gen_funcs.loss_func_gen),
        {
            "output_gen": lambda out: out["output_gen"],
            "density_high": lambda out: out["density_high"],
        },
        name="sup_constraint_gen",
    )
    constraint_gen = {sup_constraint_gen.name: sup_constraint_gen}
    if cfg.USE_TEMPODISC:
        sup_constraint_gen_tempo = ppsci.constraint.SupervisedConstraint(
            {
                "dataset": {
                    "name": "NamedArrayDataset",
                    "input": {
                        "density_low": dataset_train["density_low_tempo"],
                        "density_high": dataset_train["density_high_tempo"],
                    },
                    "transforms": (
                        {
                            "FunctionalTransform": {
                                "transform_func": data_funcs.transform,
                            },
                        },
                    ),
                },
                "batch_size": int(cfg.TRAIN.batch_size.sup_constraint // 3),
                "sampler": {
                    "name": "BatchSampler",
                    "drop_last": False,
                    "shuffle": False,
                },
            },
            ppsci.loss.FunctionalLoss(gen_funcs.loss_func_gen_tempo),
            {
                "output_gen": lambda out: out["output_gen"],
                "density_high": lambda out: out["density_high"],
            },
            name="sup_constraint_gen_tempo",
        )
        constraint_gen[sup_constraint_gen_tempo.name] = sup_constraint_gen_tempo

    # Discriminators
    # manually build constraint(s)
    if cfg.USE_SPATIALDISC:
        sup_constraint_disc = ppsci.constraint.SupervisedConstraint(
            {
                "dataset": {
                    "name": "NamedArrayDataset",
                    "input": {
                        "density_low": dataset_train["density_low"],
                        "density_high": dataset_train["density_high"],
                    },
                    "label": {
                        "out_disc_from_target": np.ones(
                            (np.shape(dataset_train["density_high"])[0], 1),
                            dtype=paddle.get_default_dtype(),
                        ),
                        "out_disc_from_gen": np.ones(
                            (np.shape(dataset_train["density_high"])[0], 1),
                            dtype=paddle.get_default_dtype(),
                        ),
                    },
                    "transforms": (
                        {
                            "FunctionalTransform": {
                                "transform_func": data_funcs.transform,
                            },
                        },
                    ),
                },
                "batch_size": cfg.TRAIN.batch_size.sup_constraint,
                "sampler": {
                    "name": "BatchSampler",
                    "drop_last": False,
                    "shuffle": False,
                },
            },
            ppsci.loss.FunctionalLoss(disc_funcs.loss_func),
            name="sup_constraint_disc",
        )
        constraint_disc = {sup_constraint_disc.name: sup_constraint_disc}

    # temporal Discriminators
    # manually build constraint(s)
    if cfg.USE_TEMPODISC:
        sup_constraint_disc_tempo = ppsci.constraint.SupervisedConstraint(
            {
                "dataset": {
                    "name": "NamedArrayDataset",
                    "input": {
                        "density_low": dataset_train["density_low_tempo"],
                        "density_high": dataset_train["density_high_tempo"],
                    },
                    "label": {
                        "out_disc_tempo_from_target": np.ones(
                            (np.shape(dataset_train["density_high_tempo"])[0], 1),
                            dtype=paddle.get_default_dtype(),
                        ),
                        "out_disc_tempo_from_gen": np.ones(
                            (np.shape(dataset_train["density_high_tempo"])[0], 1),
                            dtype=paddle.get_default_dtype(),
                        ),
                    },
                    "transforms": (
                        {
                            "FunctionalTransform": {
                                "transform_func": data_funcs.transform,
                            },
                        },
                    ),
                },
                "batch_size": int(cfg.TRAIN.batch_size.sup_constraint // 3),
                "sampler": {
                    "name": "BatchSampler",
                    "drop_last": False,
                    "shuffle": False,
                },
            },
            ppsci.loss.FunctionalLoss(disc_funcs.loss_func_tempo),
            name="sup_constraint_disc_tempo",
        )
        constraint_disc_tempo = {
            sup_constraint_disc_tempo.name: sup_constraint_disc_tempo
        }

    # initialize solver
    solver_gen = ppsci.solver.Solver(
        model_list,
        constraint_gen,
        cfg.output_dir,
        optimizer_gen,
        lr_scheduler_gen,
        cfg.TRAIN.epochs_gen,
        cfg.TRAIN.iters_per_epoch,
        eval_during_train=cfg.TRAIN.eval_during_train,
        use_amp=cfg.USE_AMP,
        amp_level=cfg.TRAIN.amp_level,
    )
    if cfg.USE_SPATIALDISC:
        solver_disc = ppsci.solver.Solver(
            model_list,
            constraint_disc,
            cfg.output_dir,
            optimizer_disc,
            lr_scheduler_disc,
            cfg.TRAIN.epochs_disc,
            cfg.TRAIN.iters_per_epoch,
            eval_during_train=cfg.TRAIN.eval_during_train,
            use_amp=cfg.USE_AMP,
            amp_level=cfg.TRAIN.amp_level,
        )
    if cfg.USE_TEMPODISC:
        solver_disc_tempo = ppsci.solver.Solver(
            model_list,
            constraint_disc_tempo,
            cfg.output_dir,
            optimizer_disc_tempo,
            lr_scheduler_disc_tempo,
            cfg.TRAIN.epochs_disc_tempo,
            cfg.TRAIN.iters_per_epoch,
            eval_during_train=cfg.TRAIN.eval_during_train,
            use_amp=cfg.USE_AMP,
            amp_level=cfg.TRAIN.amp_level,
        )

    PRED_INTERVAL = 200
    for i in range(1, cfg.TRAIN.epochs + 1):
        logger.message(f"\nEpoch: {i}\n")
        # plotting during training
        if i == 1 or i % PRED_INTERVAL == 0 or i == cfg.TRAIN.epochs:
            func_module.predict_and_save_plot(
                cfg.output_dir, i, solver_gen, dataset_valid, cfg.TILE_RATIO
            )

        disc_funcs.model_gen = model_gen
        # train disc, input: (x,y,G(x))
        if cfg.USE_SPATIALDISC:
            solver_disc.train()

        # train disc tempo, input: (y_3,G(x)_3)
        if cfg.USE_TEMPODISC:
            solver_disc_tempo.train()

        # train gen, input: (x,)
        solver_gen.train()

    ############### evaluation for training ###############
    img_target = (
        func_module.get_image_array(
            os.path.join(cfg.output_dir, "predict", "target.png")
        )
        / 255.0
    )
    img_pred = (
        func_module.get_image_array(
            os.path.join(
                cfg.output_dir, "predict", f"pred_epoch_{cfg.TRAIN.epochs}.png"
            )
        )
        / 255.0
    )
    eval_mse, eval_psnr, eval_ssim = func_module.evaluate_img(img_target, img_pred)
    logger.message(f"MSE: {eval_mse}, PSNR: {eval_psnr}, SSIM: {eval_ssim}")


def evaluate(cfg: DictConfig):
    if cfg.EVAL.save_outs:
        from matplotlib import image as Img

        os.makedirs(osp.join(cfg.output_dir, "eval_outs"), exist_ok=True)

    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "eval.log"), "info")

    gen_funcs = func_module.GenFuncs(cfg.WEIGHT_GEN, None)

    # load dataset
    dataset_valid = hdf5storage.loadmat(cfg.DATASET_PATH_VALID)

    # define Generator model
    model_gen = ppsci.arch.Generator(**cfg.MODEL.gen_net)
    model_gen.register_input_transform(gen_funcs.transform_in)

    # define model_list
    model_list = ppsci.arch.ModelList((model_gen,))

    # load pretrained model
    save_load.load_pretrain(model_list, cfg.EVAL.pretrained_model_path)

    # set validator
    eval_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": {
                "density_low": dataset_valid["density_low"],
            },
            "label": {"density_high": dataset_valid["density_high"]},
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "batch_size": 1,
    }
    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        ppsci.loss.MSELoss("mean"),
        {"density_high": lambda out: out["output_gen"]},
        metric={"metric": ppsci.metric.L2Rel()},
        name="sup_validator_gen",
    )

    # customized evalution
    def scale(data):
        smax = np.max(data)
        smin = np.min(data)
        return (data - smin) / (smax - smin)

    eval_mse_list = []
    eval_psnr_list = []
    eval_ssim_list = []
    for i, (input, label, _) in enumerate(sup_validator.data_loader):
        output_dict = model_list({"density_low": input["density_low"]})
        output_arr = scale(np.squeeze(output_dict["output_gen"].numpy()))
        target_arr = scale(np.squeeze(label["density_high"].numpy()))

        eval_mse, eval_psnr, eval_ssim = func_module.evaluate_img(
            target_arr, output_arr
        )
        eval_mse_list.append(eval_mse)
        eval_psnr_list.append(eval_psnr)
        eval_ssim_list.append(eval_ssim)

        if cfg.EVAL.save_outs:
            Img.imsave(
                osp.join(cfg.output_dir, "eval_outs", f"out_{i}.png"),
                output_arr,
                vmin=0.0,
                vmax=1.0,
                cmap="gray",
            )
    logger.message(
        f"MSE: {np.mean(eval_mse_list)}, PSNR: {np.mean(eval_psnr_list)}, SSIM: {np.mean(eval_ssim_list)}"
    )


@hydra.main(version_base=None, config_path="./conf", config_name="tempogan.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
