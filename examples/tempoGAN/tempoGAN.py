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

    # initialize parameters and import classes
    USE_AMP = cfg.USE_AMP
    USE_SPATIALDISC = cfg.USE_SPATIALDISC
    USE_TEMPODISC = cfg.USE_TEMPODISC

    weight_gen = cfg.WEIGHT_GEN  # lambda_l1, lambda_l2, lambda_t
    weight_gen_layer = (
        cfg.WEIGHT_GEN_LAYER if USE_SPATIALDISC else None
    )  # lambda_layer, lambda_layer1, lambda_layer2, lambda_layer3, lambda_layer4
    weight_disc = cfg.WEIGHT_DISC
    tile_ratio = cfg.TILE_RATIO

    gen_funcs = func_module.GenFuncs(weight_gen, weight_gen_layer)
    dics_funcs = func_module.DiscFuncs(weight_disc)
    data_funcs = func_module.DataFuncs(tile_ratio)

    # load dataset
    dataset_train = hdf5storage.loadmat(cfg.DATASET_PATH)
    dataset_valid = hdf5storage.loadmat(cfg.DATASET_PATH_VALID)

    # init Generator params
    in_channel = 1
    rb_channel0 = (2, 8, 8)
    rb_channel1 = (128, 128, 128)
    rb_channel2 = (32, 8, 8)
    rb_channel3 = (2, 1, 1)
    out_channels_tuple = (rb_channel0, rb_channel1, rb_channel2, rb_channel3)
    kernel_sizes_tuple = (((5, 5),) * 2 + ((1, 1),),) * 4
    strides_tuple = ((1, 1, 1),) * 4
    use_bns_tuple = ((True, True, True),) * 3 + ((False, False, False),)
    acts_tuple = (("relu", None, None),) * 4

    # define Generator model
    model_gen = ppsci.arch.Generator(
        cfg.MODEL.input_keys,  # 'NCHW'
        cfg.MODEL.output_keys,
        in_channel,
        out_channels_tuple,
        kernel_sizes_tuple,
        strides_tuple,
        use_bns_tuple,
        acts_tuple,
    )

    model_gen.register_input_transform(gen_funcs.transform_in)
    dics_funcs.model_gen = model_gen

    model_tuple = (model_gen,)

    # init Discriminators params
    in_channel = 2
    in_channel_tempo = 3
    out_channels = (32, 64, 128, 256)
    in_shape = np.shape(dataset_train["density_high"][0])
    h, w = in_shape[1] // tile_ratio, in_shape[2] // tile_ratio
    down_sample_ratio = 2 ** (len(out_channels) - 1)
    fc_channel = int(
        out_channels[-1] * (h / down_sample_ratio) * (w / down_sample_ratio)
    )
    kernel_sizes = ((4, 4),) * 4
    strides = (2,) * 3 + (1,)
    use_bns = (False,) + (True,) * 3
    acts = ("leaky_relu",) * 4 + (None,)

    # define Discriminators
    if USE_SPATIALDISC:
        output_keys_disc = (
            tuple(f"out0_layer{i}" for i in range(4))
            + ("out_disc_from_target",)
            + tuple(f"out1_layer{i}" for i in range(4))
            + ("out_disc_from_gen",)
        )
        model_disc = ppsci.arch.Discriminator(
            ("input_disc_from_target", "input_disc_from_gen"),  # 'NCHW'
            output_keys_disc,
            in_channel,
            out_channels,
            fc_channel,
            kernel_sizes,
            strides,
            use_bns,
            acts,
        )
        model_disc.register_input_transform(dics_funcs.transform_in)
        model_tuple += (model_disc,)

    # define temporal Discriminators
    if USE_TEMPODISC:
        output_keys_disc_tempo = (
            tuple(f"out0_tempo_layer{i}" for i in range(4))
            + ("out_disc_tempo_from_target",)
            + tuple(f"out1_tempo_layer{i}" for i in range(4))
            + ("out_disc_tempo_from_gen",)
        )
        model_disc_tempo = ppsci.arch.Discriminator(
            ("input_tempo_disc_from_target", "input_tempo_disc_from_gen"),  # 'NCHW'
            output_keys_disc_tempo,
            in_channel_tempo,
            out_channels,
            fc_channel,
            kernel_sizes,
            strides,
            use_bns,
            acts,
        )
        model_disc_tempo.register_input_transform(dics_funcs.transform_in_tempo)
        model_tuple += (model_disc_tempo,)

    # define model_list
    model_list = ppsci.arch.ModelList(model_tuple)

    # set training hyper-parameters
    ITERS_PER_EPOCH = cfg.TRAIN.iters_per_epoch
    EPOCHS = cfg.TRAIN.epochs
    EPOCHS_GEN = EPOCHS_DISC = EPOCHS_DISC_TEMPO = 1
    BATCH_SIZE = cfg.TRAIN.batch_size.sup_constraint

    # initialize Adam optimizer
    lr_scheduler_gen = ppsci.optimizer.lr_scheduler.Step(
        step_size=EPOCHS // 2, **cfg.TRAIN.lr_scheduler
    )()
    optimizer_gen = ppsci.optimizer.Adam(lr_scheduler_gen)((model_gen,))
    if USE_SPATIALDISC:
        lr_scheduler_disc = ppsci.optimizer.lr_scheduler.Step(
            step_size=EPOCHS // 2, **cfg.TRAIN.lr_scheduler
        )()
        optimizer_disc = ppsci.optimizer.Adam(lr_scheduler_disc)((model_disc,))
    if USE_TEMPODISC:
        lr_scheduler_disc_tempo = ppsci.optimizer.lr_scheduler.Step(
            step_size=EPOCHS // 2, **cfg.TRAIN.lr_scheduler
        )()
        optimizer_disc_tempo = ppsci.optimizer.Adam(lr_scheduler_disc_tempo)(
            (model_disc_tempo,)
        )

    # Generator
    # maunally build constraint(s)
    sup_constraint_gen = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": {
                    "density_low": dataset_train["density_low"],
                    "density_high": dataset_train["density_high"],
                },
                "label": {"density_high": dataset_train["density_high"]},
                "transforms": (
                    {
                        "FunctionalTransform": {
                            "transform_func": data_funcs.transform,
                        },
                    },
                ),
            },
            "batch_size": BATCH_SIZE,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": False,
            },
        },
        ppsci.loss.FunctionalLoss(gen_funcs.loss_func_gen),
        name="sup_constraint_gen",
    )
    constraint_gen = {sup_constraint_gen.name: sup_constraint_gen}
    if USE_TEMPODISC:
        sup_constraint_gen_tempo = ppsci.constraint.SupervisedConstraint(
            {
                "dataset": {
                    "name": "NamedArrayDataset",
                    "input": {
                        "density_low": dataset_train["density_low_tempo"],
                        "density_high": dataset_train["density_high_tempo"],
                    },
                    "label": {"density_high": dataset_train["density_high_tempo"]},
                    "transforms": (
                        {
                            "FunctionalTransform": {
                                "transform_func": data_funcs.transform,
                            },
                        },
                    ),
                },
                "batch_size": int(BATCH_SIZE // 3),
                "sampler": {
                    "name": "BatchSampler",
                    "drop_last": False,
                    "shuffle": False,
                },
            },
            ppsci.loss.FunctionalLoss(gen_funcs.loss_func_gen_tempo),
            name="sup_constraint_gen_tempo",
        )
        constraint_gen[sup_constraint_gen_tempo.name] = sup_constraint_gen_tempo

    # Discriminators
    # maunally build constraint(s)
    if USE_SPATIALDISC:
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
                "batch_size": BATCH_SIZE,
                "sampler": {
                    "name": "BatchSampler",
                    "drop_last": False,
                    "shuffle": False,
                },
            },
            ppsci.loss.FunctionalLoss(dics_funcs.loss_func),
            name="sup_constraint_disc",
        )
        constraint_disc = {
            sup_constraint_disc.name: sup_constraint_disc,
        }

    # temporal Discriminators
    # maunally build constraint(s)
    if USE_TEMPODISC:
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
                "batch_size": int(BATCH_SIZE // 3),
                "sampler": {
                    "name": "BatchSampler",
                    "drop_last": False,
                    "shuffle": False,
                },
            },
            ppsci.loss.FunctionalLoss(dics_funcs.loss_func_tempo),
            name="sup_constraint_disc_tempo",
        )
        constraint_disc_tempo = {
            sup_constraint_disc_tempo.name: sup_constraint_disc_tempo,
        }

    # initialize solver
    solver_gen = ppsci.solver.Solver(
        model_list,
        constraint_gen,
        cfg.output_dir,
        optimizer_gen,
        lr_scheduler_gen,
        EPOCHS_GEN,
        ITERS_PER_EPOCH,
        eval_during_train=cfg.TRAIN.eval_during_train,
        use_amp=USE_AMP,
        amp_level=cfg.TRAIN.amp_level,
    )
    if USE_SPATIALDISC:
        solver_disc = ppsci.solver.Solver(
            model_list,
            constraint_disc,
            cfg.output_dir,
            optimizer_disc,
            lr_scheduler_disc,
            EPOCHS_DISC,
            ITERS_PER_EPOCH,
            eval_during_train=cfg.TRAIN.eval_during_train,
            use_amp=USE_AMP,
            amp_level=cfg.TRAIN.amp_level,
        )
    if USE_TEMPODISC:
        solver_disc_tempo = ppsci.solver.Solver(
            model_list,
            constraint_disc_tempo,
            cfg.output_dir,
            optimizer_disc_tempo,
            lr_scheduler_disc_tempo,
            EPOCHS_DISC_TEMPO,
            ITERS_PER_EPOCH,
            eval_during_train=cfg.TRAIN.eval_during_train,
            use_amp=USE_AMP,
            amp_level=cfg.TRAIN.amp_level,
        )

    PRED_INTERVAL = 200
    for i in range(1, EPOCHS + 1):
        logger.message(f"\nEpoch: {i}\n")
        # plotting during training
        if i == 1 or i % PRED_INTERVAL == 0 or i == EPOCHS:
            func_module.predict_and_save_plot(
                cfg.output_dir, i, solver_gen, dataset_valid, tile_ratio
            )

        dics_funcs.model_gen = model_gen
        # train disc, input: (x,y,G(x))
        if USE_SPATIALDISC:
            solver_disc.train()

        # train disc tempo, input: (y_3,G(x)_3)
        if USE_TEMPODISC:
            solver_disc_tempo.train()

        # train gen, input: (x,)
        solver_gen.train()


def evaluate(cfg: DictConfig):
    ############### evaluation after training ###############
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
