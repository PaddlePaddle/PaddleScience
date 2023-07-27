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

# import scipy.io

import functions as func_module
import hdf5storage
import numpy as np
from functions import DataFuncs
from functions import DiscFuncs
from functions import GenFuncs

import ppsci
from ppsci.utils import config
from ppsci.utils import logger

if __name__ == "__main__":
    args = config.parse_args()
    ppsci.utils.misc.set_random_seed(42)
    DATASET_PATH = "./datasets/tempoGAN/2d_train.mat"
    DATASET_PATH_VALID = "./datasets/tempoGAN/2d_valid.mat"
    OUTPUT_DIR = "./output_tempoGAN/" if args.output_dir is None else args.output_dir

    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # initialize parameters and import classes
    use_amp = True
    use_spatialdisc = True
    use_tempodisc = True

    weight_gl = [5.0, 0.0, 1.0]  # lambda_l1, lambda_l2, lambda_t
    weight_gl_layer = (
        [-1e-5, -1e-5, -1e-5, -1e-5, -1e-5] if use_spatialdisc else None
    )  # lambda_layer, lambda_layer1, lambda_layer2, lambda_layer3, lambda_layer4
    weight_dld = 1.0
    tile_ratio = 1

    gen_funcs = GenFuncs(weight_gl, weight_gl_layer)
    dics_funcs = DiscFuncs(weight_dld)
    data_funcs = DataFuncs(tile_ratio)

    # load dataset
    dataset_train = hdf5storage.loadmat(DATASET_PATH)
    dataset_valid = hdf5storage.loadmat(DATASET_PATH_VALID)

    # init Generator params
    in_channel = 1
    rb_channel0 = [2, 8, 8]
    rb_channel1 = [128, 128, 128]
    rb_channel2 = [32, 8, 8]
    rb_channel3 = [2, 1, 1]
    out_channels_list = [rb_channel0, rb_channel1, rb_channel2, rb_channel3]
    kernel_sizes_list = [[(5, 5)] * 2 + [(1, 1)]] * 4
    strides_list = [[1] * 3] * 4
    use_bns_list = [[True] * 3] * 3 + [[False] * 3]
    acts_list = [["relu", None, None]] * 4

    # define Generator model
    model_gen = ppsci.arch.Generator(
        ("in_d_low_inp",),  # 'NCHW'
        ("out_gen",),
        in_channel,
        out_channels_list,
        kernel_sizes_list,
        strides_list,
        use_bns_list,
        acts_list,
    )

    model_gen.register_input_transform(gen_funcs.transform_in)

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
    kernel_sizes = ((4, 4), (4, 4), (4, 4), (4, 4))
    strides = (2, 2, 2, 1)
    use_bns = (False, True, True, True)
    acts = ("leaky_relu", "leaky_relu", "leaky_relu", "leaky_relu", None)

    # define Discriminators
    if use_spatialdisc:
        output_keys_disc = (
            "out_disc_label_layer1",
            "out_disc_label_layer2",
            "out_disc_label_layer3",
            "out_disc_label_layer4",
            "out_disc_label",
            "out_disc_gen_out_layer1",
            "out_disc_gen_out_layer2",
            "out_disc_gen_out_layer3",
            "out_disc_gen_out_layer4",
            "out_disc_gen_out",
        )
        model_disc = ppsci.arch.Discriminator(
            ("in_d_high_disc_label", "in_d_high_disc_gen"),  # 'NCHW'
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

    # define Discriminators tempo
    if use_tempodisc:
        output_keys_disc_t = (
            "out_disc_t_label_layer1",
            "out_disc_t_label_layer2",
            "out_disc_t_label_layer3",
            "out_disc_t_label_layer4",
            "out_disc_t_label",
            "out_disc_t_gen_out_layer1",
            "out_disc_t_gen_out_layer2",
            "out_disc_t_gen_out_layer3",
            "out_disc_t_gen_out_layer4",
            "out_disc_t_gen_out",
        )
        model_disc_tempo = ppsci.arch.Discriminator(
            ("in_d_high_disc_t_label", "in_d_high_disc_t_gen"),  # 'NCHW'
            output_keys_disc_t,
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
    ITERS_PER_EPOCH = 2
    EPOCHS = 40000 if args.epochs is None else args.epochs
    EPOCHS_GEN = EPOCHS_DISC = EPOCHS_DISC_TEMPO = 1
    BATCH_SIZE = 8

    # initialize Adam optimizer
    lr_scheduler_gen = ppsci.optimizer.lr_scheduler.Step(
        epochs=EPOCHS,
        iters_per_epoch=ITERS_PER_EPOCH,
        learning_rate=2e-4,
        step_size=EPOCHS // 2,
        gamma=0.05,
        by_epoch=True,
    )()
    optimizer_gen = ppsci.optimizer.Adam(lr_scheduler_gen)(model_gen)
    if use_spatialdisc:
        lr_scheduler_disc = ppsci.optimizer.lr_scheduler.Step(
            EPOCHS, ITERS_PER_EPOCH, 2e-4, EPOCHS // 2, 0.05, by_epoch=True
        )()
        optimizer_disc = ppsci.optimizer.Adam(lr_scheduler_disc)(model_disc)
    if use_tempodisc:
        lr_scheduler_disc_tempo = ppsci.optimizer.lr_scheduler.Step(
            EPOCHS, ITERS_PER_EPOCH, 2e-4, EPOCHS // 2, 0.05, by_epoch=True
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
    if use_tempodisc:
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
    if use_spatialdisc:
        sup_constraint_disc = ppsci.constraint.SupervisedConstraint(
            {
                "dataset": {
                    "name": "NamedArrayDataset",
                    "input": {
                        "density_low": dataset_train["density_low"],
                        "density_high": dataset_train["density_high"],
                    },
                    "label": {
                        "out_disc_label": np.ones(
                            (np.shape(dataset_train["density_high"])[0], 1),
                            dtype=np.float32,
                        ),
                        "out_disc_gen_out": np.ones(
                            (np.shape(dataset_train["density_high"])[0], 1),
                            dtype=np.float32,
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

    # Discriminators tempo
    # maunally build constraint(s)
    if use_tempodisc:
        sup_constraint_disc_tempo = ppsci.constraint.SupervisedConstraint(
            {
                "dataset": {
                    "name": "NamedArrayDataset",
                    "input": {
                        "density_low": dataset_train["density_low_tempo"],
                        "density_high": dataset_train["density_high_tempo"],
                    },
                    "label": {
                        "out_disc_t_label": np.ones(
                            (np.shape(dataset_train["density_high_tempo"])[0], 1),
                            dtype=np.float32,
                        ),
                        "out_disc_t_gen_out": np.ones(
                            (np.shape(dataset_train["density_high_tempo"])[0], 1),
                            dtype=np.float32,
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
        OUTPUT_DIR,
        optimizer_gen,
        lr_scheduler_gen,
        EPOCHS_GEN,
        ITERS_PER_EPOCH,
        eval_during_train=False,
        use_amp=use_amp,
        amp_level="O2",
    )
    if use_spatialdisc:
        solver_disc = ppsci.solver.Solver(
            model_list,
            constraint_disc,
            OUTPUT_DIR,
            optimizer_disc,
            lr_scheduler_disc,
            EPOCHS_DISC,
            ITERS_PER_EPOCH,
            eval_during_train=False,
            use_amp=use_amp,
            amp_level="O2",
        )
    if use_tempodisc:
        solver_disc_tempo = ppsci.solver.Solver(
            model_list,
            constraint_disc_tempo,
            OUTPUT_DIR,
            optimizer_disc_tempo,
            lr_scheduler_disc_tempo,
            EPOCHS_DISC_TEMPO,
            ITERS_PER_EPOCH,
            eval_during_train=False,
            use_amp=use_amp,
            amp_level="O2",
        )

    PRED_INTERVAL = 200
    for i in range(1, EPOCHS + 1):
        ppsci.utils.logger.info(f"\nEpoch: {i}\n")
        # plotting during training
        if i == 1 or i % PRED_INTERVAL == 0:
            if i == 1:
                dics_funcs.model_gen = model_gen
            func_module.predict_and_save_plot(
                OUTPUT_DIR, i, solver_gen, dataset_valid, tile_ratio
            )

        dics_funcs.model_gen = model_gen
        # train disc, input: (x,y,G(x))
        if use_spatialdisc:
            solver_disc.train()

        # train disc tempo, input: (y_3,G(x)_3)
        if use_tempodisc:
            solver_disc_tempo.train()

        # train gen, input: (x,)
        solver_gen.train()

    ############### evaluation after training ###############
    img_label = (
        func_module.get_image_array(f"{OUTPUT_DIR}predict/label_epoch_1.png") / 255.0
    )
    img_pred = (
        func_module.get_image_array(f"{OUTPUT_DIR}predict/pred_epoch_{EPOCHS}.png")
        / 255.0
    )
    eval_mse, eval_psnr, eval_ssim = func_module.evaluate_img(img_label, img_pred)
    ppsci.utils.logger.info(f"MSE: {eval_mse}, PSNR: {eval_psnr}, SSIM: {eval_ssim}")
