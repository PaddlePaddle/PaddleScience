# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Two-stage training
# 1. Train a embedding model by running train_enn_v2.py.
# 2. Load pretrained embedding model and freeze it, then train a transformer model by running train_transformer_v2.py.

# This file is for step2: training a transformer model, based on frozen pretrained embedding model.
# This file is based on PaddleScience/ppsci API.
from typing import Dict

import paddle

import ppsci
from ppsci.arch import base
from ppsci.utils import logger
from ppsci.utils import save_load


def build_embedding_model(embedding_model_path: str) -> ppsci.arch.LorenzEmbedding:
    input_keys = ("states",)
    output_keys = ("pred_states", "recover_states")
    regularization_key = "k_matrix"
    model = ppsci.arch.LorenzEmbedding(input_keys, output_keys + (regularization_key,))
    save_load.load_pretrain(model, embedding_model_path)
    return model


class OutputTransform(object):
    def __init__(self, model: base.Arch):
        self.model = model
        self.model.eval()

    def __call__(self, x: Dict[str, paddle.Tensor]):
        pred_embeds = x["pred_embeds"]
        pred_states = self.model.decoder(pred_embeds)

        return pred_states


if __name__ == "__main__":
    # train time-series: 2048    time-steps: 256    block-size: 64  stride: 64
    # valid time-series: 64      time-steps: 1024   block-size: 256 stride: 1024
    # test  time-series: 256     time-steps: 1024
    ppsci.utils.set_random_seed(42)

    NUM_LAYERS = 4
    NUM_CTX = 64
    EMBED_SIZE = 32
    NUM_HEADS = 4

    EPOCHS = 200
    TRAIN_BLOCK_SIZE = 64
    VALID_BLOCK_SIZE = 256
    input_keys = ("embeds",)
    output_keys = ("pred_embeds",)

    VIS_DATA_NUMS = 16

    TRAIN_FILE_PATH = "./datasets/lorenz_training_rk.hdf5"
    VALID_FILE_PATH = "./datasets/lorenz_valid_rk.hdf5"
    EMBEDDING_MODEL_PATH = "./output/lorenz_enn/checkpoints/latest"
    OUTPUT_DIR = "./output/lorenz_transformer"
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    embedding_model = build_embedding_model(EMBEDDING_MODEL_PATH)
    output_transform = OutputTransform(embedding_model)

    # maunally build constraint(s)
    train_dataloader_cfg = {
        "dataset": {
            "name": "LorenzDataset",
            "input_keys": input_keys,
            "label_keys": output_keys,
            "file_path": TRAIN_FILE_PATH,
            "block_size": TRAIN_BLOCK_SIZE,
            "stride": 64,
            "embedding_model": embedding_model,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": True,
        },
        "batch_size": 16,
        "num_workers": 4,
    }

    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        ppsci.loss.MSELoss(),
        name="Sup",
    )
    constraint = {sup_constraint.name: sup_constraint}

    # set iters_per_epoch by dataloader length
    ITERS_PER_EPOCH = len(constraint["Sup"].data_loader)

    # manually init model
    model = ppsci.arch.PhysformerGPT2(
        input_keys,
        output_keys,
        NUM_LAYERS,
        NUM_CTX,
        EMBED_SIZE,
        NUM_HEADS,
    )

    # init optimizer and lr scheduler
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=0.1)
    lr_scheduler = ppsci.optimizer.lr_scheduler.CosineWarmRestarts(
        EPOCHS,
        ITERS_PER_EPOCH,
        0.001,
        T_0=14,
        T_mult=2,
        eta_min=1e-9,
    )()
    optimizer = ppsci.optimizer.Adam(
        lr_scheduler,
        weight_decay=1e-8,
        grad_clip=clip,
    )([model])

    # maunally build validator
    eval_dataloader_cfg = {
        "dataset": {
            "name": "LorenzDataset",
            "file_path": VALID_FILE_PATH,
            "input_keys": input_keys,
            "label_keys": output_keys,
            "block_size": VALID_BLOCK_SIZE,
            "stride": 1024,
            "embedding_model": embedding_model,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "batch_size": 16,
        "num_workers": 4,
    }

    mse_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        ppsci.loss.MSELoss(),
        metric={"MSE": ppsci.metric.MSE()},
        name="MSE_Validator",
    )
    validator = {mse_validator.name: mse_validator}

    # set visualizer(optional)
    states = mse_validator.data_loader.dataset.data
    embedding_data = mse_validator.data_loader.dataset.embedding_data
    vis_datas = {
        "embeds": embedding_data[:VIS_DATA_NUMS, :-1, :],
        "states": states[:VIS_DATA_NUMS, 1:, :],
    }

    visualizer = {
        "visulzie_states": ppsci.visualize.VisualizerScatter3D(
            vis_datas,
            {
                "pred_states": lambda d: output_transform(d),
                "states": lambda d: d["states"],
            },
            num_timestamps=1,
            prefix="result_states",
        )
    }

    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        optimizer,
        lr_scheduler,
        EPOCHS,
        ITERS_PER_EPOCH,
        eval_during_train=True,
        eval_freq=50,
        validator=validator,
        visualizer=visualizer,
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()
    # visualize prediction after finished training
    solver.visualize()

    # directly evaluate pretrained model(optional)
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/eval.log", "info")
    solver = ppsci.solver.Solver(
        model,
        output_dir=OUTPUT_DIR,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=f"{OUTPUT_DIR}/checkpoints/latest",
    )
    solver.eval()
    # visualize prediction for pretrained model(optional)
    solver.visualize()
