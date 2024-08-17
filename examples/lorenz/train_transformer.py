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
# 1. Train a embedding model by running train_enn.py.
# 2. Load pretrained embedding model and freeze it, then train a transformer model by running train_transformer.py.

# This file is for step2: training a transformer model, based on frozen pretrained embedding model.
# This file is based on PaddleScience/ppsci API.
from os import path as osp
from typing import Dict

import hydra
import paddle
from omegaconf import DictConfig

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


def train(cfg: DictConfig):
    # train time-series: 2048    time-steps: 256    block-size: 64  stride: 64
    # valid time-series: 64      time-steps: 1024   block-size: 256 stride: 1024
    # test  time-series: 256     time-steps: 1024
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    embedding_model = build_embedding_model(cfg.EMBEDDING_MODEL_PATH)
    output_transform = OutputTransform(embedding_model)

    # manually build constraint(s)
    train_dataloader_cfg = {
        "dataset": {
            "name": "LorenzDataset",
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.MODEL.output_keys,
            "file_path": cfg.TRAIN_FILE_PATH,
            "block_size": cfg.TRAIN_BLOCK_SIZE,
            "stride": 64,
            "embedding_model": embedding_model,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": True,
        },
        "batch_size": cfg.TRAIN.batch_size,
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
    model = ppsci.arch.PhysformerGPT2(**cfg.MODEL)

    # init optimizer and lr scheduler
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=0.1)
    lr_scheduler = ppsci.optimizer.lr_scheduler.CosineWarmRestarts(
        iters_per_epoch=ITERS_PER_EPOCH, **cfg.TRAIN.lr_scheduler
    )()
    optimizer = ppsci.optimizer.Adam(
        lr_scheduler, grad_clip=clip, **cfg.TRAIN.optimizer
    )(model)

    # manually build validator
    eval_dataloader_cfg = {
        "dataset": {
            "name": "LorenzDataset",
            "file_path": cfg.VALID_FILE_PATH,
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.MODEL.output_keys,
            "block_size": cfg.VALID_BLOCK_SIZE,
            "stride": 1024,
            "embedding_model": embedding_model,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "batch_size": cfg.EVAL.batch_size,
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
    vis_data = {
        "embeds": embedding_data[: cfg.VIS_DATA_NUMS, :-1, :],
        "states": states[: cfg.VIS_DATA_NUMS, 1:, :],
    }

    visualizer = {
        "visualize_states": ppsci.visualize.VisualizerScatter3D(
            vis_data,
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
        cfg.output_dir,
        optimizer,
        lr_scheduler,
        cfg.TRAIN.epochs,
        ITERS_PER_EPOCH,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_freq=cfg.TRAIN.eval_freq,
        validator=validator,
        visualizer=visualizer,
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()
    # visualize prediction after finished training
    solver.visualize()


def evaluate(cfg: DictConfig):
    # directly evaluate pretrained model(optional)
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    embedding_model = build_embedding_model(cfg.EMBEDDING_MODEL_PATH)
    output_transform = OutputTransform(embedding_model)

    # manually init model
    model = ppsci.arch.PhysformerGPT2(**cfg.MODEL)

    # manually build validator
    eval_dataloader_cfg = {
        "dataset": {
            "name": "LorenzDataset",
            "file_path": cfg.VALID_FILE_PATH,
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.MODEL.output_keys,
            "block_size": cfg.VALID_BLOCK_SIZE,
            "stride": 1024,
            "embedding_model": embedding_model,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "batch_size": cfg.EVAL.batch_size,
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
        "embeds": embedding_data[: cfg.VIS_DATA_NUMS, :-1, :],
        "states": states[: cfg.VIS_DATA_NUMS, 1:, :],
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
        output_dir=cfg.output_dir,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
    )
    solver.eval()
    # visualize prediction for pretrained model(optional)
    solver.visualize()


def export(cfg: DictConfig):
    # set model
    embedding_model = build_embedding_model(cfg.EMBEDDING_MODEL_PATH)
    model_cfg = {
        **cfg.MODEL,
        "embedding_model": embedding_model,
        "input_keys": ["states"],
        "output_keys": ["pred_states"],
    }
    model = ppsci.arch.PhysformerGPT2(**model_cfg)

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        pretrained_model_path=cfg.INFER.pretrained_model_path,
    )
    # export model
    from paddle.static import InputSpec

    input_spec = [
        {
            key: InputSpec([None, 255, 3], "float32", name=key)
            for key in model.input_keys
        },
    ]

    solver.export(input_spec, cfg.INFER.export_path)


def inference(cfg: DictConfig):
    from deploy.python_infer import pinn_predictor

    predictor = pinn_predictor.PINNPredictor(cfg)

    dataset_cfg = {
        "name": "LorenzDataset",
        "file_path": cfg.VALID_FILE_PATH,
        "input_keys": cfg.MODEL.input_keys,
        "label_keys": cfg.MODEL.output_keys,
        "block_size": cfg.VALID_BLOCK_SIZE,
        "stride": 1024,
    }

    dataset = ppsci.data.dataset.build_dataset(dataset_cfg)

    input_dict = {
        "states": dataset.data[: cfg.VIS_DATA_NUMS, :-1, :],
    }
    output_dict = predictor.predict(input_dict, cfg.INFER.batch_size)

    # mapping data to cfg.INFER.output_keys
    output_keys = ["pred_states"]
    output_dict = {
        store_key: output_dict[infer_key]
        for store_key, infer_key in zip(output_keys, output_dict.keys())
    }

    input_dict = {
        "states": dataset.data[: cfg.VIS_DATA_NUMS, 1:, :],
    }

    data_dict = {**input_dict, **output_dict}
    for i in range(cfg.VIS_DATA_NUMS):
        ppsci.visualize.save_plot_from_3d_dict(
            f"./lorenz_transformer_pred_{i}",
            {key: value[i] for key, value in data_dict.items()},
            ("states", "pred_states"),
        )


@hydra.main(version_base=None, config_path="./conf", config_name="transformer.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    elif cfg.mode == "export":
        export(cfg)
    elif cfg.mode == "infer":
        inference(cfg)
    else:
        raise ValueError(
            f"cfg.mode should in ['train', 'eval', 'export', 'infer'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()
