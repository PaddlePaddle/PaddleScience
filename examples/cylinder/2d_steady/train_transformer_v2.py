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

import numpy as np
import paddle

import ppsci
from ppsci.arch import base


def build_embedding_model(embedding_model_path: str) -> ppsci.arch.CylinderEmbedding:
    input_keys = ["states", "visc"]
    output_keys = ["pred_states", "recover_states"]
    regularization_key = "k_matrix"
    model = ppsci.arch.CylinderEmbedding(input_keys, output_keys + [regularization_key])
    model.set_state_dict(paddle.load(embedding_model_path))
    return model


class OutputTransform(object):
    def __init__(self, model: base.NetBase):
        self.model = model
        self.model.eval()

    def __call__(self, x: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
        pred_embeds = x["pred_embeds"]
        pred_states = self.model.decoder(pred_embeds)
        # pred_states.shape=(B, T, C, H, W)
        return pred_states


if __name__ == "__main__":
    ppsci.utils.set_random_seed(42)

    num_layers = 6
    num_ctx = 16
    embed_size = 128
    num_heads = 4

    epochs = 200
    train_block_size = 16
    valid_block_size = 256
    input_keys = ["embeds"]
    output_keys = ["pred_embeds"]
    weights = [1.0]

    train_file_path = "/path/to/cylinder_training.hdf5"
    valid_file_path = "/path/to/cylinder_valid.hdf5"
    embedding_model_path = "./output/cylinder_enn/checkpoints/latest.pdparams"
    output_dir = "./output/cylinder_transformer"

    embedding_model = build_embedding_model(embedding_model_path)
    output_transform = OutputTransform(embedding_model)

    # maunally build constraint(s)
    train_dataloader = {
        "dataset": {
            "name": "CylinderDataset",
            "file_path": train_file_path,
            "block_size": train_block_size,
            "stride": 4,
            "embedding_model": embedding_model,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": True,
        },
        "batch_size": 4,
        "num_workers": 4,
        "use_shared_memory": False,
    }

    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_file_path,
        input_keys,
        output_keys,
        {},
        train_dataloader,
        ppsci.loss.MSELoss(),
        weight_dict={key: value for key, value in zip(output_keys, weights)},
        name="Sup",
    )
    constraint = {sup_constraint.name: sup_constraint}

    # set iters_per_epoch by dataloader length
    iters_per_epoch = len(constraint["Sup"].data_loader)

    # manually init model
    model = ppsci.arch.PhysformerGPT2(
        input_keys,
        output_keys,
        num_layers,
        num_ctx,
        embed_size,
        num_heads,
    )

    # init optimizer and lr scheduler
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=0.1)
    lr_scheduler = ppsci.optimizer.lr_scheduler.CosineWarmRestarts(
        epochs,
        iters_per_epoch,
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
    eval_dataloader = {
        "dataset": {
            "name": "CylinderDataset",
            "file_path": valid_file_path,
            "block_size": valid_block_size,
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
        "use_shared_memory": False,
    }

    mse_metric = ppsci.validate.SupervisedValidator(
        input_keys,
        output_keys,
        eval_dataloader,
        ppsci.loss.MSELoss(),
        metric={"MSE": ppsci.metric.MSE()},
        weight_dict={key: value for key, value in zip(output_keys, weights)},
        name="MSE_Metric",
    )
    validator = {mse_metric.name: mse_metric}

    # set visualizer(optional)
    states = mse_metric.data_loader.dataset.data
    embedding_data = mse_metric.data_loader.dataset.embedding_data

    vis_datas = {
        "embeds": embedding_data[:1, :-1],
        "states": states[:1, 1:],
    }

    visualizer = {
        "visulzie_states": ppsci.visualize.Visualizer2DPlot(
            vis_datas,
            ppsci.utils.misc.PrettyOrderedDict(
                [
                    ("target_ux", lambda d: d["states"][:, :, 0]),
                    ("pred_ux", lambda d: output_transform(d)[:, :, 0]),
                    ("target_uy", lambda d: d["states"][:, :, 1]),
                    ("pred_uy", lambda d: output_transform(d)[:, :, 1]),
                    ("target_p", lambda d: d["states"][:, :, 2]),
                    ("preds_p", lambda d: output_transform(d)[:, :, 2]),
                ]
            ),
            num_timestamps=10,
            stride=20,
            xticks=np.linspace(-2, 14, 9),
            yticks=np.linspace(-4, 4, 5),
            prefix="result_states",
        )
    }

    train_solver = ppsci.solver.Solver(
        "train",
        model,
        constraint,
        output_dir,
        optimizer,
        lr_scheduler,
        epochs,
        iters_per_epoch,
        eval_during_train=True,
        eval_freq=50,
        validator=validator,
        visualizer=visualizer,
    )
    train_solver.train()

    eval_solver = ppsci.solver.Solver(
        "eval",
        model,
        output_dir=output_dir,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=f"{output_dir}/checkpoints/latest",
    )
    eval_solver.eval()

    # visualize the prediction of final checkpoint
    eval_solver.visualize()
