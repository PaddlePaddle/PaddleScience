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

from selectors import BaseSelector
from typing import Dict

import paddle

import ppsci
from ppsci.arch import base


def build_embedding_model(embedding_model_path: str) -> base.NetBase:
    input_keys = ["states"]
    output_keys = ["pred_states", "recover_states"]
    regularization_key = "k_matrix"
    model = ppsci.arch.LorenzEmbedding(input_keys, output_keys + [regularization_key])
    model.set_state_dict(paddle.load(embedding_model_path))
    return model


class InputTransform(object):
    def __init__(self, model: base.NetBase) -> None:
        self.model = model
        self.model.eval()

    def __call__(self, x: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
        input_keys = self.model.input_keys
        x = self.model.concat_to_tensor(x, input_keys, axis=-1)
        y = self.model.encoder(x)
        y = self.model.split_to_dict((y,), input_keys)
        return y


class OutputTransform(object):
    def __init__(self, model: base.NetBase) -> None:
        self.model = model
        self.model.eval()

    def __call__(self, x: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
        pred_embeds = x["pred_embeds"]
        pred_states = self.model.decoder(pred_embeds)

        return pred_states


if __name__ == "__main__":
    # train time-series: 2048    time-steps: 256    block-size: 64  stride: 64
    # valid time-series: 64      time-steps: 1024   block-size: 256 stride: 1024
    # test  time-series: 256     time-steps: 1024
    ppsci.utils.set_random_seed(42)

    num_layers = 4
    num_ctx = 64
    embed_size = 32
    num_heads = 4

    epochs = 200
    train_block_size = 64
    valid_block_size = 256
    input_keys = ["embeds"]
    output_keys = ["pred_embeds"]
    weights = [1.0]

    train_file_path = "your data path/lorenz_training_rk.hdf5"
    valid_file_path = "your data path/lorenz_valid_rk.hdf5"
    embedding_model_path = "./output/lorenz_enn/checkpoints/latest.pdparams"
    output_dir = "./output/lorenz_transformer"

    embedding_model = build_embedding_model(embedding_model_path)
    output_transform = OutputTransform(embedding_model)

    # maunally build constraint(s)
    train_dataloader_cfg = {
        "dataset": {
            "name": "LorenzDataset",
            "file_path": train_file_path,
            "block_size": train_block_size,
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
        "use_shared_memory": False,
    }

    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_file_path,
        input_keys,
        output_keys,
        {},
        train_dataloader_cfg,
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
            "name": "LorenzDataset",
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
        "embeds": embedding_data[:16, :-1, :],
        "states": states[:16, 1:, :],
    }

    visualizer = {
        "visulzie_states": ppsci.visualize.VisualizerScatter3D(
            vis_datas,
            {
                "pred_states": lambda d: output_transform(d),
                "states": lambda d: d["states"],
            },
            1,
            "result_states",
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
