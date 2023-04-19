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
"""
This code is refer from: 
https://github.com/zabaras/transformer-physx
"""
################################ 导入相关的库 ###############################################
import random

import numpy as np
import paddle

import paddlescience as psci
from paddlescience import config
from paddlescience.algorithm.algorithm_trphysx import TrPhysx
from paddlescience.data import build_dataloader
from paddlescience.network.embedding_koopman import RosslerEmbedding
from paddlescience.network.physx_transformer import PhysformerGPT2
from paddlescience.optimizer.lr_sheduler import CosineAnnealingWarmRestarts
from paddlescience.visu import RosslerViz

config.enable_visualdl()

################################ 设置超参数 ##################################################
# hyper parameters
seed = 12345

# dataset config
train_data_path = "/path/to/rossler_training.hdf5"
train_block_size = 32
train_stride = 16
train_batch_size = 64
train_ndata = 2048

valid_data_path = "/path/to/rossler_valid.hdf5"
valid_block_size = 256
valid_stride = 1024
valid_batch_size = 16

# embedding model config
state_dims = [3]
n_embd = 32
embedding_model_params = "./output/trphysx/rossler/enn/dynamic_net_params_300.pdparams"

# transformer model config
n_layer = 4
n_ctx = 32
n_head = 4

# optimize config
clip_norm = 0.1
learning_rate = 0.001
T_0 = 14
T_mult = 2
eta_min = 1e-8
weight_decay = 1e-8

# train config
max_epochs = 200
checkpoint_path = "./output/trphysx/rossler/transformer/"


def set_seed(seed=12345):
    """Set random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def main():
    set_seed()
    ################################ 定义数据集 ##############################################
    dataset_args = dict(
        file_path=train_data_path,
        block_size=train_block_size,
        stride=train_stride,
        ndata=train_ndata,
    )
    train_dataloader = build_dataloader(
        "RosslerDataset",
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        dataset_args=dataset_args,
    )

    dataset_args = dict(
        file_path=valid_data_path,
        block_size=valid_block_size,
        stride=valid_stride,
        ndata=valid_batch_size,
    )
    valid_dataloader = build_dataloader(
        "RosslerDataset",
        batch_size=valid_batch_size,
        shuffle=False,
        drop_last=False,
        dataset_args=dataset_args,
    )

    ################################ 定义模型 ###############################################
    embedding_net = RosslerEmbedding(state_dims=state_dims, n_embd=n_embd)
    viz = RosslerViz(checkpoint_path)
    net = PhysformerGPT2(
        n_layer,
        n_ctx,
        n_embd,
        n_head,
        embedding_net,
        pretrained_model=embedding_model_params,
        viz=viz,
    )

    ################################ 优化器设置 ##############################################
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=clip_norm)
    scheduler = CosineAnnealingWarmRestarts(learning_rate, T_0, T_mult, eta_min=eta_min)
    optimizer = paddle.optimizer.Adam(
        parameters=net.parameters(),
        learning_rate=scheduler,
        grad_clip=clip,
        weight_decay=weight_decay,
    )

    ################################ 定义Solver并训练 #########################################
    algo = TrPhysx(net)

    solver = psci.solver.Solver(
        pde=None,
        algo=algo,
        opt=optimizer,
        data_driven=True,
        lr_scheduler=scheduler,
        lr_update_method="step",
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
    )

    solver.solve(
        num_epoch=max_epochs, checkpoint_freq=25, checkpoint_path=checkpoint_path
    )


if __name__ == "__main__":
    main()
