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

import os
import sys
import random
import numpy as np

import paddle
from paddle.optimizer.lr import ExponentialDecay

from paddlescience.data import build_dataloader
from paddlescience.network.embedding_koopman import LorenzEmbedding
from paddlescience.algorithm.algorithm_trphysx import TrPhysx

import paddlescience as psci
from paddlescience import config

config.enable_visualdl()
# hyper parameters
seed = 12345

# dataset config
train_data_path = 'your data path/lorenz_training_rk.hdf5'
train_block_size = 16
train_stride = 16
train_batch_size = 512

valid_data_path = 'your data path/lorenz_valid_rk.hdf5'
valid_block_size = 32
valid_stride = 1025
valid_batch_size = 8

# embedding model config
state_dims = [3]
n_embd = 32

# optimize config
clip_norm = 0.1
learning_rate = 0.001
gamma = 0.995
weight_decay = 1e-8

# train config
max_epochs = 300
checkpoint_path = './output/trphysx/lorenz/enn/'


def main():

    # create train dataloader
    dataset_args = dict(
        file_path=train_data_path,
        block_size=train_block_size,
        stride=train_stride, )
    train_dataloader = build_dataloader(
        'LorenzDataset',
        mode='train',
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        dataset_args=dataset_args)
    # create test dataloader
    dataset_args = dict(
        file_path=valid_data_path,
        block_size=valid_block_size,
        stride=valid_stride,
        ndata=valid_batch_size, )
    valid_dataloader = build_dataloader(
        'LorenzDataset',
        mode='val',
        batch_size=valid_batch_size,
        shuffle=False,
        drop_last=False,
        dataset_args=dataset_args)

    # create model
    net = LorenzEmbedding(state_dims=state_dims, n_embd=n_embd)
    # set the mean and std of the dataset
    net.mu = paddle.to_tensor(train_dataloader.dataset.mu)
    net.std = paddle.to_tensor(train_dataloader.dataset.std)

    # optimizer for training
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=clip_norm)
    scheduler = ExponentialDecay(learning_rate=learning_rate, gamma=gamma)
    optimizer = paddle.optimizer.Adam(
        parameters=net.parameters(),
        learning_rate=scheduler,
        grad_clip=clip,
        weight_decay=weight_decay)

    algo = TrPhysx(net)

    solver = psci.solver.Solver(
        pde=None,
        algo=algo,
        opt=optimizer,
        data_driven=True,
        lr_scheduler=scheduler,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader, )

    solver.solve(
        num_epoch=max_epochs,
        checkpoint_freq=10,
        checkpoint_path=checkpoint_path)


if __name__ == '__main__':
    main()
