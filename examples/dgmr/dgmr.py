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

"""
Reference: https://github.com/openclimatefix/skillful_nowcasting
"""
from os import path as osp

import hydra
import matplotlib.pyplot as plt
import numpy as np
import paddle
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger


def visualize(
    cfg: DictConfig,
    x: paddle.Tensor,
    y: paddle.Tensor,
    y_hat: paddle.Tensor,
    batch_idx: int,
) -> None:
    images = x[0]
    future_images = y[0]
    generated_images = y_hat[0]
    fig, axes = plt.subplots(2, 2)
    for i, ax in enumerate(axes.flat):
        alpha = images[i][0].numpy()
        alpha[alpha < 1] = 0
        alpha[alpha > 1] = 1
        ax.imshow(images[i].transpose([1, 2, 0]).numpy(), alpha=alpha, cmap="viridis")
        ax.axis("off")
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.savefig(osp.join(cfg.output_dir, "Input_Image_Stack_Frame.png"))
    fig, axes = plt.subplots(3, 3)
    for i, ax in enumerate(axes.flat):
        alpha = future_images[i][0].numpy()
        alpha[alpha < 1] = 0
        alpha[alpha > 1] = 1
        ax.imshow(
            future_images[i].transpose([1, 2, 0]).numpy(), alpha=alpha, cmap="viridis"
        )
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.savefig(osp.join(cfg.output_dir, "Target_Image_Frame.png"))
    fig, axes = plt.subplots(3, 3)
    for i, ax in enumerate(axes.flat):
        alpha = generated_images[i][0].numpy()
        alpha[alpha < 1] = 0
        alpha[alpha > 1] = 1
        ax.imshow(
            generated_images[i].transpose([1, 2, 0]).numpy(),
            alpha=alpha,
            cmap="viridis",
        )
        ax.axis("off")
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.savefig(osp.join(cfg.output_dir, "Generated_Image_Frame.png"))


def train(cfg: DictConfig):
    print("Not supported.")


def evaluate(cfg: DictConfig):
    # set model
    model = ppsci.arch.DGMR(**cfg.MODEL)
    # load evaluate data
    dataset = ppsci.data.dataset.DGMRDataset(**cfg.DATASET)
    val_loader = paddle.io.DataLoader(dataset, batch_size=cfg.DATALOADER.batch_size)
    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
    )
    solver.model.eval()

    # evaluate pretrained model
    d_loss = []
    g_loss = []
    grid_loss = []
    for batch_idx, batch in enumerate(val_loader):
        with paddle.no_grad():
            out_dict = solver.model.validation_step(batch, batch_idx)
            # visualize
            images, future_images = batch
            images = images.astype(dtype="float32")
            future_images = future_images.astype(dtype="float32")
            generated_images = solver.model.generator(images)
            visualize(cfg, images, future_images, generated_images, batch_idx)
        d_loss.append(out_dict[0])
        g_loss.append(out_dict[1])
        grid_loss.append(out_dict[2])
    logger.message(f"d_loss: {np.array(d_loss).mean()}")
    logger.message(f"g_loss: {np.array(g_loss).mean()}")
    logger.message(f"grid_loss: {np.array(grid_loss).mean()}")


@hydra.main(version_base=None, config_path="./conf", config_name="dgmr.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
