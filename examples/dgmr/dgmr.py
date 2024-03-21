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
import paddle
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger
import numpy as np
import matplotlib.pyplot as plt

def visualize(cfg: DictConfig, x: paddle.Tensor, y: paddle.Tensor, y_hat:
    paddle.Tensor, batch_idx: int) ->None:
    images = x[0]
    future_images = y[0]
    generated_images = y_hat[0]
    fig, axes = plt.subplots(2, 2)
    for i, ax in enumerate(axes.flat):
        alpha = images[i][0].numpy()
        alpha[alpha < 1] = 0
        alpha[alpha > 1] = 1
        ax.imshow(images[i].transpose([1, 2, 0]).numpy(), alpha=alpha, cmap="viridis") 
        ax.axis('off')  
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.savefig(osp.join(cfg.output_dir, "Input_Image_Stack_Frame.png"))
    fig, axes = plt.subplots(3,3)
    for i, ax in enumerate(axes.flat):
        alpha = future_images[i][0].numpy() 
        alpha[alpha < 1] = 0
        alpha[alpha > 1] = 1
        ax.imshow(future_images[i].transpose([1, 2, 0]).numpy(), alpha=alpha, cmap="viridis")  
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.savefig(osp.join(cfg.output_dir, 'Target_Image_Frame.png'))
    fig, axes = plt.subplots(3, 3)
    for i, ax in enumerate(axes.flat):
        alpha = generated_images[i][0].numpy() 
        alpha[alpha < 1] = 0
        alpha[alpha > 1] = 1
        ax.imshow(generated_images[i].transpose([1, 2, 0]).numpy(), alpha=alpha, cmap="viridis") 
        ax.axis('off')  
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.savefig(osp.join(cfg.output_dir, 'Generated_Image_Frame.png'))

def validation(solver, batch, batch_idx):
    images, future_images = batch
    images_value = list(images.values())[0]
    future_images_value = list(future_images.values())[0]
    # Two discriminator steps per generator step
    for _ in range(2):
        predictions = solver.predict(images)
        predictions_value = list(predictions.values())[0]
        generated_sequence = paddle.concat(x=[images_value, predictions_value], axis=1)
        real_sequence = paddle.concat(x=[images_value, future_images_value], axis=1)
        concatenated_inputs = paddle.concat(x=[real_sequence,
            generated_sequence], axis=0)
        concatenated_outputs = solver.model.discriminator(concatenated_inputs)
        score_real, score_generated = paddle.split(x=
            concatenated_outputs, num_or_sections=[real_sequence.shape[
            0], generated_sequence.shape[0]], axis=0)
        score_real_spatial, score_real_temporal = paddle.split(x=
            score_real, num_or_sections=score_real.shape[1], axis=1)
        score_generated_spatial, score_generated_temporal = (paddle
            .split(x=score_generated, num_or_sections=score_generated.shape[1], axis=1))
        discriminator_loss = loss_hinge_disc(score_generated_spatial,
            score_real_spatial) + loss_hinge_disc(score_generated_temporal,
            score_real_temporal)

    predictions_value = [list(solver.predict(images).values())[0] for _ in range(6)]
    grid_cell_reg = grid_cell_regularizer(paddle.stack(x=predictions_value,
        axis=0), future_images_value)
    generated_sequence = [paddle.concat(x=[images_value, x], axis=1) for x in
        predictions_value]
    real_sequence = paddle.concat(x=[images_value, future_images_value], axis=1)
    generated_scores = []
    for g_seq in generated_sequence:
        concatenated_inputs = paddle.concat(x=[real_sequence, g_seq],
            axis=0)
        concatenated_outputs = solver.model.discriminator(concatenated_inputs)
        score_real, score_generated = paddle.split(x=
            concatenated_outputs, num_or_sections=[real_sequence.shape[
            0], g_seq.shape[0]], axis=0)
        generated_scores.append(score_generated)
    generator_disc_loss = loss_hinge_gen(paddle.concat(x=
        generated_scores, axis=0))
    generator_loss = generator_disc_loss + 20 * grid_cell_reg

    return discriminator_loss, generator_loss, grid_cell_reg

def loss_hinge_disc(score_generated, score_real):
    """Discriminator hinge loss."""
    l1 = paddle.nn.functional.relu(x=1.0 - score_real)
    loss = paddle.mean(x=l1)
    l2 = paddle.nn.functional.relu(x=1.0 + score_generated)
    loss += paddle.mean(x=l2)
    return loss


def loss_hinge_gen(score_generated):
    """Generator hinge loss."""
    loss = -paddle.mean(x=score_generated)
    return loss


def grid_cell_regularizer(generated_samples, batch_targets):
    """Grid cell regularizer.

    Args:
      generated_samples: Tensor of size [n_samples, batch_size, 18, 256, 256, 1].
      batch_targets: Tensor of size [batch_size, 18, 256, 256, 1].

    Returns:
      loss: A tensor of shape [batch_size].
    """
    gen_mean = paddle.mean(x=generated_samples, axis=0)
    weights = paddle.clip(x=batch_targets, min=0.0, max=24.0)
    loss = paddle.mean(x=paddle.abs(x=gen_mean - batch_targets) * weights)
    return loss

def train(cfg: DictConfig):
    print("Not supported.")

def evaluate(cfg: DictConfig):
    # set model
    model = ppsci.arch.DGMR(**cfg.MODEL)
    # load evaluate data
    dataset = ppsci.data.dataset.DGMRDataset(**cfg.DATASET)
    val_loader = paddle.io.DataLoader(
        dataset,
        batch_size=cfg.DATALOADER.batch_size
    )
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
            out_dict = validation(solver, batch, batch_idx)
            
            # visualize
            images = batch[0]['input_frames']
            future_images = batch[1]['target_frames']
            generated_images = solver.predict(batch[0])['future_images']
            visualize(cfg, images, future_images, generated_images, batch_idx)

        d_loss.append(out_dict[0])
        g_loss.append(out_dict[1])
        grid_loss.append(out_dict[2])
    logger.message(f'd_loss: {np.array(d_loss).mean()}')
    logger.message(f'g_loss: {np.array(g_loss).mean()}')
    logger.message(f'grid_loss: {np.array(grid_loss).mean()}')

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