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
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import paddle
import paddle.nn.functional as F
from matplotlib import image as Img
from PIL import Image
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

import ppsci
from ppsci.utils import logger


# train
def interpolate(
    data: paddle.Tensor, ratio: int, mode: str = "nearest"
) -> paddle.Tensor:
    """Interpolate twice.

    Args:
        data (paddle.Tensor): The data to be interpolated.
        ratio (int): Ratio of one interpolation.
        mode (str, optional): Interpolation method. Defaults to "nearest".

    Returns:
        paddle.Tensor: Data interpolated.
    """
    for _ in range(2):
        data = F.interpolate(
            data,
            [data.shape[-2] * ratio, data.shape[-1] * ratio],
            mode=mode,
        )
    return data


def reshape_input(input_dict: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
    """Reshape input data for temporally Discriminator. Reshape data from N, C, W, H to N * C, 1, H, W.
        Which will merge N dimension and C dimension to 1 dimension but still keep 4 dimensions
        to ensure the data can be used for training.

    Args:
        input_dict (Dict[str, paddle.Tensor]): input data dict.

    Returns:
        Dict[str, paddle.Tensor]: reshaped data dict.
    """
    for key in input_dict:
        input = input_dict[key]
        N, C, H, W = input.shape
        input_dict[key] = paddle.reshape(input, [N * C, 1, H, W])
    return input_dict


def dereshape_input(
    input_dict: Dict[str, paddle.Tensor], C: int
) -> Dict[str, paddle.Tensor]:
    """Dereshape input data for temporally Discriminator. Deeshape data from N * C, 1, H, W to N, C, W, H.

    Args:
        input_dict (Dict[str, paddle.Tensor]): input data dict.
        C (int): Channel of dereshape.

    Returns:
        Dict[str, paddle.Tensor]: dereshaped data dict.
    """
    for key in input_dict:
        input = input_dict[key]
        N, _, H, W = input.shape
        if N < C:
            logger.warning(
                f"batch_size is smaller than {C}! Tempo needs at least {C} frames, input will be copied."
            )
            input_dict[key] = paddle.concat([input[:1]] * C, axis=1)
        else:
            N_new = int(N // C)
            input_dict[key] = paddle.reshape(input[: N_new * C], [-1, C, H, W])
    return input_dict


# predict
def split_data(data: np.ndarray, tile_ratio: int) -> np.ndarray:
    """Split a numpy image to tiles equally.

    Args:
        data (np.ndarray): The image to be splited.
        tile_ratio (int): How many tiles of one dim.
            Number of result tiles is tile_ratio * tile_ratio for a 2d image.

    Returns:
        np.ndarray: Tiles in [N,C,H,W] shape.
    """
    _, _, h, w = data.shape
    tile_h, tile_w = h // tile_ratio, w // tile_ratio
    tiles = []
    for i in range(tile_ratio):
        for j in range(tile_ratio):
            tiles.append(
                data[
                    :1,
                    :,
                    i * tile_h : i * tile_h + tile_h,
                    j * tile_w : j * tile_w + tile_w,
                ],
            )
    return np.concatenate(tiles, axis=0)


def concat_data(data: np.ndarray, tile_ratio: int) -> np.ndarray:
    """Concat numpy tiles to a image equally.

    Args:
        data (np.ndarray): The tiles to be upsplited.
        tile_ratio (int): How many tiles of one dim.
            Number of input tiles is tile_ratio * tile_ratio for 2d result.

    Returns:
        np.ndarray: Image in [H,W] shape.
    """
    _, _, tile_h, tile_w = data.shape
    h, w = tile_h * tile_ratio, tile_w * tile_ratio
    data_whole = np.ones([h, w], dtype=paddle.get_default_dtype())
    tile_idx = 0
    for i in range(tile_ratio):
        for j in range(tile_ratio):
            data_whole[
                i * tile_h : i * tile_h + tile_h,
                j * tile_w : j * tile_w + tile_w,
            ] = data[tile_idx][0]
            tile_idx += 1
    return data_whole


def predict_and_save_plot(
    OUTPUT_DIR: str,
    epoch_id: int,
    solver_gen: ppsci.solver.Solver,
    dataset_valid: np.ndarray,
    tile_ratio: int = 1,
):
    """Predicting and plotting.

    Args:
        OUTPUT_DIR (str): Output dir path.
        epoch_id (int): Which epoch it is.
        solver_gen (ppsci.solver.Solver): Solver for predicting.
        dataset_valid (np.ndarray): Valid dataset.
        tile_ratio (int, optional): How many tiles of one dim. Defaults to 1.
    """
    dir_pred = "predict/"
    os.makedirs(f"{OUTPUT_DIR}{dir_pred}", exist_ok=True)

    start_idx = 190
    density_low = dataset_valid["density_low"][start_idx : start_idx + 3]
    density_high = dataset_valid["density_high"][start_idx : start_idx + 3]

    # tile
    density_low = (
        split_data(density_low, tile_ratio) if tile_ratio != 1 else density_low
    )
    density_high = (
        split_data(density_high, tile_ratio) if tile_ratio != 1 else density_high
    )

    pred_dict = solver_gen.predict(
        {
            "density_low": density_low,
            "density_high": density_high,
        },
        {"density_high": lambda out: out["output_gen"]},
        batch_size=tile_ratio * tile_ratio if tile_ratio != 1 else 3,
        no_grad=False,
    )
    if epoch_id == 1:
        # plot interpolated input image
        input_img = np.expand_dims(dataset_valid["density_low"][start_idx], axis=0)
        input_img = paddle.to_tensor(input_img, dtype=paddle.get_default_dtype())
        input_img = F.interpolate(
            input_img,
            [input_img.shape[-2] * 4, input_img.shape[-1] * 4],
            mode="nearest",
        ).numpy()
        Img.imsave(
            os.path.join(OUTPUT_DIR, dir_pred, "input.png"),
            np.squeeze(input_img),
            vmin=0.0,
            vmax=1.0,
            cmap="gray",
        )
        # plot target image
        Img.imsave(
            os.path.join(OUTPUT_DIR, dir_pred, "target.png"),
            np.squeeze(dataset_valid["density_high"][start_idx]),
            vmin=0.0,
            vmax=1.0,
            cmap="gray",
        )
    # plot pred image
    pred_img = (
        concat_data(pred_dict["density_high"].numpy(), tile_ratio)
        if tile_ratio != 1
        else np.squeeze(pred_dict["density_high"][0].numpy())
    )
    Img.imsave(
        os.path.join(OUTPUT_DIR, dir_pred, f"pred_epoch_{str(epoch_id)}.png"),
        pred_img,
        vmin=0.0,
        vmax=1.0,
        cmap="gray",
    )


# evaluation
def evaluate_img(
    img_target: np.ndarray, img_pred: np.ndarray
) -> Union[float, float, float]:
    """Evaluate two images.

    Args:
        img_target (np.ndarray): Target image.
        img_pred (np.ndarray): Image generated by prediction.

    Returns:
        Union[float, float, float]: MSE, PSNR, SSIM.
    """
    eval_mse = mean_squared_error(img_target, img_pred)
    eval_psnr = peak_signal_noise_ratio(img_target, img_pred)
    eval_ssim = structural_similarity(img_target, img_pred, data_range=1.0)
    return eval_mse, eval_psnr, eval_ssim


def get_image_array(img_path):
    return np.array(Image.open(img_path).convert("L"))


class GenFuncs:
    """All functions used for Generator, including functions of transform and loss.

    Args:
        weight_gen (List[float]): Weights of L1 loss.
        weight_gen_layer (List[float], optional): Weights of layers loss. Defaults to None.
    """

    def __init__(
        self, weight_gen: List[float], weight_gen_layer: List[float] = None
    ) -> None:
        self.weight_gen = weight_gen
        self.weight_gen_layer = weight_gen_layer

    def transform_in(self, _in):
        ratio = 2
        input_dict = reshape_input(_in)
        density_low = input_dict["density_low"]
        density_low_inp = interpolate(density_low, ratio, "nearest")
        return {"input_gen": density_low_inp}

    def loss_func_gen(self, output_dict: Dict, *args) -> paddle.Tensor:
        """Calculate loss of generator when use spatial discraminitor.
            The loss consists of l1 loss, l2 loss and layer loss when use spatial discraminitor.
            Notice that all item of loss is optional because weight of them might be 0.

        Args:
            output_dict (Dict): output dict of model.

        Returns:
            paddle.Tensor: Loss of generator.
        """
        # l1 loss
        loss_l1 = F.l1_loss(
            output_dict["output_gen"], output_dict["density_high"], "mean"
        )
        losses = loss_l1 * self.weight_gen[0]

        # l2 loss
        loss_l2 = F.mse_loss(
            output_dict["output_gen"], output_dict["density_high"], "mean"
        )
        losses += loss_l2 * self.weight_gen[1]

        if self.weight_gen_layer is not None:
            # disc(generator_out) loss
            out_disc_from_gen = output_dict["out_disc_from_gen"][-1]
            label_ones = paddle.ones_like(out_disc_from_gen)
            loss_gen = F.binary_cross_entropy_with_logits(
                out_disc_from_gen, label_ones, reduction="mean"
            )
            losses += loss_gen

            # layer loss
            key_list = list(output_dict.keys())
            # ["out0_layer0","out0_layer1","out0_layer2","out0_layer3","out_disc_from_target",
            # "out1_layer0","out1_layer1","out1_layer2","out1_layer3","out_disc_from_gen"]
            loss_layer = 0
            for i in range(1, len(self.weight_gen_layer)):
                # i = 0,1,2,3
                loss_layer += (
                    self.weight_gen_layer[i]
                    * F.mse_loss(
                        output_dict[key_list[i]],
                        output_dict[key_list[5 + i]],
                        reduction="sum",
                    )
                    / 2
                )
            losses += loss_layer * self.weight_gen_layer[0]

        return losses

    def loss_func_gen_tempo(self, output_dict: Dict, *args) -> paddle.Tensor:
        """Calculate loss of generator when use temporal discraminitor.
            The loss is cross entropy loss when use temporal discraminitor.

        Args:
            output_dict (Dict): output dict of model.

        Returns:
            paddle.Tensor: Loss of generator.
        """
        out_disc_tempo_from_gen = output_dict["out_disc_tempo_from_gen"][-1]
        label_t_ones = paddle.ones_like(out_disc_tempo_from_gen)

        loss_gen_t = F.binary_cross_entropy_with_logits(
            out_disc_tempo_from_gen, label_t_ones, reduction="mean"
        )
        losses = loss_gen_t * self.weight_gen[2]
        return losses


class DiscFuncs:
    """All functions used for Discriminator and temporally Discriminator, including functions of transform and loss.

    Args:
        weight_disc (float): Weight of loss generated by the discriminator to judge the true target.
    """

    def __init__(self, weight_disc: float) -> None:
        self.weight_disc = weight_disc
        self.model_gen = None

    def transform_in(self, _in):
        ratio = 2
        input_dict = reshape_input(_in)
        density_low = input_dict["density_low"]
        density_high_from_target = input_dict["density_high"]

        density_low_inp = interpolate(density_low, ratio, "nearest")

        density_high_from_gen = self.model_gen(input_dict)["output_gen"]
        density_high_from_gen.stop_gradient = True

        density_input_from_target = paddle.concat(
            [density_low_inp, density_high_from_target], axis=1
        )
        density_input_from_gen = paddle.concat(
            [density_low_inp, density_high_from_gen], axis=1
        )
        return {
            "input_disc_from_target": density_input_from_target,
            "input_disc_from_gen": density_input_from_gen,
        }

    def transform_in_tempo(self, _in):
        density_high_from_target = _in["density_high"]

        input_dict = reshape_input(_in)
        density_high_from_gen = self.model_gen(input_dict)["output_gen"]
        density_high_from_gen.stop_gradient = True

        input_trans = {
            "input_tempo_disc_from_target": density_high_from_target,
            "input_tempo_disc_from_gen": density_high_from_gen,
        }

        return dereshape_input(input_trans, 3)

    def loss_func(self, output_dict, *args):
        out_disc_from_target = output_dict["out_disc_from_target"]
        out_disc_from_gen = output_dict["out_disc_from_gen"]

        label_ones = paddle.ones_like(out_disc_from_target)
        label_zeros = paddle.zeros_like(out_disc_from_gen)

        loss_disc_from_target = F.binary_cross_entropy_with_logits(
            out_disc_from_target, label_ones, reduction="mean"
        )
        loss_disc_from_gen = F.binary_cross_entropy_with_logits(
            out_disc_from_gen, label_zeros, reduction="mean"
        )
        losses = loss_disc_from_target * self.weight_disc + loss_disc_from_gen
        return losses

    def loss_func_tempo(self, output_dict, *args):
        out_disc_tempo_from_target = output_dict["out_disc_tempo_from_target"]
        out_disc_tempo_from_gen = output_dict["out_disc_tempo_from_gen"]

        label_ones = paddle.ones_like(out_disc_tempo_from_target)
        label_zeros = paddle.zeros_like(out_disc_tempo_from_gen)

        loss_disc_tempo_from_target = F.binary_cross_entropy_with_logits(
            out_disc_tempo_from_target, label_ones, reduction="mean"
        )
        loss_disc_tempo_from_gen = F.binary_cross_entropy_with_logits(
            out_disc_tempo_from_gen, label_zeros, reduction="mean"
        )
        losses = (
            loss_disc_tempo_from_target * self.weight_disc + loss_disc_tempo_from_gen
        )
        return losses


class DataFuncs:
    """All functions used for data transform.

    Args:
        tile_ratio (int, optional): How many tiles of one dim. Defaults to 1.
        density_min (float, optional): Minimize density of one tile. Defaults to 0.02.
        max_turn (int, optional): Maximize turn of taking a tile from one image. Defaults to 20.
    """

    def __init__(
        self, tile_ratio: int = 1, density_min: float = 0.02, max_turn: int = 20
    ) -> None:
        self.tile_ratio = tile_ratio
        self.density_min = density_min
        self.max_turn = max_turn

    def transform(
        self,
        input_item: Dict[str, np.ndarray],
        label_item: Dict[str, np.ndarray],
        weight_item: Dict[str, np.ndarray],
    ) -> Union[
        Dict[str, paddle.Tensor], Dict[str, paddle.Tensor], Dict[str, paddle.Tensor]
    ]:
        if self.tile_ratio == 1:
            return input_item, label_item, weight_item
        for _ in range(self.max_turn):
            rand_ratio = np.random.rand()
            density_low = self.cut_data(input_item["density_low"], rand_ratio)
            density_high = self.cut_data(input_item["density_high"], rand_ratio)
            if self.is_valid_tile(density_low):
                break

        input_item["density_low"] = density_low
        input_item["density_high"] = density_high
        return input_item, label_item, weight_item

    def cut_data(self, data: np.ndarray, rand_ratio: float) -> paddle.Tensor:
        # data: C,H,W
        _, H, W = data.shape
        if H % self.tile_ratio != 0 or W % self.tile_ratio != 0:
            exit(
                f"ERROR: input images cannot be divided into {self.tile_ratio} parts evenly!"
            )
        tile_shape = [H // self.tile_ratio, W // self.tile_ratio]
        rand_shape = np.floor(rand_ratio * (np.array([H, W]) - np.array(tile_shape)))
        start = [int(rand_shape[0]), int(rand_shape[1])]
        end = [int(rand_shape[0] + tile_shape[0]), int(rand_shape[1] + tile_shape[1])]
        data = paddle.slice(
            paddle.to_tensor(data), axes=[-2, -1], starts=start, ends=end
        )

        return data

    def is_valid_tile(self, tile: paddle.Tensor):
        img_density = tile[0].sum()
        return img_density >= (
            self.density_min * tile.shape[0] * tile.shape[1] * tile.shape[2]
        )
