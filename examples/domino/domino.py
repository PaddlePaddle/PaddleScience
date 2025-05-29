# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# refs: https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/external_aerodynamics/domino

import multiprocessing
import os
import re
import time

import hydra
import numpy as np
import paddle
import paddle.distributed as dist
import pyvista as pv
import vtk
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from omegaconf import OmegaConf
from paddle import DataParallel
from paddle.amp import GradScaler
from paddle.amp import auto_cast
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler
from scipy.spatial import KDTree
from vtk.util import numpy_support

from ppsci.arch.physicsnemo import DoMINO
from ppsci.arch.physicsnemo import create_directory
from ppsci.arch.physicsnemo import get_fields
from ppsci.arch.physicsnemo import get_node_to_elem
from ppsci.arch.physicsnemo import get_volume_data
from ppsci.arch.physicsnemo import load_checkpoint
from ppsci.arch.physicsnemo import mean_std_sampling
from ppsci.arch.physicsnemo import save_checkpoint
from ppsci.arch.physicsnemo import write_to_vtp
from ppsci.arch.physicsnemo import write_to_vtu
from ppsci.data.dataset.domino_datapipe import DoMINODataPipe
from ppsci.data.dataset.domino_datapipe import OpenFoamDataset
from ppsci.data.dataset.domino_datapipe import cal_normal_positional_encoding
from ppsci.data.dataset.domino_datapipe import calculate_center_of_mass
from ppsci.data.dataset.domino_datapipe import create_grid
from ppsci.data.dataset.domino_datapipe import get_filenames
from ppsci.data.dataset.domino_datapipe import normalize
from ppsci.data.dataset.domino_datapipe import unnormalize
from ppsci.data.process.openfoam import process_files
from ppsci.utils.sdf import signed_distance_field

AIR_DENSITY = 1.205
STREAM_VELOCITY = 30.00

paddle.set_device("gpu")


def process(cfg: DictConfig):
    print(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")
    volume_variable_names = list(cfg.variables.volume.solution.keys())
    num_vol_vars = 0
    for j in volume_variable_names:
        if cfg.variables.volume.solution[j] == "vector":
            num_vol_vars += 3
        else:
            num_vol_vars += 1

    surface_variable_names = list(cfg.variables.surface.solution.keys())
    num_surf_vars = 0
    for j in surface_variable_names:
        if cfg.variables.surface.solution[j] == "vector":
            num_surf_vars += 3
        else:
            num_surf_vars += 1

    fm_data = OpenFoamDataset(
        cfg.data_processor.input_dir,
        kind=cfg.data_processor.kind,
        volume_variables=volume_variable_names,
        surface_variables=surface_variable_names,
        model_type=cfg.model.model_type,
    )
    output_dir = cfg.data_processor.output_dir
    create_directory(output_dir)  # noqa: F405
    n_processors = cfg.data_processor.num_processors

    num_files = len(fm_data)
    ids = np.arange(num_files)
    num_elements = int(num_files / n_processors) + 1
    process_list = []
    ctx = multiprocessing.get_context("spawn")
    for i in range(n_processors):
        if i != n_processors - 1:
            sf = ids[i * num_elements : i * num_elements + num_elements]
        else:
            sf = ids[i * num_elements :]
        # print(sf)
        process = ctx.Process(target=process_files, args=(sf, i, fm_data, output_dir))

        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()


def relative_loss_fn(output, target, padded_value=-10):
    mask = abs(target - padded_value) > 1e-3
    mask = mask.to(dtype=output.dtype)
    masked_loss = paddle.sum(((output - target) ** 2.0) * mask, (0, 1)) / paddle.sum(
        mask, (0, 1)
    )
    masked_truth = paddle.sum(((target) ** 2.0) * mask, (0, 1)) / paddle.sum(
        mask, (0, 1)
    )
    loss = paddle.mean(masked_loss / masked_truth)
    return loss


def mse_loss_fn(output, target, padded_value=-10):
    mask = abs(target - padded_value) > 1e-3
    mask = mask.to(dtype=output.dtype)
    masked_loss = paddle.sum(((output - target) ** 2.0) * mask, (0, 1)) / paddle.sum(
        mask, (0, 1)
    )
    loss = paddle.mean(masked_loss)
    return loss


def mse_loss_fn_surface(output, target, normals, padded_value=-10):
    masked_loss_pres = paddle.mean(
        ((output[:, :, :1] - target[:, :, :1]) ** 2.0), (0, 1)
    )

    ws_x_true = target[:, :, 1:2]
    ws_x_pred = output[:, :, 1:2]
    masked_loss_ws_x = paddle.mean(((ws_x_pred - ws_x_true) ** 2.0), (0, 1))

    ws_y_true = target[:, :, 2:3]
    ws_y_pred = output[:, :, 2:3]
    masked_loss_ws_y = paddle.mean(((ws_y_pred - ws_y_true) ** 2.0), (0, 1))

    ws_z_true = target[:, :, 3:4]
    ws_z_pred = output[:, :, 3:4]
    masked_loss_ws_z = paddle.mean(((ws_z_pred - ws_z_true) ** 2.0), (0, 1))

    loss = (
        paddle.mean(masked_loss_pres)
        + paddle.mean(masked_loss_ws_x)
        + paddle.mean(masked_loss_ws_y)
        + paddle.mean(masked_loss_ws_z)
    )
    loss = loss / 4
    return loss


def relative_loss_fn_surface(output, target, normals, padded_value=-10):
    masked_loss_pres = paddle.mean(
        ((output[:, :, :1] - target[:, :, :1]) ** 2.0), (0, 1)
    ) / paddle.mean(((target[:, :, :1]) ** 2.0), (0, 1))

    ws_x_true = target[:, :, 1:2]
    ws_x_pred = output[:, :, 1:2]
    masked_loss_ws_x = paddle.mean(
        ((ws_x_pred - ws_x_true) ** 2.0), (0, 1)
    ) / paddle.mean(((ws_x_true) ** 2.0), (0, 1))

    ws_y_true = target[:, :, 2:3]
    ws_y_pred = output[:, :, 2:3]
    masked_loss_ws_y = paddle.mean(
        ((ws_y_pred - ws_y_true) ** 2.0), (0, 1)
    ) / paddle.mean(((ws_y_true) ** 2.0), (0, 1))

    ws_z_true = target[:, :, 3:4]
    ws_z_pred = output[:, :, 3:4]
    masked_loss_ws_z = paddle.mean(
        ((ws_z_pred - ws_z_true) ** 2.0), (0, 1)
    ) / paddle.mean(((ws_z_true) ** 2.0), (0, 1))

    loss = (
        paddle.mean(masked_loss_pres)
        + paddle.mean(masked_loss_ws_x)
        + paddle.mean(masked_loss_ws_y)
        + paddle.mean(masked_loss_ws_z)
    )
    loss = loss / 4
    return loss


def relative_loss_fn_area(output, target, normals, area, padded_value=-10):
    scale_factor = 1.0  # Get this from the dataset
    area = area * 10**4
    pres_x_true = target[:, :, :1] * normals[:, :, 0:1] * area * scale_factor**2.0
    pres_x_pred = output[:, :, :1] * normals[:, :, 0:1] * area * scale_factor**2.0

    masked_loss_pres_x = paddle.mean(
        ((pres_x_pred - pres_x_true) ** 2.0), (0, 1)
    ) / paddle.mean(((pres_x_true) ** 2.0), (0, 1))

    ws_x_true = target[:, :, 1:2] * area * scale_factor**2.0
    ws_x_pred = output[:, :, 1:2] * area * scale_factor**2.0
    masked_loss_ws_x = paddle.mean(
        ((ws_x_pred - ws_x_true) ** 2.0), (0, 1)
    ) / paddle.mean(((ws_x_true) ** 2.0), (0, 1))

    ws_y_true = target[:, :, 2:3] * area * scale_factor**2.0
    ws_y_pred = output[:, :, 2:3] * area * scale_factor**2.0
    masked_loss_ws_y = paddle.mean(
        ((ws_y_pred - ws_y_true) ** 2.0), (0, 1)
    ) / paddle.mean(((ws_y_true) ** 2.0), (0, 1))

    ws_z_true = target[:, :, 3:4] * area * scale_factor**2.0
    ws_z_pred = output[:, :, 3:4] * area * scale_factor**2.0
    masked_loss_ws_z = paddle.mean(
        ((ws_z_pred - ws_z_true) ** 2.0), (0, 1)
    ) / paddle.mean(((ws_z_true) ** 2.0), (0, 1))

    loss = (
        paddle.mean(masked_loss_pres_x)
        + paddle.mean(masked_loss_ws_x)
        + paddle.mean(masked_loss_ws_y)
        + paddle.mean(masked_loss_ws_z)
    )
    loss = loss / 4
    return loss


def mse_loss_fn_area(output, target, normals, area, padded_value=-10):
    scale_factor = 1.0  # Get this from the dataset
    area = area * 10**4

    pres_x_true = target[:, :, :1] * normals[:, :, 0:1] * area * scale_factor**2.0
    pres_x_pred = output[:, :, :1] * normals[:, :, 0:1] * area * scale_factor**2.0

    masked_loss_pres_x = paddle.mean(((pres_x_pred - pres_x_true) ** 2.0), (0, 1))

    ws_x_true = target[:, :, 1:2] * area * scale_factor**2.0
    ws_x_pred = output[:, :, 1:2] * area * scale_factor**2.0
    masked_loss_ws_x = paddle.mean(((ws_x_pred - ws_x_true) ** 2.0), (0, 1))

    ws_y_true = target[:, :, 2:3] * area * scale_factor**2.0
    ws_y_pred = output[:, :, 2:3] * area * scale_factor**2.0
    masked_loss_ws_y = paddle.mean(((ws_y_pred - ws_y_true) ** 2.0), (0, 1))

    ws_z_true = target[:, :, 3:4] * area * scale_factor**2.0
    ws_z_pred = output[:, :, 3:4] * area * scale_factor**2.0
    masked_loss_ws_z = paddle.mean(((ws_z_pred - ws_z_true) ** 2.0), (0, 1))

    loss = (
        paddle.mean(masked_loss_pres_x)
        + paddle.mean(masked_loss_ws_x)
        + paddle.mean(masked_loss_ws_y)
        + paddle.mean(masked_loss_ws_z)
    )
    loss = loss / 4
    return loss


def integral_loss_fn(output, target, area, normals, padded_value=-10):
    vel_inlet = 30.0  # Get this from the dataset
    mask = abs(target - padded_value) > 1e-3
    mask = mask.to(dtype=output.dtype)
    area = paddle.unsqueeze(area, -1)
    output_true = target * mask * area * (vel_inlet) ** 2.0
    output_pred = output * mask * area * (vel_inlet) ** 2.0

    output_true[:, :, 0] = output_true[:, :, 0] * normals[:, :, 0]
    output_pred[:, :, 0] = output_pred[:, :, 0] * normals[:, :, 0]

    masked_pred = paddle.sum(output_pred, (1))
    masked_truth = paddle.sum(output_true, (1))

    loss = (masked_pred - masked_truth) ** 2.0
    loss = paddle.mean(loss)
    return loss


def integral_loss_fn_new(output, target, area, normals, padded_value=-10):
    drag_loss = drag_loss_fn(output, target, area, normals, padded_value=-10)
    lift_loss = lift_loss_fn(output, target, area, normals, padded_value=-10)
    return lift_loss + drag_loss


def lift_loss_fn(output, target, area, normals, padded_value=-10):
    vel_inlet = 30.0  # Get this from the dataset
    mask = abs(target - padded_value) > 1e-3
    mask = mask.to(dtype=output.dtype)
    area = paddle.unsqueeze(area, -1)
    output_true = target * mask * area * (vel_inlet) ** 2.0
    output_pred = output * mask * area * (vel_inlet) ** 2.0

    pres_true = output_true[:, :, 0] * normals[:, :, 2]
    pres_pred = output_pred[:, :, 0] * normals[:, :, 2]

    wz_true = output_true[:, :, -1]
    wz_pred = output_pred[:, :, -1]

    masked_pred = paddle.sum(pres_pred + wz_pred, (1)) / (
        paddle.sum(area) * (vel_inlet) ** 2.0
    )
    masked_truth = paddle.sum(pres_true + wz_true, (1)) / (
        paddle.sum(area) * (vel_inlet) ** 2.0
    )

    loss = (masked_pred - masked_truth) ** 2.0
    loss = paddle.mean(loss)
    return loss


def drag_loss_fn(output, target, area, normals, padded_value=-10):
    vel_inlet = 30.0  # Get this from the dataset
    mask = abs(target - padded_value) > 1e-3
    mask = mask.to(dtype=output.dtype)
    area = paddle.unsqueeze(area, -1)
    output_true = target * mask * area * (vel_inlet) ** 2.0
    output_pred = output * mask * area * (vel_inlet) ** 2.0

    pres_true = output_true[:, :, 0] * normals[:, :, 0]
    pres_pred = output_pred[:, :, 0] * normals[:, :, 0]

    wx_true = output_true[:, :, 1]
    wx_pred = output_pred[:, :, 1]

    masked_pred = paddle.sum(pres_pred + wx_pred, (1)) / (
        paddle.sum(area) * (vel_inlet) ** 2.0
    )
    masked_truth = paddle.sum(pres_true + wx_true, (1)) / (
        paddle.sum(area) * (vel_inlet) ** 2.0
    )

    loss = (masked_pred - masked_truth) ** 2.0
    loss = paddle.mean(loss)
    return loss


def validation_step(
    dataloader,
    model,
    device,
    use_sdf_basis=False,
    use_surface_normals=False,
    integral_scaling_factor=1.0,
    loss_fn_type="mse",
):
    running_vloss = 0.0
    with paddle.no_grad():
        for i_batch, sampled_batched in enumerate(dataloader):
            prediction_vol, prediction_surf = model(sampled_batched)

            if prediction_vol is not None:
                target_vol = sampled_batched["volume_fields"]
                if loss_fn_type == "rmse":
                    loss_norm_vol = relative_loss_fn(
                        prediction_vol, target_vol, padded_value=-10
                    )
                else:
                    loss_norm_vol = mse_loss_fn(
                        prediction_vol, target_vol, padded_value=-10
                    )

            if prediction_surf is not None:
                target_surf = sampled_batched["surface_fields"]
                surface_normals = sampled_batched["surface_normals"]
                surface_areas = sampled_batched["surface_areas"]
                if loss_fn_type == "rmse":
                    loss_norm_surf = relative_loss_fn_surface(
                        prediction_surf, target_surf, surface_normals, padded_value=-10
                    )
                    loss_norm_surf_area = relative_loss_fn_area(
                        prediction_surf,
                        target_surf,
                        surface_normals,
                        surface_areas,
                        padded_value=-10,
                    )
                else:
                    loss_norm_surf = mse_loss_fn_surface(
                        prediction_surf, target_surf, surface_normals, padded_value=-10
                    )
                    loss_norm_surf_area = mse_loss_fn_area(
                        prediction_surf,
                        target_surf,
                        surface_normals,
                        surface_areas,
                        padded_value=-10,
                    )
                loss_integral = (
                    integral_loss_fn_new(
                        prediction_surf,
                        target_surf,
                        surface_areas,
                        surface_normals,
                        padded_value=-10,
                    )
                ) * integral_scaling_factor

            if prediction_surf is not None and prediction_vol is not None:
                vloss = (
                    loss_norm_vol
                    + 0.5 * loss_norm_surf
                    + loss_integral
                    + 0.5 * loss_norm_surf_area
                )
            elif prediction_vol is not None:
                vloss = loss_norm_vol
            elif prediction_surf is not None:
                vloss = 0.5 * loss_norm_surf + loss_integral + 0.5 * loss_norm_surf_area

            running_vloss += vloss

    avg_vloss = running_vloss / (i_batch + 1)

    return avg_vloss


def train_epoch(
    dataloader,
    model,
    optimizer,
    scaler,
    epoch_index,
    device,
    integral_scaling_factor,
    loss_fn_type,
):

    running_loss = 0.0
    last_loss = 0.0
    loss_interval = 1

    for i_batch, sampled_batched in enumerate(dataloader):
        with auto_cast(enable=False):
            prediction_vol, prediction_surf = model(sampled_batched)

            if prediction_vol is not None:
                target_vol = sampled_batched["volume_fields"]
                if loss_fn_type == "rmse":
                    loss_norm_vol = relative_loss_fn(
                        prediction_vol, target_vol, padded_value=-10
                    )
                else:
                    loss_norm_vol = mse_loss_fn(
                        prediction_vol, target_vol, padded_value=-10
                    )

            if prediction_surf is not None:

                target_surf = sampled_batched["surface_fields"]
                surface_areas = sampled_batched["surface_areas"]
                surface_normals = sampled_batched["surface_normals"]
                if loss_fn_type == "rmse":
                    loss_norm_surf = relative_loss_fn_surface(
                        prediction_surf, target_surf, surface_normals, padded_value=-10
                    )
                    loss_norm_surf_area = relative_loss_fn_area(
                        prediction_surf,
                        target_surf,
                        surface_normals,
                        surface_areas,
                        padded_value=-10,
                    )
                else:
                    loss_norm_surf = mse_loss_fn_surface(
                        prediction_surf, target_surf, surface_normals, padded_value=-10
                    )
                    loss_norm_surf_area = mse_loss_fn_area(
                        prediction_surf,
                        target_surf,
                        surface_normals,
                        surface_areas,
                        padded_value=-10,
                    )
                loss_integral = (
                    integral_loss_fn_new(
                        prediction_surf,
                        target_surf,
                        surface_areas,
                        surface_normals,
                        padded_value=-10,
                    )
                ) * integral_scaling_factor

            if prediction_vol is not None and prediction_surf is not None:
                loss_norm = (
                    loss_norm_vol
                    + 0.5 * loss_norm_surf
                    + loss_integral
                    + 0.5 * loss_norm_surf_area
                )
            elif prediction_vol is not None:
                loss_norm = loss_norm_vol
            elif prediction_surf is not None:
                loss_norm = (
                    0.5 * loss_norm_surf + loss_integral + 0.5 * loss_norm_surf_area
                )

        loss = loss_norm
        loss = loss / loss_interval
        scaler.scale(loss).backward()

        if ((i_batch + 1) % loss_interval == 0) or (i_batch + 1 == len(dataloader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.clear_gradients()
        # Gather data and report
        running_loss += loss.item()

        if prediction_vol is not None and prediction_surf is not None:
            print(
                f"Device {device}, batch processed: {i_batch + 1}, loss volume: {loss_norm_vol:.5f} \
            , loss surface: {loss_norm_surf:.5f}, loss integral: {loss_integral:.5f}, loss surface area: {loss_norm_surf_area:.5f}"
            )
        elif prediction_vol is not None:
            print(
                f"Device {device}, batch processed: {i_batch + 1}, loss volume: {loss_norm_vol:.5f}"
            )
        elif prediction_surf is not None:
            print(
                f"Device {device}, batch processed: {i_batch + 1} \
            , loss surface: {loss_norm_surf:.5f}, loss integral: {loss_integral:.5f}, loss surface area: {loss_norm_surf_area:.5f}"
            )

    last_loss = running_loss / (i_batch + 1)  # loss per batch
    print(f" Device {device},  batch: {i_batch + 1}, loss norm: {loss:.5f}")
    tb_x = epoch_index * len(dataloader) + i_batch + 1
    print(f"Loss/train: {last_loss}/{tb_x}")

    return last_loss


def compute_scaling_factors(cfg: DictConfig):

    model_type = cfg.model.model_type

    if model_type == "volume" or model_type == "combined":
        vol_save_path = os.path.join(
            "outputs", cfg.project.name, "volume_scaling_factors.npy"
        )
        if not os.path.exists(vol_save_path):
            input_path = cfg.data.input_dir

            volume_variable_names = list(cfg.variables.volume.solution.keys())

            fm_dict = DoMINODataPipe(
                input_path,
                phase="train",
                grid_resolution=cfg.model.interp_res,
                volume_variables=volume_variable_names,
                surface_variables=None,
                normalize_coordinates=True,
                sampling=False,
                sample_in_bbox=True,
                volume_points_sample=cfg.model.volume_points_sample,
                geom_points_sample=cfg.model.geom_points_sample,
                positional_encoding=cfg.model.positional_encoding,
                model_type=cfg.model.model_type,
                bounding_box_dims=cfg.data.bounding_box,
                bounding_box_dims_surf=cfg.data.bounding_box_surface,
                compute_scaling_factors=True,
            )

            # Calculate mean
            if cfg.model.normalization == "mean_std_scaling":
                for j in range(len(fm_dict)):
                    d_dict = fm_dict[j]
                    vol_fields = d_dict["volume_fields"]

                    if vol_fields is not None:
                        if j == 0:
                            vol_fields_sum = np.mean(vol_fields, 0)
                        else:
                            vol_fields_sum += np.mean(vol_fields, 0)
                    else:
                        vol_fields_sum = 0.0

                vol_fields_mean = vol_fields_sum / len(fm_dict)

                for j in range(len(fm_dict)):
                    d_dict = fm_dict[j]
                    vol_fields = d_dict["volume_fields"]

                    if vol_fields is not None:
                        if j == 0:
                            vol_fields_sum_square = np.mean(
                                (vol_fields - vol_fields_mean) ** 2.0, 0
                            )
                        else:
                            vol_fields_sum_square += np.mean(
                                (vol_fields - vol_fields_mean) ** 2.0, 0
                            )
                    else:
                        vol_fields_sum_square = 0.0

                vol_fields_std = np.sqrt(vol_fields_sum_square / len(fm_dict))

                vol_scaling_factors = [vol_fields_mean, vol_fields_std]

            if cfg.model.normalization == "min_max_scaling":
                for j in range(len(fm_dict)):
                    d_dict = fm_dict[j]
                    vol_fields = d_dict["volume_fields"]

                    if vol_fields is not None:
                        vol_mean = np.mean(vol_fields, 0)
                        vol_std = np.std(vol_fields, 0)
                        vol_idx = mean_std_sampling(
                            vol_fields, vol_mean, vol_std, tolerance=12.0
                        )
                        vol_fields_sampled = np.delete(vol_fields, vol_idx, axis=0)
                        if j == 0:
                            vol_fields_max = np.amax(vol_fields_sampled, 0)
                            vol_fields_min = np.amin(vol_fields_sampled, 0)
                        else:
                            vol_fields_max1 = np.amax(vol_fields_sampled, 0)
                            vol_fields_min1 = np.amin(vol_fields_sampled, 0)

                            for k in range(vol_fields.shape[-1]):
                                if vol_fields_max1[k] > vol_fields_max[k]:
                                    vol_fields_max[k] = vol_fields_max1[k]

                                if vol_fields_min1[k] < vol_fields_min[k]:
                                    vol_fields_min[k] = vol_fields_min1[k]
                    else:
                        vol_fields_max = 0.0
                        vol_fields_min = 0.0

                    if j > 20:
                        break
                vol_scaling_factors = [vol_fields_max, vol_fields_min]
            np.save(vol_save_path, vol_scaling_factors)

    if model_type == "surface" or model_type == "combined":
        surf_save_path = os.path.join(
            "outputs", cfg.project.name, "surface_scaling_factors.npy"
        )

        if not os.path.exists(surf_save_path):
            input_path = cfg.data.input_dir

            volume_variable_names = list(cfg.variables.volume.solution.keys())
            surface_variable_names = list(cfg.variables.surface.solution.keys())

            fm_dict = DoMINODataPipe(
                input_path,
                phase="train",
                grid_resolution=cfg.model.interp_res,
                volume_variables=None,
                surface_variables=surface_variable_names,
                normalize_coordinates=True,
                sampling=False,
                sample_in_bbox=True,
                volume_points_sample=cfg.model.volume_points_sample,
                geom_points_sample=cfg.model.geom_points_sample,
                positional_encoding=cfg.model.positional_encoding,
                model_type=cfg.model.model_type,
                bounding_box_dims=cfg.data.bounding_box,
                bounding_box_dims_surf=cfg.data.bounding_box_surface,
                compute_scaling_factors=True,
            )

            # Calculate mean
            if cfg.model.normalization == "mean_std_scaling":
                for j in range(len(fm_dict)):
                    d_dict = fm_dict[j]
                    surf_fields = d_dict["surface_fields"]

                    if surf_fields is not None:
                        if j == 0:
                            surf_fields_sum = np.mean(surf_fields, 0)
                        else:
                            surf_fields_sum += np.mean(surf_fields, 0)
                    else:
                        surf_fields_sum = 0.0

                surf_fields_mean = surf_fields_sum / len(fm_dict)

                for j in range(len(fm_dict)):
                    d_dict = fm_dict[j]
                    surf_fields = d_dict["surface_fields"]

                    if surf_fields is not None:
                        if j == 0:
                            surf_fields_sum_square = np.mean(
                                (surf_fields - surf_fields_mean) ** 2.0, 0
                            )
                        else:
                            surf_fields_sum_square += np.mean(
                                (surf_fields - surf_fields_mean) ** 2.0, 0
                            )
                    else:
                        surf_fields_sum_square = 0.0

                surf_fields_std = np.sqrt(surf_fields_sum_square / len(fm_dict))

                surf_scaling_factors = [surf_fields_mean, surf_fields_std]

            if cfg.model.normalization == "min_max_scaling":
                for j in range(len(fm_dict)):
                    d_dict = fm_dict[j]
                    surf_fields = d_dict["surface_fields"]

                    if surf_fields is not None:
                        surf_mean = np.mean(surf_fields, 0)
                        surf_std = np.std(surf_fields, 0)
                        surf_idx = mean_std_sampling(
                            surf_fields, surf_mean, surf_std, tolerance=12.0
                        )
                        surf_fields_sampled = np.delete(surf_fields, surf_idx, axis=0)
                        if j == 0:
                            surf_fields_max = np.amax(surf_fields_sampled, 0)
                            surf_fields_min = np.amin(surf_fields_sampled, 0)
                        else:
                            surf_fields_max1 = np.amax(surf_fields_sampled, 0)
                            surf_fields_min1 = np.amin(surf_fields_sampled, 0)

                            for k in range(surf_fields.shape[-1]):
                                if surf_fields_max1[k] > surf_fields_max[k]:
                                    surf_fields_max[k] = surf_fields_max1[k]

                                if surf_fields_min1[k] < surf_fields_min[k]:
                                    surf_fields_min[k] = surf_fields_min1[k]
                    else:
                        surf_fields_max = 0.0
                        surf_fields_min = 0.0

                    if j > 20:
                        break

                surf_scaling_factors = [surf_fields_max, surf_fields_min]
            np.save(surf_save_path, surf_scaling_factors)


def train(cfg: DictConfig) -> None:
    compute_scaling_factors(cfg)
    input_path = cfg.data.input_dir
    input_path_val = cfg.data.input_dir_val
    model_type = cfg.model.model_type

    dist.init_parallel_env()

    print(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

    num_vol_vars = 0
    volume_variable_names = []
    if model_type == "volume" or model_type == "combined":
        volume_variable_names = list(cfg.variables.volume.solution.keys())
        for j in volume_variable_names:
            if cfg.variables.volume.solution[j] == "vector":
                num_vol_vars += 3
            else:
                num_vol_vars += 1
    else:
        num_vol_vars = None

    num_surf_vars = 0
    surface_variable_names = []
    if model_type == "surface" or model_type == "combined":
        surface_variable_names = list(cfg.variables.surface.solution.keys())
        num_surf_vars = 0
        for j in surface_variable_names:
            if cfg.variables.surface.solution[j] == "vector":
                num_surf_vars += 3
            else:
                num_surf_vars += 1
    else:
        num_surf_vars = None

    vol_save_path = os.path.join(
        "outputs", cfg.project.name, "volume_scaling_factors.npy"
    )
    surf_save_path = os.path.join(
        "outputs", cfg.project.name, "surface_scaling_factors.npy"
    )
    if os.path.exists(vol_save_path) and os.path.exists(surf_save_path):
        vol_factors = np.load(vol_save_path)
        surf_factors = np.load(surf_save_path)
    else:
        vol_factors = None
        surf_factors = None

    train_dataset = DoMINODataPipe(
        input_path,
        phase="train",
        grid_resolution=cfg.model.interp_res,
        volume_variables=volume_variable_names,
        surface_variables=surface_variable_names,
        normalize_coordinates=True,
        sampling=True,
        sample_in_bbox=True,
        volume_points_sample=cfg.model.volume_points_sample,
        surface_points_sample=cfg.model.surface_points_sample,
        geom_points_sample=cfg.model.geom_points_sample,
        positional_encoding=cfg.model.positional_encoding,
        volume_factors=vol_factors,
        surface_factors=surf_factors,
        scaling_type=cfg.model.normalization,
        model_type=cfg.model.model_type,
        bounding_box_dims=cfg.data.bounding_box,
        bounding_box_dims_surf=cfg.data.bounding_box_surface,
        num_surface_neighbors=cfg.model.num_surface_neighbors,
    )

    val_dataset = DoMINODataPipe(
        input_path_val,
        phase="val",
        grid_resolution=cfg.model.interp_res,
        volume_variables=volume_variable_names,
        surface_variables=surface_variable_names,
        normalize_coordinates=True,
        sampling=True,
        sample_in_bbox=True,
        volume_points_sample=cfg.model.volume_points_sample,
        surface_points_sample=cfg.model.surface_points_sample,
        geom_points_sample=cfg.model.geom_points_sample,
        positional_encoding=cfg.model.positional_encoding,
        volume_factors=vol_factors,
        surface_factors=surf_factors,
        scaling_type=cfg.model.normalization,
        model_type=cfg.model.model_type,
        bounding_box_dims=cfg.data.bounding_box,
        bounding_box_dims_surf=cfg.data.bounding_box_surface,
        num_surface_neighbors=cfg.model.num_surface_neighbors,
    )
    print(f">>>>>> paddle.distributed.get_rank(): {paddle.distributed.get_rank()}")
    print(
        f">>>>>> paddle.distributed.get_world_size(): {paddle.distributed.get_world_size()}"
    )
    train_sampler = DistributedBatchSampler(
        train_dataset,
        batch_size=1,
        num_replicas=paddle.distributed.get_world_size(),
        rank=paddle.distributed.get_rank(),
        **cfg.train.sampler,
    )

    val_sampler = DistributedBatchSampler(
        val_dataset,
        batch_size=1,
        num_replicas=paddle.distributed.get_world_size(),
        rank=paddle.distributed.get_rank(),
        **cfg.val.sampler,
    )

    train_dataloader = DataLoader(train_dataset, **cfg.train.dataloader)
    val_dataloader = DataLoader(val_dataset, **cfg.val.dataloader)

    model = DoMINO(
        input_features=3,
        output_features_vol=num_vol_vars,
        output_features_surf=num_surf_vars,
        model_parameters=cfg.model,
    )

    if paddle.distributed.get_world_size() > 1:
        model = DataParallel(
            model,
        )

    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=0.001
    )
    scheduler = paddle.optimizer.lr.MultiStepDecay(
        learning_rate=optimizer.get_lr(),
        milestones=[50, 100, 150, 200, 250, 300, 350, 400],
        gamma=0.5,
    )
    optimizer.set_lr_scheduler(scheduler)

    # Initialize the scaler for mixed precision
    scaler = GradScaler()

    epoch_number = 0

    model_save_path = os.path.join(cfg.output, "models")
    param_save_path = os.path.join(cfg.output, "param")
    best_model_path = os.path.join(model_save_path, "best_model")
    if paddle.distributed.get_rank() == 0:
        create_directory(model_save_path)
        create_directory(param_save_path)
        create_directory(best_model_path)

    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.barrier()

    init_epoch = load_checkpoint(
        to_absolute_path(cfg.resume_dir),
        models=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )

    if init_epoch != 0:
        init_epoch += 1  # Start with the next epoch
    epoch_number = init_epoch

    # retrive the smallest validation loss if available
    numbers = []
    for filename in os.listdir(best_model_path):
        match = re.search(r"\d+\.\d*[1-9]\d*", filename)
        if match:
            number = float(match.group(0))
            numbers.append(number)

    best_vloss = min(numbers) if numbers else 1_000_000.0

    initial_integral_factor_orig = cfg.model.integral_loss_scaling_factor

    for epoch in range(init_epoch, cfg.train.epochs):
        start_time = time.time()
        print(f"Device {paddle.distributed.get_rank()}, epoch {epoch_number}:")

        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        initial_integral_factor = initial_integral_factor_orig

        model.train()
        avg_loss = train_epoch(
            dataloader=train_dataloader,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch_index=epoch,
            device=paddle.distributed.get_rank(),
            integral_scaling_factor=initial_integral_factor,
            loss_fn_type=cfg.model.loss_function,
        )

        model.eval()
        avg_vloss = validation_step(
            dataloader=val_dataloader,
            model=model,
            device=paddle.distributed.get_rank(),
            use_sdf_basis=cfg.model.use_sdf_in_basis_func,
            use_surface_normals=cfg.model.use_surface_normals,
            integral_scaling_factor=initial_integral_factor,
            loss_fn_type=cfg.model.loss_function,
        )

        scheduler.step()
        print(
            f"Device {paddle.distributed.get_rank()} "
            f"LOSS train {avg_loss:.5f} "
            f"valid {avg_vloss:.5f} "
            f"Current lr {scheduler.get_lr()}"
            f"Integral factor {initial_integral_factor}"
        )

        # Track best performance, and save the model's state
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.barrier()

        if avg_vloss < best_vloss:  # This only considers GPU: 0, is that okay?
            best_vloss = avg_vloss
        print(
            f"Device { paddle.distributed.get_rank()}, Best val loss {best_vloss}, Time taken {time.time() - start_time}"
        )

        if (
            paddle.distributed.get_rank() == 0
            and (epoch + 1) % cfg.train.checkpoint_interval == 0.0
        ):
            save_checkpoint(
                to_absolute_path(model_save_path),
                models=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
            )

        epoch_number += 1

        if scheduler.get_lr() == 1e-6:
            print("Training ended")
            exit()


def loss_fn(output, target):
    masked_loss = paddle.mean(((output - target) ** 2.0), (0, 1, 2))
    loss = paddle.mean(masked_loss)
    return loss


def test_step(data_dict, model, device, cfg, vol_factors, surf_factors):
    running_tloss_vol = 0.0
    running_tloss_surf = 0.0

    if cfg.model.model_type == "volume" or cfg.model.model_type == "combined":
        output_features_vol = True
    else:
        output_features_vol = None

    if cfg.model.model_type == "surface" or cfg.model.model_type == "combined":
        output_features_surf = True
    else:
        output_features_surf = None

    with paddle.no_grad():
        point_batch_size = 256000

        # Non-dimensionalization factors
        air_density = data_dict["air_density"]
        stream_velocity = data_dict["stream_velocity"]
        length_scale = data_dict["length_scale"]

        # STL nodes
        geo_centers = data_dict["geometry_coordinates"]

        # Bounding box grid
        s_grid = data_dict["surf_grid"]
        sdf_surf_grid = data_dict["sdf_surf_grid"]
        # Scaling factors
        surf_max = data_dict["surface_min_max"][:, 1]
        surf_min = data_dict["surface_min_max"][:, 0]

        if output_features_vol is not None:
            # Represent geometry on computational grid
            # Computational domain grid
            p_grid = data_dict["grid"]
            sdf_grid = data_dict["sdf_grid"]
            # Scaling factors
            vol_max = data_dict["volume_min_max"][:, 1]
            vol_min = data_dict["volume_min_max"][:, 0]

            # Normalize based on computational domain
            geo_centers_vol = 2.0 * (geo_centers - vol_min) / (vol_max - vol_min) - 1
            encoding_g_vol = model.module.geo_rep(geo_centers_vol, p_grid, sdf_grid)

            # Normalize based on BBox around surface (car)
            geo_centers_surf = (
                2.0 * (geo_centers - surf_min) / (surf_max - surf_min) - 1
            )
            encoding_g_surf = model.module.geo_rep(
                geo_centers_surf, s_grid, sdf_surf_grid
            )

        if output_features_surf is not None:
            # Represent geometry on bounding box
            geo_centers_surf = (
                2.0 * (geo_centers - surf_min) / (surf_max - surf_min) - 1
            )
            encoding_g_surf = model.module.geo_rep(
                geo_centers_surf, s_grid, sdf_surf_grid
            )

        geo_encoding = 0.5 * encoding_g_surf
        # Average the encodings
        if output_features_vol is not None:
            geo_encoding += 0.5 * encoding_g_vol

        if output_features_vol is not None:
            # First calculate volume predictions if required
            volume_mesh_centers = data_dict["volume_mesh_centers"]
            target_vol = data_dict["volume_fields"]
            # SDF on volume mesh nodes
            sdf_nodes = data_dict["sdf_nodes"]
            # Positional encoding based on closest point on surface to a volume node
            pos_volume_closest = data_dict["pos_volume_closest"]
            # Positional encoding based on center of mass of geometry to volume node
            pos_volume_center_of_mass = data_dict["pos_volume_center_of_mass"]
            p_grid = data_dict["grid"]

            prediction_vol = np.zeros_like(target_vol.cpu().numpy())
            num_points = volume_mesh_centers.shape[1]
            subdomain_points = int(np.floor(num_points / point_batch_size))

            for p in range(subdomain_points + 1):
                start_idx = p * point_batch_size
                end_idx = (p + 1) * point_batch_size
                with paddle.no_grad():
                    target_batch = target_vol[:, start_idx:end_idx]
                    volume_mesh_centers_batch = volume_mesh_centers[
                        :, start_idx:end_idx
                    ]
                    sdf_nodes_batch = sdf_nodes[:, start_idx:end_idx]
                    pos_volume_closest_batch = pos_volume_closest[:, start_idx:end_idx]
                    pos_normals_com_batch = pos_volume_center_of_mass[
                        :, start_idx:end_idx
                    ]
                    geo_encoding_local = model.module.geo_encoding_local(
                        geo_encoding, volume_mesh_centers_batch, p_grid
                    )
                    if cfg.model.use_sdf_in_basis_func:
                        pos_encoding = paddle.concat(
                            (
                                sdf_nodes_batch,
                                pos_volume_closest_batch,
                                pos_normals_com_batch,
                            ),
                            axis=-1,
                        )
                    else:
                        pos_encoding = pos_normals_com_batch
                    pos_encoding = model.module.position_encoder(
                        pos_encoding, eval_mode="volume"
                    )
                    tpredictions_batch = model.module.calculate_solution(
                        volume_mesh_centers_batch,
                        geo_encoding_local,
                        pos_encoding,
                        stream_velocity,
                        air_density,
                        num_sample_points=20,
                        eval_mode="volume",
                    )
                    running_tloss_vol += loss_fn(tpredictions_batch, target_batch)
                    prediction_vol[
                        :, start_idx:end_idx
                    ] = tpredictions_batch.cpu().numpy()

            prediction_vol = unnormalize(prediction_vol, vol_factors[0], vol_factors[1])

            prediction_vol[:, :, :3] = (
                prediction_vol[:, :, :3] * stream_velocity[0, 0].cpu().numpy()
            )
            prediction_vol[:, :, 3] = (
                prediction_vol[:, :, 3]
                * stream_velocity[0, 0].cpu().numpy() ** 2.0
                * air_density[0, 0].cpu().numpy()
            )
            prediction_vol[:, :, 4] = (
                prediction_vol[:, :, 4]
                * stream_velocity[0, 0].cpu().numpy()
                * length_scale[0].cpu().numpy()
            )
        else:
            prediction_vol = None

        if output_features_surf is not None:
            # Next calculate surface predictions
            # Sampled points on surface
            surface_mesh_centers = data_dict["surface_mesh_centers"]
            surface_normals = data_dict["surface_normals"]
            surface_areas = data_dict["surface_areas"]

            # Neighbors of sampled points on surface
            surface_mesh_neighbors = data_dict["surface_mesh_neighbors"]
            surface_neighbors_normals = data_dict["surface_neighbors_normals"]
            surface_neighbors_areas = data_dict["surface_neighbors_areas"]
            surface_areas = paddle.unsqueeze(surface_areas, -1)
            surface_neighbors_areas = paddle.unsqueeze(surface_neighbors_areas, -1)
            pos_surface_center_of_mass = data_dict["pos_surface_center_of_mass"]
            num_points = surface_mesh_centers.shape[1]
            subdomain_points = int(np.floor(num_points / point_batch_size))

            target_surf = data_dict["surface_fields"]
            prediction_surf = np.zeros_like(target_surf.cpu().numpy())

            surface_areas = paddle.unsqueeze(surface_areas, -1)
            surface_neighbors_areas = paddle.unsqueeze(surface_neighbors_areas, -1)

            for p in range(subdomain_points + 1):
                start_idx = p * point_batch_size
                end_idx = (p + 1) * point_batch_size
                with paddle.no_grad():
                    target_batch = target_surf[:, start_idx:end_idx]
                    surface_mesh_centers_batch = surface_mesh_centers[
                        :, start_idx:end_idx
                    ]
                    surface_mesh_neighbors_batch = surface_mesh_neighbors[
                        :, start_idx:end_idx
                    ]
                    surface_normals_batch = surface_normals[:, start_idx:end_idx]
                    surface_neighbors_normals_batch = surface_neighbors_normals[
                        :, start_idx:end_idx
                    ]
                    surface_areas_batch = surface_areas[:, start_idx:end_idx]
                    surface_neighbors_areas_batch = surface_neighbors_areas[
                        :, start_idx:end_idx
                    ]
                    pos_surface_center_of_mass_batch = pos_surface_center_of_mass[
                        :, start_idx:end_idx
                    ]
                    geo_encoding_local = model.module.geo_encoding_local_surface(
                        0.5 * encoding_g_surf, surface_mesh_centers_batch, s_grid
                    )
                    pos_encoding = pos_surface_center_of_mass_batch
                    pos_encoding = model.module.position_encoder(
                        pos_encoding, eval_mode="surface"
                    )

                    if cfg.model.surface_neighbors:
                        tpredictions_batch = (
                            model.module.calculate_solution_with_neighbors(
                                surface_mesh_centers_batch,
                                geo_encoding_local,
                                pos_encoding,
                                surface_mesh_neighbors_batch,
                                surface_normals_batch,
                                surface_neighbors_normals_batch,
                                surface_areas_batch,
                                surface_neighbors_areas_batch,
                                stream_velocity,
                                air_density,
                            )
                        )
                    else:
                        tpredictions_batch = model.module.calculate_solution(
                            surface_mesh_centers_batch,
                            geo_encoding_local,
                            pos_encoding,
                            stream_velocity,
                            air_density,
                            num_sample_points=1,
                            eval_mode="surface",
                        )
                    running_tloss_surf += loss_fn(tpredictions_batch, target_batch)
                    prediction_surf[
                        :, start_idx:end_idx
                    ] = tpredictions_batch.cpu().numpy()

            prediction_surf = (
                unnormalize(prediction_surf, surf_factors[0], surf_factors[1])
                * stream_velocity[0, 0].cpu().numpy() ** 2.0
                * air_density[0, 0].cpu().numpy()
            )

        else:
            prediction_surf = None

    return prediction_vol, prediction_surf


def test(cfg: DictConfig):
    print(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

    input_path = cfg.eval.test_path

    model_type = cfg.model.model_type

    dist.init_parallel_env()

    if model_type == "volume" or model_type == "combined":
        volume_variable_names = list(cfg.variables.volume.solution.keys())
        num_vol_vars = 0
        for j in volume_variable_names:
            if cfg.variables.volume.solution[j] == "vector":
                num_vol_vars += 3
            else:
                num_vol_vars += 1
    else:
        num_vol_vars = None

    if model_type == "surface" or model_type == "combined":
        surface_variable_names = list(cfg.variables.surface.solution.keys())
        num_surf_vars = 0
        for j in surface_variable_names:
            if cfg.variables.surface.solution[j] == "vector":
                num_surf_vars += 3
            else:
                num_surf_vars += 1
    else:
        num_surf_vars = None

    vol_save_path = os.path.join(
        "outputs", cfg.project.name, "volume_scaling_factors.npy"
    )
    surf_save_path = os.path.join(
        "outputs", cfg.project.name, "surface_scaling_factors.npy"
    )
    if os.path.exists(vol_save_path) and os.path.exists(surf_save_path):
        vol_factors = np.load(vol_save_path)
        surf_factors = np.load(surf_save_path)
    else:
        vol_factors = None
        surf_factors = None

    model = DoMINO(
        input_features=3,
        output_features_vol=num_vol_vars,
        output_features_surf=num_surf_vars,
        model_parameters=cfg.model,
    )

    checkpoint = paddle.load(
        to_absolute_path(os.path.join(cfg.resume_dir, cfg.eval.checkpoint_name)),
    )

    model.set_state_dict(checkpoint)

    print("Model loaded")

    if paddle.distributed.get_world_size() > 1:
        model = DataParallel(
            model,
        )

    dirnames_per_gpu = get_filenames(input_path)

    pred_save_path = cfg.eval.save_path
    create_directory(pred_save_path)

    for count, dirname in enumerate(dirnames_per_gpu):
        # print(f"Processing file {dirname}")
        filepath = os.path.join(input_path, dirname)
        tag = int(re.findall(r"(\w+?)(\d+)", dirname)[0][1])
        stl_path = os.path.join(filepath, f"drivaer_{tag}.stl")
        vtp_path = os.path.join(filepath, f"boundary_{tag}.vtp")
        vtu_path = os.path.join(filepath, f"volume_{tag}.vtu")

        vtp_pred_save_path = os.path.join(
            pred_save_path, f"boundary_{tag}_predicted.vtp"
        )
        vtu_pred_save_path = os.path.join(pred_save_path, f"volume_{tag}_predicted.vtu")

        # Read STL
        reader = pv.get_reader(stl_path)
        mesh_stl = reader.read()
        stl_vertices = mesh_stl.points
        stl_faces = np.array(mesh_stl.faces).reshape((-1, 4))[
            :, 1:
        ]  # Assuming triangular elements
        mesh_indices_flattened = stl_faces.flatten()
        length_scale = np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0))
        stl_sizes = mesh_stl.compute_cell_sizes(length=False, area=True, volume=False)
        stl_sizes = np.array(stl_sizes.cell_data["Area"], dtype=np.float32)
        stl_centers = np.array(mesh_stl.cell_centers().points, dtype=np.float32)

        # Center of mass calculation
        center_of_mass = calculate_center_of_mass(stl_centers, stl_sizes)

        if cfg.data.bounding_box_surface is None:
            s_max = np.amax(stl_vertices, 0)
            s_min = np.amin(stl_vertices, 0)
        else:
            bounding_box_dims_surf = []
            bounding_box_dims_surf.append(np.asarray(cfg.data.bounding_box_surface.max))
            bounding_box_dims_surf.append(np.asarray(cfg.data.bounding_box_surface.min))
            s_max = np.float32(bounding_box_dims_surf[0])
            s_min = np.float32(bounding_box_dims_surf[1])

        nx, ny, nz = cfg.model.interp_res

        surf_grid = create_grid(s_max, s_min, [nx, ny, nz])
        surf_grid_reshaped = surf_grid.reshape(nx * ny * nz, 3)

        # SDF calculation on the grid using WARP
        sdf_surf_grid = (
            signed_distance_field(
                stl_vertices,
                mesh_indices_flattened,
                surf_grid_reshaped,
                use_sign_winding_number=True,
            )
            .numpy()
            .reshape(nx, ny, nz)
        )
        surf_grid = np.float32(surf_grid)
        sdf_surf_grid = np.float32(sdf_surf_grid)
        surf_grid_max_min = np.float32(np.asarray([s_min, s_max]))

        # Read VTP
        if model_type == "surface" or model_type == "combined":
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(vtp_path)
            reader.Update()
            polydata_surf = reader.GetOutput()

            celldata_all = get_node_to_elem(polydata_surf)

            celldata = celldata_all.GetCellData()
            surface_fields = get_fields(celldata, surface_variable_names)
            surface_fields = np.concatenate(surface_fields, axis=-1)

            mesh = pv.PolyData(polydata_surf)
            surface_coordinates = np.array(mesh.cell_centers().points, dtype=np.float32)

            interp_func = KDTree(surface_coordinates)
            dd, ii = interp_func.query(
                surface_coordinates, k=cfg.model.num_surface_neighbors
            )

            surface_neighbors = surface_coordinates[ii]
            surface_neighbors = surface_neighbors[:, 1:]

            surface_normals = np.array(mesh.cell_normals, dtype=np.float32)
            surface_sizes = mesh.compute_cell_sizes(
                length=False, area=True, volume=False
            )
            surface_sizes = np.array(surface_sizes.cell_data["Area"], dtype=np.float32)

            # Normalize cell normals
            surface_normals = (
                surface_normals / np.linalg.norm(surface_normals, axis=1)[:, np.newaxis]
            )
            surface_neighbors_normals = surface_normals[ii]
            surface_neighbors_normals = surface_neighbors_normals[:, 1:]
            surface_neighbors_sizes = surface_sizes[ii]
            surface_neighbors_sizes = surface_neighbors_sizes[:, 1:]

            dx, dy, dz = (
                (s_max[0] - s_min[0]) / nx,
                (s_max[1] - s_min[1]) / ny,
                (s_max[2] - s_min[2]) / nz,
            )

            if cfg.model.positional_encoding:
                pos_surface_center_of_mass = cal_normal_positional_encoding(
                    surface_coordinates, center_of_mass, cell_length=[dx, dy, dz]
                )
            else:
                pos_surface_center_of_mass = surface_coordinates - center_of_mass

            surface_coordinates = normalize(surface_coordinates, s_max, s_min)
            surface_neighbors = normalize(surface_neighbors, s_max, s_min)
            surf_grid = normalize(surf_grid, s_max, s_min)

        else:
            surface_coordinates = None
            surface_fields = None
            surface_sizes = None
            surface_normals = None
            surface_neighbors = None
            surface_neighbors_normals = None
            surface_neighbors_sizes = None
            pos_surface_center_of_mass = None

        # Read VTU
        if model_type == "volume" or model_type == "combined":
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(vtu_path)
            reader.Update()
            polydata_vol = reader.GetOutput()
            volume_coordinates, volume_fields = get_volume_data(
                polydata_vol, volume_variable_names
            )
            volume_fields = np.concatenate(volume_fields, axis=-1)
            # print(f"Processed vtu {vtu_path}")

            bounding_box_dims = []
            bounding_box_dims.append(np.asarray(cfg.data.bounding_box.max))
            bounding_box_dims.append(np.asarray(cfg.data.bounding_box.min))

            if bounding_box_dims is None:
                c_max = s_max + (s_max - s_min) / 2
                c_min = s_min - (s_max - s_min) / 2
                c_min[2] = s_min[2]
            else:
                c_max = np.float32(bounding_box_dims[0])
                c_min = np.float32(bounding_box_dims[1])

            dx, dy, dz = (
                (c_max[0] - c_min[0]) / nx,
                (c_max[1] - c_min[1]) / ny,
                (c_max[2] - c_min[2]) / nz,
            )
            # Generate a grid of specified resolution to map the bounding box
            # The grid is used for capturing structured geometry features and SDF representation of geometry
            grid = create_grid(c_max, c_min, [nx, ny, nz])
            grid_reshaped = grid.reshape(nx * ny * nz, 3)

            # SDF calculation on the grid using WARP
            sdf_grid = (
                signed_distance_field(
                    stl_vertices,
                    mesh_indices_flattened,
                    grid_reshaped,
                    use_sign_winding_number=True,
                )
                .numpy()
                .reshape(nx, ny, nz)
            )

            # SDF calculation
            sdf_nodes, sdf_node_closest_point = signed_distance_field(
                stl_vertices,
                mesh_indices_flattened,
                volume_coordinates,
                include_hit_points=True,
                use_sign_winding_number=True,
            )
            sdf_nodes = sdf_nodes.numpy().reshape(-1, 1)
            sdf_node_closest_point = sdf_node_closest_point.numpy()

            if cfg.model.positional_encoding:
                pos_volume_closest = cal_normal_positional_encoding(
                    volume_coordinates, sdf_node_closest_point, cell_length=[dx, dy, dz]
                )
                pos_volume_center_of_mass = cal_normal_positional_encoding(
                    volume_coordinates, center_of_mass, cell_length=[dx, dy, dz]
                )
            else:
                pos_volume_closest = volume_coordinates - sdf_node_closest_point
                pos_volume_center_of_mass = volume_coordinates - center_of_mass

            volume_coordinates = normalize(volume_coordinates, c_max, c_min)
            grid = normalize(grid, c_max, c_min)
            vol_grid_max_min = np.asarray([c_min, c_max])

        else:
            volume_coordinates = None
            volume_fields = None
            pos_volume_closest = None
            pos_volume_center_of_mass = None

        # print(f"Processed sdf and normalized")

        geom_centers = np.float32(stl_vertices)

        if model_type == "combined":
            # Add the parameters to the dictionary
            data_dict = {
                "pos_volume_closest": pos_volume_closest,
                "pos_volume_center_of_mass": pos_volume_center_of_mass,
                "pos_surface_center_of_mass": pos_surface_center_of_mass,
                "geometry_coordinates": geom_centers,
                "grid": grid,
                "surf_grid": surf_grid,
                "sdf_grid": sdf_grid,
                "sdf_surf_grid": sdf_surf_grid,
                "sdf_nodes": sdf_nodes,
                "surface_mesh_centers": surface_coordinates,
                "surface_mesh_neighbors": surface_neighbors,
                "surface_normals": surface_normals,
                "surface_neighbors_normals": surface_neighbors_normals,
                "surface_areas": surface_sizes,
                "surface_neighbors_areas": surface_neighbors_sizes,
                "volume_fields": volume_fields,
                "volume_mesh_centers": volume_coordinates,
                "surface_fields": surface_fields,
                "volume_min_max": vol_grid_max_min,
                "surface_min_max": surf_grid_max_min,
                "length_scale": np.array(length_scale, dtype=np.float32),
                "stream_velocity": np.expand_dims(
                    np.array(STREAM_VELOCITY, dtype=np.float32), axis=-1
                ),
                "air_density": np.expand_dims(
                    np.array(AIR_DENSITY, dtype=np.float32), axis=-1
                ),
            }
        elif model_type == "surface":
            data_dict = {
                "pos_surface_center_of_mass": np.float32(pos_surface_center_of_mass),
                "geometry_coordinates": np.float32(geom_centers),
                "surf_grid": np.float32(surf_grid),
                "sdf_surf_grid": np.float32(sdf_surf_grid),
                "surface_mesh_centers": np.float32(surface_coordinates),
                "surface_mesh_neighbors": np.float32(surface_neighbors),
                "surface_normals": np.float32(surface_normals),
                "surface_neighbors_normals": np.float32(surface_neighbors_normals),
                "surface_areas": np.float32(surface_sizes),
                "surface_neighbors_areas": np.float32(surface_neighbors_sizes),
                "surface_fields": np.float32(surface_fields),
                "surface_min_max": np.float32(surf_grid_max_min),
                "length_scale": np.array(length_scale, dtype=np.float32),
                "stream_velocity": np.expand_dims(
                    np.array(STREAM_VELOCITY, dtype=np.float32), axis=-1
                ),
                "air_density": np.expand_dims(
                    np.array(AIR_DENSITY, dtype=np.float32), axis=-1
                ),
            }
        elif model_type == "volume":
            data_dict = {
                "pos_volume_closest": pos_volume_closest,
                "pos_volume_center_of_mass": pos_volume_center_of_mass,
                "geometry_coordinates": geom_centers,
                "grid": grid,
                "surf_grid": surf_grid,
                "sdf_grid": sdf_grid,
                "sdf_surf_grid": sdf_surf_grid,
                "sdf_nodes": sdf_nodes,
                "volume_fields": volume_fields,
                "volume_mesh_centers": volume_coordinates,
                "volume_min_max": vol_grid_max_min,
                "surface_min_max": surf_grid_max_min,
                "length_scale": np.array(length_scale, dtype=np.float32),
                "stream_velocity": np.expand_dims(
                    np.array(STREAM_VELOCITY, dtype=np.float32), axis=-1
                ),
                "air_density": np.expand_dims(
                    np.array(AIR_DENSITY, dtype=np.float32), axis=-1
                ),
            }

        data_dict = {
            key: paddle.to_tensor(np.expand_dims(np.float32(value), 0))
            for key, value in data_dict.items()
        }

        prediction_vol, prediction_surf = test_step(
            data_dict,
            model,
            paddle.distributed.get_rank(),
            cfg,
            vol_factors,
            surf_factors,
        )

        if prediction_surf is not None:
            surface_sizes = np.expand_dims(surface_sizes, -1)

            force_x_pred = np.sum(
                prediction_surf[0, :, 0] * surface_normals[:, 0] * surface_sizes[:, 0]
                - prediction_surf[0, :, 1] * surface_sizes[:, 0]
            )
            force_x_true = np.sum(
                surface_fields[:, 0] * surface_normals[:, 0] * surface_sizes[:, 0]
                - surface_fields[:, 1] * surface_sizes[:, 0]
            )
            print(dirname, force_x_pred, force_x_true)

        if prediction_vol is not None:
            target_vol = volume_fields
            prediction_vol = prediction_vol[0]
            c_min = vol_grid_max_min[0]
            c_max = vol_grid_max_min[1]
            volume_coordinates = unnormalize(volume_coordinates, c_max, c_min)
            ids_in_bbox = np.where(
                (volume_coordinates[:, 0] < c_min[0])
                | (volume_coordinates[:, 0] > c_max[0])
                | (volume_coordinates[:, 1] < c_min[1])
                | (volume_coordinates[:, 1] > c_max[1])
                | (volume_coordinates[:, 2] < c_min[2])
                | (volume_coordinates[:, 2] > c_max[2])
            )
            target_vol[ids_in_bbox] = 0.0
            prediction_vol[ids_in_bbox] = 0.0
            l2_gt = np.sum(np.square(target_vol), (0))
            l2_error = np.sum(np.square(prediction_vol - target_vol), (0))
            print(
                "L-2 norm:",
                dirname,
                np.sqrt(l2_error),
                np.sqrt(l2_gt),
                np.sqrt(l2_error) / np.sqrt(l2_gt),
            )

        if prediction_surf is not None:
            surfParam_vtk = numpy_support.numpy_to_vtk(prediction_surf[0, :, 0:1])
            surfParam_vtk.SetName(f"{surface_variable_names[0]}Pred")
            celldata_all.GetCellData().AddArray(surfParam_vtk)

            surfParam_vtk = numpy_support.numpy_to_vtk(prediction_surf[0, :, 1:])
            surfParam_vtk.SetName(f"{surface_variable_names[1]}Pred")
            celldata_all.GetCellData().AddArray(surfParam_vtk)

            write_to_vtp(celldata_all, vtp_pred_save_path)

        if prediction_vol is not None:

            volParam_vtk = numpy_support.numpy_to_vtk(prediction_vol[:, 0:3])
            volParam_vtk.SetName(f"{volume_variable_names[0]}Pred")
            polydata_vol.GetPointData().AddArray(volParam_vtk)

            volParam_vtk = numpy_support.numpy_to_vtk(prediction_vol[:, 3:4])
            volParam_vtk.SetName(f"{volume_variable_names[1]}Pred")
            polydata_vol.GetPointData().AddArray(volParam_vtk)

            volParam_vtk = numpy_support.numpy_to_vtk(prediction_vol[:, 4:5])
            volParam_vtk.SetName(f"{volume_variable_names[2]}Pred")
            polydata_vol.GetPointData().AddArray(volParam_vtk)

            write_to_vtu(polydata_vol, vtu_pred_save_path)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        test(cfg)
    elif cfg.mode == "process":
        process(cfg)
    else:
        raise ValueError(
            f"cfg.mode should in ['process', 'train', 'eval'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()
