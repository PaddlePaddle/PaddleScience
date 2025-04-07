# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

import gencast
import hydra
import numpy as np
import paddle
import xarray
from graphcast import datasets
from graphcast import utils
from graphcast import vis
from omegaconf import DictConfig


def crps(targets, predictions, bias_corrected=True):
    if predictions.sizes.get("sample", 1) < 2:
        raise ValueError("predictions must have dim 'sample' with size at least 2.")
    sum_dims = ["sample", "sample2"]
    preds2 = predictions.rename({"sample": "sample2"})
    num_samps = predictions.sizes["sample"]
    num_samps2 = (num_samps - 1) if bias_corrected else num_samps
    mean_abs_diff = np.abs(predictions - preds2).sum(dim=sum_dims, skipna=False) / (
        num_samps * num_samps2
    )
    mean_abs_err = (
        np.abs(targets - predictions).sum(dim="sample", skipna=False) / num_samps
    )
    return mean_abs_err - 0.5 * mean_abs_diff


def eval(cfg: DictConfig):

    base_seed = cfg.seed
    chunks = []
    for i in range(cfg.num_ensemble_members):

        logging.info("Sample %d/%d", i, cfg.num_ensemble_members)

        seed = i + base_seed
        paddle.seed(seed)

        # Initialize the GenCast model with the given configuration.
        model = gencast.GenCast(cfg)
        # Load the model parameters from the specified path.
        model.load_dict(paddle.load(cfg.param_path))
        # Load the dataset using the given configuration.
        dataset = datasets.ERA5Data(config=cfg)

        # Generate predictions using the model; targets are initialized to NaN
        pred = model(
            dataset.inputs_template,
            dataset.targets_template * np.nan,
            dataset.forcings_template,
        )

        # Denormalize the predictions
        stacked_pred = datasets.dataset_to_stacked(pred)
        stacked_pred = stacked_pred.transpose("lat", "lon", ...)
        lat_dim, lon_dim, batch_dim, feat_dim = stacked_pred.shape
        stacked_pred = stacked_pred.data.reshape(lat_dim * lon_dim, batch_dim, -1)
        stacked_pred_denormalized = dataset.denormalize(stacked_pred)
        outputs_lat_lon_leading = stacked_pred_denormalized.reshape(
            (lat_dim, lon_dim) + stacked_pred_denormalized.shape[1:]
        )
        dims = ("lat", "lon", "batch", "channels")
        xarray_lat_lon_leading = xarray.DataArray(
            data=outputs_lat_lon_leading, dims=dims
        )
        pred_xarray = utils.restore_leading_axes(xarray_lat_lon_leading)
        pred_denormalized = datasets.stacked_to_dataset(
            pred_xarray.variable, dataset.targets_template
        )

        # Add new dimensions and coordinates to each data variable
        sample_coord = xarray.DataArray([i], dims="sample")
        pred_denormalized = pred_denormalized.expand_dims(sample=sample_coord)
        chunks.append(pred_denormalized)

    predictions = xarray.combine_by_coords(chunks)
    # Save the predictions to a NetCDF file
    predictions.to_netcdf(os.path.join(cfg.output_dir, "predictions.nc"))

    # Calculate RMSE for each variable in the predictions
    pred_mean = predictions.mean(dim="sample")
    rmse = np.sqrt(((pred_mean - dataset.targets_template) ** 2).mean())
    logging.info(f"RMSE: {rmse.values}")

    # Visualize and save the result images
    vis.log_images(
        dataset.targets_template,
        pred_mean,
        "2m_temperature",
        level=50,
        file="result.png",
    )


@hydra.main(version_base=None, config_path="./conf", config_name="gencast.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "eval":
        eval(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
