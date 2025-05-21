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


class CustomDataLoader(paddle.io.Dataset):
    def __init__(self, target_lead_times, cfg):
        super().__init__()

        self.target_lead_times = target_lead_times
        self.cfg = cfg

    def __len__(self):
        # Return the number of time steps in target_lead_times
        return len(self.target_lead_times)

    def __getitem__(self, index):
        # Select a specific time step
        time_step = self.target_lead_times[index]

        # Multiply by 12 to get 'a'
        a = time_step * 12

        # Create a string in the format 'ah'
        ah_str = f"{a}h"

        # Update the config with this new 'ah' string
        self.cfg["target_lead_times"] = ah_str

        # Call the ERA5Data function/class
        # Assuming ERA5Data is a function or class that processes this config
        data = datasets.ERA5Data(config=self.cfg)

        return data


def train(cfg: DictConfig):
    # Initialize the GenCast model with the given configuration.
    model = gencast.GenCast(cfg)
    model.train()

    # set optimizer
    optimizer = paddle.optimizer.AdamW(
        parameters=model.parameters(),
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    # Load the dataset using the given configuration.
    nc_dataset = xarray.open_dataset(cfg.data_path)
    time_total = len(nc_dataset.time.data)
    train_loader = CustomDataLoader(
        target_lead_times=list(range(1, time_total - 1)),
        cfg=cfg,
    )

    best_loss = float("inf")
    for epoch in range(cfg.train.num_epochs):
        epoch_loss = 0
        for dataset in train_loader:
            # Forward pass and compute loss
            loss, diagnostics = model.loss(
                dataset.inputs_template,
                dataset.targets_template,
                dataset.forcings_template,
            )
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            epoch_loss += loss.item()

        # Average loss for the epoch
        epoch_loss /= len(train_loader)
        logging.info(f"Epoch {epoch}: Loss = {epoch_loss:.6f}")
        if epoch % cfg.train.snapshot_freq == 0 or epoch == 1:
            model_save_path = os.path.join(
                cfg.output_dir, f"last_model_epoch_{epoch}.pdparams"
            )
            paddle.save(model.state_dict(), model_save_path)

        # Save model if it has the best loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            model_save_path = os.path.join(cfg.output_dir, "best_model_epoch.pdparams")
            paddle.save(model.state_dict(), model_save_path)
            logging.info(f"Best model saved at epoch {epoch} with loss {best_loss:.6f}")


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
    elif cfg.mode == "train":
        train(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
