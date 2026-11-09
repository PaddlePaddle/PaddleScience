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

from __future__ import annotations

import os
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

try:
    import xarray as xr
except ModuleNotFoundError:
    pass
import numpy as np
import paddle
from paddle import io


class Normalize:
    """Normalize an array with per-channel mean and standard deviation.

    Given ``mean: (C,)`` and ``std: (C,)``, ``output[c] = (input[c] - mean[c]) / std[c]``
    where the channel dim is dim -3 of a ``(..., C, H, W)`` array. Scalars broadcast
    over every element following numpy rules. ``mean``/``std`` are kept as float32
    numpy arrays so callers can read them back for denormalization.
    """

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype="float32")
        self.std = np.asarray(std, dtype="float32")

    def __call__(self, t):
        mean = self.mean
        std = self.std
        if mean.ndim == 1 and t.ndim >= 3:
            # Per-channel stats broadcast over the channel dim (dim -3 of a
            # (..., C, H, W) array). Plain numpy broadcasting would wrongly
            # align (C,) with the last dim (W,).
            shape = (1,) * (t.ndim - 3) + mean.shape + (1, 1)
            mean = mean.reshape(shape)
            std = std.reshape(shape)
        return (t - mean) / std


def load_x_y(data_path, list_simu, out_var):
    """Load inputs/outputs of the given simulations and stack them per simu.

    Non-historical simulations are concatenated with the historical period
    along time. Returns per-simu input array (T, V, H, W), per-simu output
    array (T, Vout, H, W) and the latitude/longitude axes of the first simu.
    """
    x_all, y_all = {}, {}
    for simu in list_simu:
        input_name = "inputs_" + simu + ".nc"
        output_name = "outputs_" + simu + ".nc"
        if "hist" in simu:
            input_xr = xr.open_dataset(os.path.join(data_path, input_name))
            output_xr = xr.open_dataset(os.path.join(data_path, output_name)).mean(
                dim="member"
            )
        else:
            input_xr = xr.open_mfdataset(
                [
                    os.path.join(data_path, "inputs_historical.nc"),
                    os.path.join(data_path, input_name),
                ]
            ).compute()
            output_xr = xr.concat(
                [
                    xr.open_dataset(
                        os.path.join(data_path, "outputs_historical.nc")
                    ).mean(dim="member"),
                    xr.open_dataset(os.path.join(data_path, output_name)).mean(
                        dim="member"
                    ),
                ],
                dim="time",
            ).compute()
        output_xr = (
            output_xr.assign(
                {"pr": output_xr.pr * 86400, "pr90": output_xr.pr90 * 86400}
            )
            .rename({"lon": "longitude", "lat": "latitude"})
            .transpose("time", "latitude", "longitude")
            .drop(["quantile"])
        )
        print(input_xr.dims, output_xr.dims, simu)
        x = input_xr.to_array().to_numpy()
        x = x.transpose(1, 0, 2, 3).astype(np.float32)
        x_all[simu] = x
        y = output_xr[out_var].to_array().to_numpy()
        y = y.transpose(1, 0, 2, 3).astype(np.float32)
        y_all[simu] = y
    temp = xr.open_dataset(
        os.path.join(data_path, "inputs_" + list_simu[0] + ".nc")
    ).compute()
    if "latitude" in temp:
        lat = np.array(temp["latitude"])
        lon = np.array(temp["longitude"])
    else:
        lat = np.array(temp["lat"])
        lon = np.array(temp["lon"])
    return x_all, y_all, lat, lon


def input_for_training(x, skip_historical, history, len_historical):
    """Slide a length-``history`` window over ``x`` to build input samples."""
    time_length = x.shape[0]
    if skip_historical:
        x_train_to_return = np.array(
            [
                x[i : i + history]
                for i in range(len_historical - history + 1, time_length - history + 1)
            ]
        )
    else:
        x_train_to_return = np.array(
            [x[i : i + history] for i in range(0, time_length - history + 1)]
        )
    return x_train_to_return


def output_for_training(y, skip_historical, history, len_historical):
    """Take the window end point of each input window as the target sample."""
    time_length = y.shape[0]
    if skip_historical:
        y_train_to_return = np.array(
            [
                y[i + history - 1]
                for i in range(len_historical - history + 1, time_length - history + 1)
            ]
        )
    else:
        y_train_to_return = np.array(
            [y[i + history - 1] for i in range(0, time_length - history + 1)]
        )
    return y_train_to_return


def split_train_val(x, y, train_ratio=0.9):
    """Randomly split samples into train and val subsets."""
    shuffled_ids = np.random.permutation(x.shape[0])
    train_len = int(train_ratio * x.shape[0])
    train_ids = shuffled_ids[:train_len]
    val_ids = shuffled_ids[train_len:]
    return x[train_ids], y[train_ids], x[val_ids], y[val_ids]


class ClimateBenchDataset(io.Dataset):
    """ClimateBench dataset for climate projection (input: CO2/SO2/CH4/BC, output: tas).

    Each partition is built the same way as the original workflow: load the
    netCDF files (ssp simulations concatenated with the historical period),
    slide a length-``history`` window over time, then split train/val randomly
    (partition "train"/"val") or keep the last 21 test samples (partition
    "test"). Input normalization statistics are fitted on the train partition
    only and must be shared with val/test through ``inp_transform``/
    ``out_transform`` or ``set_normalize``.

    Args:
        root_dir (str): Dataset root containing "train_val" and "test" folders.
        input_keys (Tuple[str, ...]): Input dict keys, ("x", "lead_times").
        label_keys (Tuple[str, ...]): Label dict keys, e.g. ("tas",).
        partition (str): One of "train", "val", "test". Defaults to "train".
        history (int): Length of the input time window. Defaults to 10.
        list_train_simu (List[str]): Simulation names used for train/val.
            Defaults to ["ssp126", "ssp370", "ssp585", "historical", "hist-GHG", "hist-aer"].
        list_test_simu (List[str]): Simulation names used for test. Defaults to ["ssp245"].
        variables (List[str]): Input variable names. Defaults to ["CO2", "SO2", "CH4", "BC"].
        out_variables (Union[str, List[str]]): Output variable name(s). Defaults to "tas".
        train_ratio (float): Fraction of train_val samples used for training. Defaults to 0.9.
        seed (Optional[int]): Seed applied right before the train/val split so that
            separate "train" and "val" instances with the same seed get the same
            split. None uses the global numpy random state. Defaults to None.
        inp_transform (Optional[Normalize]): Input normalization, fitted on the train
            partition. Defaults to None (fitted here for partition "train").
        out_transform (Optional[Normalize]): Output normalization. Defaults to None
            (identity for partition "train").
    """

    def __init__(
        self,
        root_dir: str,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        partition: str = "train",
        history: int = 10,
        list_train_simu: Optional[List[str]] = None,
        list_test_simu: Optional[List[str]] = None,
        variables: Optional[List[str]] = None,
        out_variables: Union[str, List[str]] = "tas",
        train_ratio: float = 0.9,
        seed: Optional[int] = None,
        inp_transform: Optional[Normalize] = None,
        out_transform: Optional[Normalize] = None,
    ):
        super().__init__()
        if list_train_simu is None:
            list_train_simu = [
                "ssp126",
                "ssp370",
                "ssp585",
                "historical",
                "hist-GHG",
                "hist-aer",
            ]
        if list_test_simu is None:
            list_test_simu = ["ssp245"]
        if variables is None:
            variables = ["CO2", "SO2", "CH4", "BC"]
        if isinstance(out_variables, str):
            out_variables = [out_variables]
        self.root_dir = root_dir
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.partition = partition
        self.history = history
        self.list_train_simu = list_train_simu
        self.list_test_simu = list_test_simu
        self.variables = variables
        self.out_variables = out_variables
        self.train_ratio = train_ratio
        self.len_historical = 165
        self.inp_transform = inp_transform
        self.out_transform = out_transform

        if partition == "test":
            dict_x, dict_y, lat, lon = load_x_y(
                os.path.join(root_dir, "test"), list_test_simu, out_variables
            )
            x = input_for_training(
                dict_x[list_test_simu[0]],
                skip_historical=True,
                history=history,
                len_historical=self.len_historical,
            )
            y = output_for_training(
                dict_y[list_test_simu[0]],
                skip_historical=True,
                history=history,
                len_historical=self.len_historical,
            )
        else:
            dict_x, dict_y, lat, lon = load_x_y(
                os.path.join(root_dir, "train_val"), list_train_simu, out_variables
            )
            x_all = np.concatenate(
                [
                    input_for_training(
                        dict_x[simu],
                        skip_historical=i < 2,
                        history=history,
                        len_historical=self.len_historical,
                    )
                    for i, simu in enumerate(dict_x.keys())
                ],
                axis=0,
            )
            y_all = np.concatenate(
                [
                    output_for_training(
                        dict_y[simu],
                        skip_historical=i < 2,
                        history=history,
                        len_historical=self.len_historical,
                    )
                    for i, simu in enumerate(dict_y.keys())
                ],
                axis=0,
            )
            if seed is not None:
                np.random.seed(seed)
            x_train, y_train, x_val, y_val = split_train_val(x_all, y_all, train_ratio)
            if partition == "val":
                x, y = x_val, y_val
            else:
                x, y = x_train, y_train

        self.lat = lat
        self.lon = lon
        self.X_train_all = x
        self.Y_train_all = y
        if partition == "train":
            self.inp_transform = self.get_normalize(self.X_train_all)
            self.out_transform = Normalize(np.array([0.0]), np.array([1.0]))
        if partition == "test":
            self.X_train_all = self.X_train_all[-21:]
            self.Y_train_all = self.Y_train_all[-21:]
            self.get_rmse_normalization()

    def get_normalize(self, data):
        mean = np.mean(data, axis=(0, 1, 3, 4))
        std = np.std(data, axis=(0, 1, 3, 4))
        return Normalize(mean, std)

    def set_normalize(self, inp_normalize, out_normalize):
        self.inp_transform = inp_normalize
        self.out_transform = out_normalize

    def get_rmse_normalization(self):
        y_avg = paddle.to_tensor(self.Y_train_all).squeeze(1).mean(0)  # H, W
        w_lat = np.cos(np.deg2rad(self.lat))  # (H,)
        w_lat = w_lat / w_lat.mean()
        w_lat = paddle.to_tensor(w_lat).unsqueeze(-1).astype(y_avg.dtype)  # (H, 1)
        self.y_normalization = paddle.abs(paddle.mean(y_avg * w_lat))

    def get_lat_lon(self):
        return self.lat, self.lon

    def __len__(self):
        return self.X_train_all.shape[0]

    def __getitem__(self, index):
        inp = self.inp_transform(self.X_train_all[index])  # (T, V, H, W)
        out = self.out_transform(self.Y_train_all[index])  # (Vout, H, W)
        input_item = {
            self.input_keys[0]: inp,
            self.input_keys[1]: np.asarray([0.0], dtype="float32"),
        }
        label_item = {self.label_keys[0]: out}
        return input_item, label_item, None
