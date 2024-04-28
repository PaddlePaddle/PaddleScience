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

from __future__ import annotations

import importlib
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
from paddle import io

try:
    from pathlib import Path

    import xarray as xr
except ModuleNotFoundError:
    pass

NINO_WINDOW_T = 3  # Nino index is the sliding average over sst, window size is 3
CMIP6_SST_MAX = 10.198975563049316
CMIP6_SST_MIN = -16.549121856689453
CMIP5_SST_MAX = 8.991744995117188
CMIP5_SST_MIN = -9.33076286315918
CMIP6_NINO_MAX = 4.138188362121582
CMIP6_NINO_MIN = -3.5832221508026123
CMIP5_NINO_MAX = 3.8253555297851562
CMIP5_NINO_MIN = -2.691682815551758
SST_MAX = max(CMIP6_SST_MAX, CMIP5_SST_MAX)
SST_MIN = min(CMIP6_SST_MIN, CMIP5_SST_MIN)


def scale_sst(sst):
    return (sst - SST_MIN) / (SST_MAX - SST_MIN)


def scale_back_sst(sst):
    return (SST_MAX - SST_MIN) * sst + SST_MIN


def prepare_inputs_targets(
    len_time, input_gap, input_length, pred_shift, pred_length, samples_gap
):
    """Prepares the input and target indices for training.

    Args:
        len_time (int): The total number of time steps in the dataset.
        input_gap (int): time gaps between two consecutive input frames.
        input_length (int): the number of input frames.
        pred_shift (int): the lead_time of the last target to be predicted.
        pred_length (int): the number of frames to be predicted.
        samples_gap (int): stride of seq sampling.

    """

    if pred_shift < pred_length:
        raise ValueError("pred_shift should be small than pred_length")
    input_span = input_gap * (input_length - 1) + 1
    pred_gap = pred_shift // pred_length
    input_ind = np.arange(0, input_span, input_gap)
    target_ind = np.arange(0, pred_shift, pred_gap) + input_span + pred_gap - 1
    ind = np.concatenate([input_ind, target_ind]).reshape(1, input_length + pred_length)
    max_n_sample = len_time - (input_span + pred_shift - 1)
    ind = ind + np.arange(max_n_sample)[:, np.newaxis] @ np.ones(
        (1, input_length + pred_length), dtype=int
    )
    return ind[::samples_gap]


def fold(data, size=36, stride=12):
    """inverse of unfold/sliding window operation
    only applicable to the case where the size of the sliding windows is n*stride

    Args:
        data (tuple[int,...]): The input data.(N, size, *).
        size (int, optional): The size of a single datum.The  Defaults to 36.
        stride (int, optional): The step.Defaults to 12.

    Returns:
        outdata (np.array): (N_, *).N/size is the number/width of sliding blocks
    """

    if size % stride != 0:
        raise ValueError("size modulo stride should be zero")
    times = size // stride
    remain = (data.shape[0] - 1) % times
    if remain > 0:
        ls = list(data[::times]) + [data[-1, -(remain * stride) :]]
        outdata = np.concatenate(ls, axis=0)  # (36*(151//3+1)+remain*stride, *, 15)
    else:
        outdata = np.concatenate(data[::times], axis=0)  # (36*(151/3+1), *, 15)
    assert (
        outdata.shape[0] == size * ((data.shape[0] - 1) // times + 1) + remain * stride
    )
    return outdata


def data_transform(data, num_years_per_model):
    """The transform of the input data.

    Args:
        data (Tuple[list,...]): The input data.Shape of (N, 36, *).
        num_years_per_model (int): The number of years associated with each model.151/140.

    """

    length = data.shape[0]
    assert length % num_years_per_model == 0
    num_models = length // num_years_per_model
    outdata = np.stack(
        np.split(data, length / num_years_per_model, axis=0), axis=-1
    )  # (151, 36, *, 15)
    # cmip6sst outdata.shape = (151, 36, 24, 48, 15) = (year, month, lat, lon, model)
    # cmip5sst outdata.shape = (140, 36, 24, 48, 17)
    # cmip6nino outdata.shape = (151, 36, 15)
    # cmip5nino outdata.shape = (140, 36, 17)
    outdata = fold(outdata, size=36, stride=12)
    # cmip6sst outdata.shape = (1836, 24, 48, 15), 1836 == 151 * 12 + 24
    # cmip5sst outdata.shape = (1704, 24, 48, 17)
    # cmip6nino outdata.shape = (1836, 15)
    # cmip5nino outdata.shape = (1704, 17)

    # check output data
    assert outdata.shape[-1] == num_models
    assert not np.any(np.isnan(outdata))
    return outdata


def read_raw_data(ds_dir, out_dir=None):
    """read and process raw cmip data from CMIP_train.nc and CMIP_label.nc

    Args:
        ds_dir (str): the path of the dataset.
        out_dir (str): the path of output. Defaults to None.

    """

    train_cmip = xr.open_dataset(Path(ds_dir) / "CMIP_train.nc").transpose(
        "year", "month", "lat", "lon"
    )
    label_cmip = xr.open_dataset(Path(ds_dir) / "CMIP_label.nc").transpose(
        "year", "month"
    )
    # train_cmip.sst.values.shape = (4645, 36, 24, 48)

    # select longitudes
    lon = train_cmip.lon.values
    lon = lon[np.logical_and(lon >= 95, lon <= 330)]
    train_cmip = train_cmip.sel(lon=lon)

    cmip6sst = data_transform(
        data=train_cmip.sst.values[:2265], num_years_per_model=151
    )
    cmip5sst = data_transform(
        data=train_cmip.sst.values[2265:], num_years_per_model=140
    )
    cmip6nino = data_transform(
        data=label_cmip.nino.values[:2265], num_years_per_model=151
    )
    cmip5nino = data_transform(
        data=label_cmip.nino.values[2265:], num_years_per_model=140
    )

    # cmip6sst.shape = (1836, 24, 48, 15)
    # cmip5sst.shape = (1704, 24, 48, 17)
    assert len(cmip6sst.shape) == 4
    assert len(cmip5sst.shape) == 4
    assert len(cmip6nino.shape) == 2
    assert len(cmip5nino.shape) == 2
    # store processed data for faster data access
    if out_dir is not None:
        ds_cmip6 = xr.Dataset(
            {
                "sst": (["month", "lat", "lon", "model"], cmip6sst),
                "nino": (["month", "model"], cmip6nino),
            },
            coords={
                "month": np.repeat(
                    np.arange(1, 13)[None], cmip6nino.shape[0] // 12, axis=0
                ).flatten(),
                "lat": train_cmip.lat.values,
                "lon": train_cmip.lon.values,
                "model": np.arange(15) + 1,
            },
        )
        ds_cmip6.to_netcdf(Path(out_dir) / "cmip6.nc")
        ds_cmip5 = xr.Dataset(
            {
                "sst": (["month", "lat", "lon", "model"], cmip5sst),
                "nino": (["month", "model"], cmip5nino),
            },
            coords={
                "month": np.repeat(
                    np.arange(1, 13)[None], cmip5nino.shape[0] // 12, axis=0
                ).flatten(),
                "lat": train_cmip.lat.values,
                "lon": train_cmip.lon.values,
                "model": np.arange(17) + 1,
            },
        )
        ds_cmip5.to_netcdf(Path(out_dir) / "cmip5.nc")
    train_cmip.close()
    label_cmip.close()
    return cmip6sst, cmip5sst, cmip6nino, cmip5nino


def cat_over_last_dim(data):
    """treat different models (15 from CMIP6, 17 from CMIP5) as batch_size
    e.g., cmip6sst.shape = (178, 38, 24, 48, 15), converted_cmip6sst.shape = (2670, 38, 24, 48)
    e.g., cmip5sst.shape = (165, 38, 24, 48, 15), converted_cmip6sst.shape = (2475, 38, 24, 48)

    """

    return np.concatenate(np.moveaxis(data, -1, 0), axis=0)


class ENSODataset(io.Dataset):
    """The El Niño/Southern Oscillation dataset.

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("input",).
        label_keys (Tuple[str, ...]): Name of label keys, such as ("output",).
        data_dir (str): The directory  of data.
        weight_dict (Optional[Dict[str, Union[Callable, float]]]): Define the weight of each constraint variable. Defaults to None.
        in_len (int, optional): The length of input data. Defaults to 12.
        out_len (int, optional): The length of out data. Defaults to 26.
        in_stride (int, optional): The stride of input data. Defaults to 1.
        out_stride (int, optional): The stride of output data. Defaults to 1.
        train_samples_gap (int, optional): The stride of sequence sampling during training. Defaults to 10.
            e.g., samples_gap = 10, the first seq contains [0, 1, ..., T-1] frame indices, the second seq contains [10, 11, .., T+9]
        eval_samples_gap (int, optional): The stride of sequence sampling during eval. Defaults to 11.
        normalize_sst (bool, optional): Whether to use normalization. Defaults to True.
        batch_size (int, optional): Batch size. Defaults to 1.
        num_workers (int, optional): The num of workers. Defaults to 1.
        training (str, optional): Training pathse. Defaults to "train".

    """

    # Whether support batch indexing for speeding up fetching process.
    batch_index: bool = False

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        data_dir: str,
        weight_dict: Optional[Dict[str, float]] = None,
        in_len=12,
        out_len=26,
        in_stride=1,
        out_stride=1,
        train_samples_gap=10,
        eval_samples_gap=11,
        normalize_sst=True,
        # datamodule_only
        batch_size=1,
        num_workers=1,
        training="train",
    ):
        super(ENSODataset, self).__init__()
        if importlib.util.find_spec("xarray") is None:
            raise ModuleNotFoundError(
                "To use RadarDataset, please install 'xarray' via: `pip install "
                "xarray` first."
            )
        if importlib.util.find_spec("pathlib") is None:
            raise ModuleNotFoundError(
                "To use RadarDataset, please install 'pathlib' via: `pip install "
                "pathlib` first."
            )
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.data_dir = data_dir
        self.weight_dict = {} if weight_dict is None else weight_dict
        if weight_dict is not None:
            self.weight_dict = {key: 1.0 for key in self.label_keys}
            self.weight_dict.update(weight_dict)

        self.in_len = in_len
        self.out_len = out_len
        self.in_stride = in_stride
        self.out_stride = out_stride
        self.train_samples_gap = train_samples_gap
        self.eval_samples_gap = eval_samples_gap
        self.normalize_sst = normalize_sst
        # datamodule_only
        self.batch_size = batch_size
        if num_workers != 1:
            raise ValueError(
                "Current implementation does not support `num_workers != 1`!"
            )
        self.num_workers = num_workers
        self.training = training

        # pre-data
        cmip6sst, cmip5sst, cmip6nino, cmip5nino = read_raw_data(self.data_dir)
        # TODO: more flexible train/val/test split
        self.sst_train = [cmip6sst, cmip5sst[..., :-2]]
        self.nino_train = [cmip6nino, cmip5nino[..., :-2]]
        self.sst_eval = [cmip5sst[..., -2:-1]]
        self.nino_eval = [cmip5nino[..., -2:-1]]
        self.sst_test = [cmip5sst[..., -1:]]
        self.nino_test = [cmip5nino[..., -1:]]

        self.sst, self.target_nino = self.create_data()

    def create_data(
        self,
    ):
        if self.training == "train":
            sst_cmip6 = self.sst_train[0]
            nino_cmip6 = self.nino_train[0]
            sst_cmip5 = self.sst_train[1]
            nino_cmip5 = self.nino_train[1]
            samples_gap = self.train_samples_gap
        elif self.training == "eval":
            sst_cmip6 = None
            nino_cmip6 = None
            sst_cmip5 = self.sst_eval[0]
            nino_cmip5 = self.nino_eval[0]
            samples_gap = self.eval_samples_gap
        elif self.training == "test":
            sst_cmip6 = None
            nino_cmip6 = None
            sst_cmip5 = self.sst_test[0]
            nino_cmip5 = self.nino_test[0]
            samples_gap = self.eval_samples_gap

        # cmip6 (N, *, 15)
        # cmip5 (N, *, 17)
        sst = []
        target_nino = []

        nino_idx_slice = slice(
            self.in_len, self.in_len + self.out_len - NINO_WINDOW_T + 1
        )  # e.g., 12:36
        if sst_cmip6 is not None:
            assert len(sst_cmip6.shape) == 4
            assert len(nino_cmip6.shape) == 2
            idx_sst = prepare_inputs_targets(
                len_time=sst_cmip6.shape[0],
                input_length=self.in_len,
                input_gap=self.in_stride,
                pred_shift=self.out_len * self.out_stride,
                pred_length=self.out_len,
                samples_gap=samples_gap,
            )

            sst.append(cat_over_last_dim(sst_cmip6[idx_sst]))
            target_nino.append(
                cat_over_last_dim(nino_cmip6[idx_sst[:, nino_idx_slice]])
            )
        if sst_cmip5 is not None:
            assert len(sst_cmip5.shape) == 4
            assert len(nino_cmip5.shape) == 2
            idx_sst = prepare_inputs_targets(
                len_time=sst_cmip5.shape[0],
                input_length=self.in_len,
                input_gap=self.in_stride,
                pred_shift=self.out_len * self.out_stride,
                pred_length=self.out_len,
                samples_gap=samples_gap,
            )
            sst.append(cat_over_last_dim(sst_cmip5[idx_sst]))
            target_nino.append(
                cat_over_last_dim(nino_cmip5[idx_sst[:, nino_idx_slice]])
            )

        # sst data containing both the input and target
        self.sst = np.concatenate(sst, axis=0)  # (N, in_len+out_len, lat, lon)
        if self.normalize_sst:
            self.sst = scale_sst(self.sst)
        # nino data containing the target only
        self.target_nino = np.concatenate(
            target_nino, axis=0
        )  # (N, out_len+NINO_WINDOW_T-1)
        assert self.sst.shape[0] == self.target_nino.shape[0]
        assert self.sst.shape[1] == self.in_len + self.out_len
        assert self.target_nino.shape[1] == self.out_len - NINO_WINDOW_T + 1
        return self.sst, self.target_nino

    def get_datashape(self):
        return {"sst": self.sst.shape, "nino target": self.target_nino.shape}

    def __len__(self):
        return self.sst.shape[0]

    def __getitem__(self, idx):
        sst_data = self.sst[idx].astype("float32")
        sst_data = sst_data[..., np.newaxis]
        in_seq = sst_data[: self.in_len, ...]  # ( in_len, lat, lon, 1)
        target_seq = sst_data[self.in_len :, ...]  # ( in_len, lat, lon, 1)
        weight_item = self.weight_dict

        if self.training == "train":
            input_item = {self.input_keys[0]: in_seq}
            label_item = {
                self.label_keys[0]: target_seq,
            }

            return input_item, label_item, weight_item
        else:
            input_item = {self.input_keys[0]: in_seq}
            label_item = {
                self.label_keys[0]: target_seq,
                self.label_keys[1]: self.target_nino[idx],
            }

            return input_item, label_item, weight_item
