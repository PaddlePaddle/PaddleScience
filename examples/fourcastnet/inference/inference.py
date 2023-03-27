# BSD 3-Clause License
#
# Copyright (c) 2022, FourCastNet authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The code was authored by the following people:
#
# Jaideep Pathak - NVIDIA Corporation
# Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
# Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
# Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory
# Ashesh Chattopadhyay - Rice University
# Morteza Mardani - NVIDIA Corporation
# Thorsten Kurth - NVIDIA Corporation
# David Hall - NVIDIA Corporation
# Zongyi Li - California Institute of Technology, NVIDIA Corporation
# Kamyar Azizzadenesheli - Purdue University
# Pedram Hassanzadeh - Rice University
# Karthik Kashinath - NVIDIA Corporation
# Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation

import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
import h5py
import paddle
import paddle.distributed as dist
from infer_utils import date_to_hours
from infer_utils import downsample
from infer_utils import gaussian_perturb
from infer_utils import load_model
from networks.afnonet import AFNONet
from utils.data_loader_multifiles import get_data_loader
from utils.logging_utils import VDLLogger
from utils.logging_utils import get_logger
from utils.weighted_acc_rmse import unweighted_acc_paddle_channels
from utils.weighted_acc_rmse import weighted_acc_masked_paddle_channels
from utils.weighted_acc_rmse import weighted_acc_paddle_channels
from utils.weighted_acc_rmse import weighted_rmse_paddle_channels
from utils.YParams import YParams
from visu_result import visu_wind

fld = "u10"  # diff flds have diff decor times and hence differnt ics
if fld == "z500" or fld == "2m_temperature" or fld == "t850":
    DECORRELATION_TIME = 36  # 9 days (36) for z500, 2 (8 steps) days for u10, v10
else:
    DECORRELATION_TIME = 8  # 9 days (36) for z500, 2 (8 steps) days for u10, v10
idxes = {"u10": 0, "z500": 14, "2m_temperature": 2, "v10": 1, "t850": 5}


def setup(params):
    # get data loader
    valid_data_loader, valid_dataset = get_data_loader(
        params, params.inf_data_path, dist.is_initialized(), train=False
    )
    img_shape_x = valid_dataset.img_shape_x
    img_shape_y = valid_dataset.img_shape_y
    params.img_shape_x = img_shape_x
    params.img_shape_y = img_shape_y

    logger.info(
        "Loading trained model checkpoint from {}".format(
            params["best_checkpoint_path"]
        )
    )

    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)

    params["N_in_channels"] = n_in_channels
    params["N_out_channels"] = n_out_channels
    params.means = np.load(params.global_means_path)[
        0, out_channels
    ]  # needed to standardize wind data
    params.stds = np.load(params.global_stds_path)[0, out_channels]

    # load the model
    if params.nettype == "afno":
        model = AFNONet(params)
    else:
        raise ValueError(f"params.nettype({params.nettype}) is not implemented")

    checkpoint_file = params["best_checkpoint_path"]
    model = load_model(model, checkpoint_file)

    # load the validation data
    files_paths = glob.glob(params.inf_data_path + "/*.h5")
    files_paths.sort()
    # which year
    yr = 0
    valid_data_full = h5py.File(files_paths[yr], "r")["fields"]

    return valid_data_full, model


def autoregressive_inference(params, ic, valid_data_full, model):
    ic = int(ic)
    # initialize global variables
    dt = int(params.dt)
    prediction_length = int(params.prediction_length / dt)
    n_history = params.n_history
    img_shape_x = params.img_shape_x
    img_shape_y = params.img_shape_y
    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)
    means = params.means
    stds = params.stds

    # initialize memory for image sequences and RMSE/ACC
    valid_loss = paddle.zeros((prediction_length, n_out_channels))
    acc = paddle.zeros((prediction_length, n_out_channels))

    # compute metrics in a coarse resolution too if params.interp is nonzero
    valid_loss_coarse = paddle.zeros((prediction_length, n_out_channels))
    acc_coarse = paddle.zeros((prediction_length, n_out_channels))
    acc_coarse_unweighted = paddle.zeros((prediction_length, n_out_channels))

    acc_unweighted = paddle.zeros((prediction_length, n_out_channels))
    seq_real = paddle.zeros(
        (prediction_length, n_in_channels, img_shape_x, img_shape_y)
    )
    seq_pred = paddle.zeros(
        (prediction_length, n_in_channels, img_shape_x, img_shape_y)
    )

    acc_land = paddle.zeros((prediction_length, n_out_channels))
    acc_sea = paddle.zeros((prediction_length, n_out_channels))
    if params.masked_acc:
        maskarray = paddle.to_tensor(np.load(params.maskpath)[0:720])

    valid_data = valid_data_full[
        ic : (ic + prediction_length * dt + n_history * dt) : dt, in_channels, 0:720
    ]  # extract valid data from first year
    # standardize
    valid_data = (valid_data - means) / stds
    valid_data = paddle.to_tensor(valid_data, dtype=paddle.float32)

    # load time means
    if not params.use_daily_climatology:
        m = paddle.to_tensor(
            (np.load(params.time_means_path)[0][out_channels] - means) / stds,
            dtype=paddle.float32,
        )[
            :, 0:img_shape_x
        ]  # climatology
        m = paddle.unsqueeze(m, 0)
    else:
        # use daily clim like weyn et al. (different from rasp)
        dc_path = params.dc_path
        with h5py.File(dc_path, "r") as f:
            dc = f["time_means_daily"][
                ic : ic + prediction_length * dt : dt
            ]  # 1460,21,721,1440
        m = paddle.to_tensor(
            (dc[:, out_channels, 0:img_shape_x, :] - means) / stds, dtype=paddle.float32
        )

    if params.interp > 0:
        m_coarse = downsample(m, scale=params.interp)

    std = paddle.to_tensor(stds[:, 0, 0], dtype=paddle.float32)

    # autoregressive inference
    logger.info("Begin autoregressive inference")

    with paddle.no_grad():
        for i in range(valid_data.shape[0]):
            if i == 0:  # start of sequence
                first = valid_data[0 : n_history + 1]
                future = valid_data[n_history + 1]
                for h in range(n_history + 1):
                    seq_real[h] = first[h][0:n_out_channels]  # extract history from 1st
                    seq_pred[h] = seq_real[h]
                if params.perturb:
                    first = gaussian_perturb(
                        first, level=params.n_level
                    )  # perturb the ic
                future_pred = model(first)
            else:
                if i < prediction_length - 1:
                    future = valid_data[n_history + i + 1]
                future_pred = model(future_pred)  # autoregressive step

            if i < prediction_length - 1:  # not on the last step
                seq_pred[n_history + i + 1] = future_pred[0]
                seq_real[n_history + i + 1] = future
                history_stack = seq_pred[i + 1 : i + 2 + n_history]

            future_pred = history_stack
            # Compute metrics
            if params.use_daily_climatology:
                clim = m[i : i + 1]
                if params.interp > 0:
                    clim_coarse = m_coarse[i : i + 1]
            else:
                clim = m
                if params.interp > 0:
                    clim_coarse = m_coarse

            pred = paddle.unsqueeze(seq_pred[i], 0)
            tar = paddle.unsqueeze(seq_real[i], 0)
            valid_loss[i] = weighted_rmse_paddle_channels(pred, tar) * std
            acc[i] = weighted_acc_paddle_channels(pred - clim, tar - clim)
            acc_unweighted[i] = unweighted_acc_paddle_channels(pred - clim, tar - clim)

            if params.masked_acc:
                acc_land[i] = weighted_acc_masked_paddle_channels(
                    pred - clim, tar - clim, maskarray
                )
                acc_sea[i] = weighted_acc_masked_paddle_channels(
                    pred - clim, tar - clim, 1 - maskarray
                )

            if params.interp > 0:
                pred = downsample(pred, scale=params.interp)
                tar = downsample(tar, scale=params.interp)
                valid_loss_coarse[i] = weighted_rmse_paddle_channels(pred, tar) * std
                acc_coarse[i] = weighted_acc_paddle_channels(
                    pred - clim_coarse, tar - clim_coarse
                )
                acc_coarse_unweighted[i] = unweighted_acc_paddle_channels(
                    pred - clim_coarse, tar - clim_coarse
                )

            if params.log_to_screen:
                idx = idxes[fld]
                logger.info(
                    "Predicted timestep {} of {}. {} RMS Error: {}, ACC: {}".format(
                        i,
                        prediction_length,
                        fld,
                        float(valid_loss[i, idx]),
                        float(acc[i, idx]),
                    )
                )
                if params.interp > 0:
                    logger.info(
                        "[COARSE] Predicted timestep {} of {}. {} RMS Error: {}, ACC: {}".format(
                            i,
                            prediction_length,
                            fld,
                            float(valid_loss_coarse[i, idx]),
                            float(acc_coarse[i, idx]),
                        )
                    )

    return dict(
        seq_real=seq_real.numpy(),
        seq_pred=seq_pred.numpy(),
        rmse=valid_loss.numpy(),
        acc=acc.numpy(),
        acc_unweighted=acc_unweighted.numpy(),
        acc_coarse=acc_coarse.numpy(),
        acc_coarse_unweighted=acc_coarse_unweighted.numpy(),
        rmse_coarse=valid_loss_coarse.numpy(),
        acc_land=acc_land.numpy(),
        acc_sea=acc_sea.numpy(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default="./config/AFNO.yaml", type=str)
    parser.add_argument("--config", default="afno_backbone", type=str)
    parser.add_argument("--use_daily_climatology", action="store_true")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument(
        "--output_dir",
        default="output/inference_fourcastnet/",
        type=str,
        help="Path to store inference outputs; must also set --weights arg",
    )
    parser.add_argument("--interp", default=0, type=float)
    parser.add_argument(
        "--weights",
        default="output/afno_backbone_finetune_paddle/00/training_checkpoints/best_ckpt.tar",
        type=str,
        help="Path to model weights",
    )

    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params["interp"] = args.interp
    params["use_daily_climatology"] = args.use_daily_climatology

    vis = args.vis
    if vis:
        params["ics_type"] = "datetime"
        params["prediction_length"] = 20
        params["date_strings"] = ["2018-09-08 00:00:00"]

    # Set up directory
    assert args.weights is not None, "Must set --weights argument if using --output_dir"
    expDir = args.output_dir
    if not os.path.isdir(expDir):
        os.makedirs(expDir)

    params["experiment_dir"] = os.path.abspath(expDir)
    params["best_checkpoint_path"] = args.weights

    log_writer = VDLLogger(save_dir=os.path.join(os.path.abspath(expDir), "vdl/"))
    logger = get_logger(
        name="FourCastNet Inference",
        log_file=os.path.join(os.path.abspath(expDir), "inference_out.log"),
    )

    n_ics = params["n_initial_conditions"]
    if fld == "z500" or fld == "t850":
        n_samples_per_year = 1336
    else:
        n_samples_per_year = 1460

    if params["ics_type"] == "default":
        num_samples = n_samples_per_year - params.prediction_length
        stop = num_samples
        ics = np.arange(0, stop, DECORRELATION_TIME)
        if vis:  # visualization for just the first ic (or any ic)
            ics = [0]
        n_ics = len(ics)
    elif params["ics_type"] == "datetime":
        date_strings = params["date_strings"]
        ics = []
        if (
            params.perturb
        ):  # for perturbations use a single date and create n_ics perturbations
            n_ics = params["n_perturbations"]
            date = date_strings[0]
            hours_since_jan_01_epoch = date_to_hours(date)
            for ii in range(n_ics):
                ics.append(int(hours_since_jan_01_epoch / 6))
        else:
            for date in date_strings:
                hours_since_jan_01_epoch = date_to_hours(date)
                ics.append(int(hours_since_jan_01_epoch / 6))
        n_ics = len(ics)

    logger.info("Inference for {} initial conditions".format(n_ics))

    # get data and models
    valid_data_full, model = setup(params)
    results = defaultdict(list)
    # run autoregressive inference for multiple initial conditions
    for i, ic in enumerate(ics):
        logger.info("Initial condition {} of {}".format(i + 1, n_ics))
        res = autoregressive_inference(params, ic, valid_data_full, model)
        for key, value in res.items():
            if key in ["seq_real", "seq_pred"]:
                if i == 0:
                    results[key].append(value)
            else:
                results[key].append(value)

    # average acc and rmse
    rmse_avg = np.asarray(results["rmse"]).mean(axis=0)
    acc_avg = np.asarray(results["acc"]).mean(axis=0)
    if fld in ["u10", "v10"]:
        for prefix in ["u10", "v10"]:
            idx = idxes[prefix]
            for i in range(rmse_avg.shape[0]):
                log_data = dict(
                    acc=float(acc_avg[i, idx]), rmse=float(rmse_avg[i, idx])
                )
                log_writer.log_metrics(metrics=log_data, prefix=prefix, step=i)
    else:
        for prefix in ["z500", "2m_temperature", "t850"]:
            idx = idxes[prefix]
            for i in range(rmse_avg.shape[0]):
                log_data = dict(
                    acc=float(acc_avg[i, idx]), rmse=float(rmse_avg[i, idx])
                )
                log_writer.log_metrics(metrics=log_data, prefix=prefix, step=i)
    log_writer.close()

    # save predictions and loss
    save_path = os.path.join(
        params["experiment_dir"], "autoregressive_predictions_" + fld + ".h5"
    )
    if params.log_to_screen:
        logger.info("Saving files at {}".format(save_path))
    with h5py.File(save_path, "w") as f:
        for key, value in results.items():
            value = np.asarray(value)
            if key == "seq_real":
                f.create_dataset(
                    "ground_truth", data=value, shape=value.shape, dtype=np.float32
                )
            elif key == "seq_pred":
                f.create_dataset(
                    "predicted", data=value, shape=value.shape, dtype=np.float32
                )
            else:
                f.create_dataset(key, data=value, shape=value.shape, dtype=np.float32)

    if vis:
        visu_wind(
            save_path,
            os.path.join(params["experiment_dir"], "visu"),
            params.means,
            params.stds,
        )
