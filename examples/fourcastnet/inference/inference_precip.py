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

import os
import sys
import time
import glob
import numpy as np
from collections import defaultdict
import argparse
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
import h5py

import paddle
import paddle.distributed as dist

from utils.weighted_acc_rmse import weighted_rmse_paddle_channels, weighted_acc_paddle_channels, unlog_tp_paddle, top_quantiles_error_paddle
from utils.YParams import YParams
from utils.data_loader_multifiles import get_data_loader
from networks.afnonet import AFNONet, PrecipNet
from utils.logging_utils import VDLLogger, get_logger

from infer_utils import gaussian_perturb, load_model, date_to_hours
from visu_result import visu_precip

DECORRELATION_TIME = 8  # 2 days for preicp


def setup(params):
    #get data loader
    valid_data_loader, valid_dataset = get_data_loader(
        params, params.inf_data_path, dist.is_initialized(), train=False)
    img_shape_x = valid_dataset.img_shape_x
    img_shape_y = valid_dataset.img_shape_y
    params.img_shape_x = img_shape_x
    params.img_shape_y = img_shape_y

    logger.info('Loading trained model checkpoint from {}'.format(params[
        'best_checkpoint_path']))

    in_channels = np.array(params.in_channels)
    out_channels = np.array(
        params.in_channels)  # for the backbone model, will be reset later
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)
    params['N_in_channels'] = n_in_channels
    params['N_out_channels'] = n_out_channels
    params.means = np.load(params.global_means_path)[
        0, out_channels]  # needed to standardize wind data
    params.stds = np.load(params.global_stds_path)[0, out_channels]

    # load wind model
    if params.nettype_wind == 'afno':
        model_wind = AFNONet(params)
    if 'model_wind_path' not in params:
        raise Exception("no backbone model weights specified")
    checkpoint_file = params['model_wind_path']
    model_wind = load_model(model_wind, checkpoint_file)

    # reset channels for precip
    params['N_out_channels'] = len(params['out_channels'])
    # load the model
    if params.nettype == 'afno':
        model = AFNONet(params)
    else:
        raise Exception("not implemented")

    model = PrecipNet(params, backbone=model)
    checkpoint_file = params['best_checkpoint_path']
    model = load_model(model, checkpoint_file)

    # load the validation data
    files_paths = glob.glob(params.inf_data_path + "/*.h5")
    files_paths.sort()
    # which year
    yr = 0

    logger.info('Loading validation data')
    logger.info('Validation data from {}'.format(files_paths[yr]))

    valid_data_full = h5py.File(files_paths[yr], 'r')['fields']

    # precip paths
    path = params.precip + '/out_of_sample'
    precip_paths = glob.glob(path + "/*.h5")
    precip_paths.sort()
    logger.info('Loading validation precip data')
    logger.info('Validation data from {}'.format(precip_paths[0]))
    valid_data_tp_full = h5py.File(precip_paths[0], 'r')['tp']
    return valid_data_full, valid_data_tp_full, model_wind, model


def autoregressive_inference(params, ic, valid_data_full, valid_data_tp_full,
                             model_wind, model):
    ic = int(ic)
    exp_dir = params['experiment_dir']
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

    #initialize memory for image sequences and RMSE/ACC, tqe for precip
    valid_loss = paddle.zeros((prediction_length, n_out_channels))
    acc = paddle.zeros((prediction_length, n_out_channels))
    acc_unweighted = paddle.zeros((prediction_length, n_out_channels))
    tqe = paddle.zeros((prediction_length, n_out_channels))

    # wind seqs
    seq_real = paddle.zeros(
        (prediction_length, n_in_channels, img_shape_x, img_shape_y))
    seq_pred = paddle.zeros(
        (prediction_length, n_in_channels, img_shape_x, img_shape_y))
    # precip sequences
    seq_real_tp = paddle.zeros(
        (prediction_length, n_out_channels, img_shape_x, img_shape_y))
    seq_pred_tp = paddle.zeros(
        (prediction_length, n_out_channels, img_shape_x, img_shape_y))

    valid_data = valid_data_full[
        ic:(ic + prediction_length * dt + n_history * dt):dt,
        in_channels, 0:720]  #extract valid data from first year
    # standardize
    valid_data = (valid_data - means) / stds
    valid_data = paddle.to_tensor(valid_data, dtype=paddle.float32)

    len_ic = prediction_length * dt
    valid_data_tp = valid_data_tp_full[ic:(
        ic + prediction_length * dt):dt, 0:720].reshape(
            len_ic, n_out_channels, 720,
            img_shape_y)  #extract valid data from first year
    # log normalize
    eps = params.precip_eps
    valid_data_tp = np.log1p(valid_data_tp / eps)
    valid_data_tp = paddle.to_tensor(valid_data_tp, dtype=paddle.float32)

    m = paddle.to_tensor(np.load(params.time_means_path_tp)[0][
        out_channels])[:, 0:img_shape_x]  # climatology
    m = paddle.unsqueeze(m, 0)

    #autoregressive inference
    logger.info('Begin autoregressive inference')

    with paddle.no_grad():
        for i in range(valid_data.shape[0]):
            if i == 0:  #start of sequence
                first = valid_data[0:n_history + 1]
                first_tp = valid_data_tp[0:1]
                future = valid_data[n_history + 1]
                future_tp = valid_data_tp[1]
                for h in range(n_history + 1):
                    seq_real[h] = first[h][
                        0:n_in_channels]  #extract history from 1st 
                    seq_pred[h] = seq_real[h]

                seq_real_tp[0] = unlog_tp_paddle(first_tp[0])
                seq_pred_tp[0] = unlog_tp_paddle(first_tp[0])
                if params.perturb:
                    first = gaussian_perturb(
                        first, level=params.n_level)  # perturb the ic
                future_pred = model_wind(first)
                future_pred_tp = model(future_pred)
            else:
                if i < prediction_length - 1:
                    future = valid_data[n_history + i + 1]
                    future_tp = valid_data_tp[i + 1]
                future_pred = model_wind(future_pred)  #autoregressive step
                future_pred_tp = model(future_pred)  # tp diagnosis

            if i < prediction_length - 1:  #not on the last step
                seq_pred[n_history + i + 1] = future_pred[0]
                seq_real[n_history + i + 1] = future
                seq_pred_tp[i + 1] = unlog_tp_paddle(
                    future_pred_tp[0]
                )  # this predicts 6-12 precip: 0 -> 6 (afno) -> 6-12 precip 
                seq_real_tp[i + 1] = unlog_tp_paddle(
                    future_tp)  # which is the i+1th validation data
                #collect history
                history_stack = seq_pred[i + 1:i + 2 + n_history]

            # ic for next wind step
            future_pred = history_stack

            pred = paddle.unsqueeze(seq_pred_tp[i], 0)
            tar = paddle.unsqueeze(seq_real_tp[i], 0)
            valid_loss[i] = weighted_rmse_paddle_channels(pred, tar)
            acc[i] = weighted_acc_paddle_channels(pred - m, tar - m)
            tqe[i] = top_quantiles_error_paddle(pred, tar)

            logger.info('Timestep {} of {}. TP RMS Error: {}, ACC: {}'.format(
                (i), prediction_length,
                float(valid_loss[i, 0]), float(acc[i, 0])))

    return dict(
        seq_real=seq_real_tp.numpy(),
        seq_pred=seq_pred_tp.numpy(),
        rmse=valid_loss.numpy(),
        acc=acc.numpy(),
        acc_unweighted=acc_unweighted.numpy(),
        tqe=tqe.numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml_config", default='./config/AFNO.yaml', type=str)
    parser.add_argument("--config", default='precip', type=str)
    parser.add_argument("--vis", action='store_true')
    parser.add_argument(
        "--output_dir",
        default='output/inference_fourcastnet_precip/',
        type=str,
        help='Path to store inference outputs; must also set --weights arg')
    parser.add_argument(
        "--weights",
        default='output/precip_paddle/00/training_checkpoints/best_ckpt.tar',
        type=str,
        help='Path to model weights, for use with output_dir option')

    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    vis = args.vis
    if vis:
        params['ics_type'] = 'datetime'
        params['prediction_length'] = 6
        params['date_strings'] = ["2018-04-04 00:00:00"]
    # Set up directory
    assert args.weights is not None, 'Must set --weights argument if using --output_dir'
    expDir = args.output_dir
    if not os.path.isdir(expDir):
        os.makedirs(expDir)

    params['experiment_dir'] = os.path.abspath(expDir)
    params['best_checkpoint_path'] = args.weights

    log_writer = VDLLogger(save_dir=os.path.join(
        os.path.abspath(expDir), 'vdl/'))
    logger = get_logger(
        name='FourCastNet Inference',
        log_file=os.path.join(os.path.abspath(expDir), 'inference_out.log'))

    n_ics = params['n_initial_conditions']
    n_samples_per_year = 1460

    if params["ics_type"] == 'default':
        num_samples = n_samples_per_year - params.prediction_length
        stop = num_samples
        ics = np.arange(0, stop, DECORRELATION_TIME)
        if vis:  # visualization for just the first ic (or any ic)
            ics = [0]
        n_ics = len(ics)
    elif params["ics_type"] == "datetime":
        date_strings = params["date_strings"]
        ics = []
        if params.perturb:  #for perturbations use a single date and create n_ics perturbations
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
    valid_data_full, valid_data_tp_full, model_wind, model = setup(params)

    #initialize lists for image sequences and RMSE/ACC
    results = defaultdict(list)

    #run autoregressive inference for multiple initial conditions
    for i, ic in enumerate(ics):
        t1 = time.time()
        logger.info("Initial condition {} of {}".format(i + 1, n_ics))
        res = autoregressive_inference(params, ic, valid_data_full,
                                       valid_data_tp_full, model_wind, model)
        for key, value in res.items():
            if key in ['seq_real', 'seq_pred']:
                if i == 0:
                    results[key].append(value)
            else:
                results[key].append(value)
        t2 = time.time() - t1
        logger.info("time for 1 autoreg inference = {}".format(t2))

    rmse_avg = np.asarray(results['rmse']).mean(axis=0)
    acc_avg = np.asarray(results['acc']).mean(axis=0)

    idx = 0
    for i in range(rmse_avg.shape[0]):
        log_data = dict(
            acc=float(acc_avg[i, idx]), rmse=float(rmse_avg[i, idx]))
        log_writer.log_metrics(metrics=log_data, prefix='tp', step=i)

    #save predictions and loss
    save_path = os.path.join(params['experiment_dir'],
                             'autoregressive_predictions_tp.h5')
    logger.info("Saving files at {}".format(save_path))
    with h5py.File(save_path, 'w') as f:
        for key, value in results.items():
            value = np.asarray(value)
            if key == 'seq_real':
                f.create_dataset(
                    'ground_truth',
                    data=value,
                    shape=value.shape,
                    dtype=np.float32)
            elif key == 'seq_pred':
                f.create_dataset(
                    'predicted',
                    data=value,
                    shape=value.shape,
                    dtype=np.float32)
            else:
                f.create_dataset(
                    key, data=value, shape=value.shape, dtype=np.float32)
    if vis:
        visu_precip(save_path, os.path.join(params['experiment_dir'], 'visu'))
