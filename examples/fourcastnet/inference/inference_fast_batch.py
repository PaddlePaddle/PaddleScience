#BSD 3-Clause License
#
#Copyright (c) 2022, FourCastNet authors
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#The code was authored by the following people:
#
#Jaideep Pathak - NVIDIA Corporation
#Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
#Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
#Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory 
#Ashesh Chattopadhyay - Rice University 
#Morteza Mardani - NVIDIA Corporation 
#Thorsten Kurth - NVIDIA Corporation 
#David Hall - NVIDIA Corporation 
#Zongyi Li - California Institute of Technology, NVIDIA Corporation 
#Kamyar Azizzadenesheli - Purdue University 
#Pedram Hassanzadeh - Rice University 
#Karthik Kashinath - NVIDIA Corporation 
#Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation

import os
import sys
import time
import numpy as np
from collections import defaultdict
import argparse
import glob
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
import h5py

import paddle
import paddle.distributed as dist

from utils.weighted_acc_rmse import weighted_rmse_paddle_channels, weighted_acc_paddle_channels, unweighted_acc_paddle_channels, weighted_acc_masked_paddle_channels
from utils.YParams import YParams
from utils.data_loader_multifiles import get_data_loader
from networks.afnonet import AFNONet

from utils.logging_utils import get_logger, VDLLogger

from infer_utils import gaussian_perturb, load_model, downsample, date_to_hours

fld = "u10"  # diff flds have diff decor times and hence differnt ics
if fld == "z500" or fld == "2m_temperature" or fld == "t850":
    DECORRELATION_TIME = 36  # 9 days (36) for z500, 2 (8 steps) days for u10, v10
else:
    DECORRELATION_TIME = 8  # 9 days (36) for z500, 2 (8 steps) days for u10, v10
idxes = {"u10": 0, "z500": 14, "2m_temperature": 2, "v10": 1, "t850": 5}


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
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)

    params['N_in_channels'] = n_in_channels
    params['N_out_channels'] = n_out_channels
    params.means = np.load(params.global_means_path)[
        0, out_channels]  # needed to standardize wind data
    params.stds = np.load(params.global_stds_path)[0, out_channels]

    # load the model
    if params.nettype == 'afno':
        model = AFNONet(params)
    else:
        raise Exception("not implemented")

    checkpoint_file = params['best_checkpoint_path']
    model = load_model(model, checkpoint_file)

    # load the validation data
    files_paths = glob.glob(params.inf_data_path + "/*.h5")
    files_paths.sort()
    # which year
    yr = 0
    if params.log_to_screen:
        logger.info('Loading inference data')
        logger.info('Inference data from {}'.format(files_paths[yr]))

    valid_data_full = h5py.File(files_paths[yr], 'r')['fields']

    return valid_data_full, model


def autoregressive_inference(params, batch_ics, valid_data_full, model):
    #initialize global variables
    batch_size = len(batch_ics)

    prediction_length = int(params.prediction_length)
    img_shape_x = params.img_shape_x
    img_shape_y = params.img_shape_y
    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)
    means = params.means
    stds = params.stds

    #initialize memory for image sequences and RMSE/ACC
    valid_loss = paddle.zeros((batch_size, prediction_length, n_out_channels))
    acc = paddle.ones((batch_size, prediction_length, n_out_channels))
    acc_unweighted = paddle.ones(
        (batch_size, prediction_length, n_out_channels))

    # compute metrics in a coarse resolution too if params.interp is nonzero
    valid_loss_coarse = paddle.zeros(
        (batch_size, prediction_length, n_out_channels))
    acc_coarse = paddle.ones((batch_size, prediction_length, n_out_channels))
    acc_coarse_unweighted = paddle.ones(
        (batch_size, prediction_length, n_out_channels))

    acc_land = paddle.zeros((batch_size, prediction_length, n_out_channels))
    acc_sea = paddle.zeros((batch_size, prediction_length, n_out_channels))
    if params.masked_acc:
        maskarray = paddle.to_tensor(np.load(params.maskpath)[0:720])
    start = time.time()
    valid_datas = np.zeros(
        (batch_size, prediction_length, n_in_channels, 720, 1440))
    logger.info('create data cost: {:.4f}'.format(time.time() - start))
    for i, ic in enumerate(batch_ics):
        valid_datas[i] = valid_data_full[ic:(ic + prediction_length),
                                         in_channels, 0:720]
    logger.info('read data cost: {:.4f}'.format(time.time() - start))

    means = paddle.to_tensor(means)
    stds = paddle.to_tensor(stds)

    #load time means
    if not params.use_daily_climatology:
        m = paddle.to_tensor(
            (np.load(params.time_means_path)[0][out_channels]),
            dtype=paddle.float32)[:, 0:img_shape_x]
        m = (m - means) / stds  # climatology
        m = paddle.unsqueeze(m, 0)
    else:
        # use daily clim like weyn et al. (different from rasp)
        dc_path = params.dc_path
        with h5py.File(dc_path, 'r') as f:
            dc = f['time_means_daily'][ic:ic +
                                       prediction_length]  # 1460,21,721,1440
        m = paddle.to_tensor(
            (dc[:, out_channels, 0:img_shape_x, :] - means) / stds,
            dtype=paddle.float32)

    if params.interp > 0:
        m_coarse = downsample(m, scale=params.interp)

    std = paddle.to_tensor(stds[:, 0, 0], dtype=paddle.float32)

    #autoregressive inference
    logger.info('Begin autoregressive inference')
    with paddle.no_grad():
        # -1, drop the last
        for i in range(prediction_length - 1):
            if i == 0:
                idx = idxes[fld]
                logger.info(
                    'Predicted timestep {} of {}. {} RMS Error: {}, ACC: {}'.
                    format(i, prediction_length, fld, 0.0, 1.0))
                if params.interp > 0:
                    logger.info(
                        '[COARSE] Predicted timestep {} of {}. {} RMS Error: {}, ACC: {}'.
                        format(i, prediction_length, fld, 0.0, 1.0))

                input = paddle.to_tensor(
                    valid_datas[:, i:i + 1], dtype=paddle.float32).squeeze(1)
                input = (input - means) / stds
                if params.perturb:
                    input = gaussian_perturb(
                        input, level=params.n_level)  # perturb the ic
            target = paddle.to_tensor(
                valid_datas[:, i + 1:i + 2], dtype=paddle.float32).squeeze(1)
            target = (target - means) / stds
            pred = model(input)
            input = pred  # the next predition input is current output

            #Compute metrics 
            if params.use_daily_climatology:
                clim = m[i + 1:i + 2]
                if params.interp > 0:
                    clim_coarse = m_coarse[i + 1:i + 2]
            else:
                clim = m
                if params.interp > 0:
                    clim_coarse = m_coarse

            valid_loss[:, i + 1] = weighted_rmse_paddle_channels(pred,
                                                                 target) * std
            acc[:, i + 1] = weighted_acc_paddle_channels(pred - clim,
                                                         target - clim)
            acc_unweighted[:, i + 1] = unweighted_acc_paddle_channels(
                pred - clim, target - clim)

            if params.masked_acc:
                acc_land[:, i] = weighted_acc_masked_paddle_channels(
                    pred - clim, target - clim, maskarray)
                acc_sea[:, i] = weighted_acc_masked_paddle_channels(
                    pred - clim, target - clim, 1 - maskarray)

            if params.interp > 0:
                pred = downsample(pred, scale=params.interp)
                tar = downsample(target, scale=params.interp)
                valid_loss_coarse[:, i] = weighted_rmse_paddle_channels(
                    pred, target) * std
                acc_coarse[:, i] = weighted_acc_paddle_channels(
                    pred - clim_coarse, tar - clim_coarse)
                acc_coarse_unweighted[:, i] = unweighted_acc_paddle_channels(
                    pred - clim_coarse, target - clim_coarse)

            idx = idxes[fld]
            logger.info(
                'Predicted timestep {} of {}. {} RMS Error: {}, ACC: {}'.
                format(i + 1, prediction_length, fld,
                       float(valid_loss[:, i + 1, idx].mean()),
                       float(acc[:, i + 1, idx].mean())))
            if params.interp > 0:
                logger.info(
                    '[COARSE] Predicted timestep {} of {}. {} RMS Error: {}, ACC: {}'.
                    format(i + 1, prediction_length, fld,
                           float(valid_loss_coarse[:, i + 1, idx].mean()),
                           float(acc_coarse[:, i, idx].mean())))

    # return (valid_loss.numpy(), acc.numpy(), acc_unweighted.numpy(),
    #         valid_loss_coarse.numpy(), acc_coarse.numpy(),
    #         acc_coarse_unweighted.numpy(), acc_land.numpy(), acc_sea.numpy())
    return dict(
        rmse=valid_loss.numpy(),
        acc=acc.numpy(),
        acc_unweighted=acc_unweighted.numpy(),
        acc_coarse=acc_coarse.numpy(),
        acc_coarse_unweighted=acc_coarse_unweighted.numpy(),
        rmse_coarse=valid_loss_coarse.numpy(),
        acc_land=acc_land.numpy(),
        acc_sea=acc_sea.numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml_config", default='./config/AFNO.yaml', type=str)
    parser.add_argument("--config", default='afno_backbone', type=str)
    parser.add_argument("--use_daily_climatology", action='store_true')
    parser.add_argument(
        "--output_dir",
        default='output/inference_fourcastnet_batch/',
        type=str,
        help='Path to store inference outputs; must also set --weights arg')
    parser.add_argument("--interp", default=0, type=float)
    parser.add_argument(
        "--weights",
        default='output/afno_backbone_finetune_paddle/00/training_checkpoints/best_ckpt.tar',
        type=str,
        help='Path to model weights, for use with output_dir option')

    batch_size = 8

    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)

    # Set up directory
    assert args.weights is not None, 'Must set --weights argument if using --output_dir'
    expDir = args.output_dir
    if not os.path.isdir(expDir):
        os.makedirs(expDir)

    params['interp'] = args.interp
    params['use_daily_climatology'] = args.use_daily_climatology

    params['experiment_dir'] = os.path.abspath(expDir)
    params['best_checkpoint_path'] = args.weights

    log_writer = VDLLogger(save_dir=os.path.join(
        os.path.abspath(expDir), 'vdl/'))
    logger = get_logger(
        name='FourCastNet Inference',
        log_file=os.path.join(os.path.abspath(expDir), 'inference_out.log'))

    n_ics = params['n_initial_conditions']
    if fld == "z500" or fld == "t850":
        n_samples_per_year = 1336
    else:
        n_samples_per_year = 1460

    if params["ics_type"] == 'default':
        num_samples = n_samples_per_year - params.prediction_length
        stop = num_samples
        ics = np.arange(0, stop, DECORRELATION_TIME)
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
    valid_data_full, model = setup(params)

    #initialize lists for image sequences and RMSE/ACC
    results = defaultdict(list)
    #run autoregressive inference for multiple initial conditions
    batch_ics = []
    for i, ic in enumerate(ics):
        batch_ics.append(ic)
        if len(batch_ics) >= batch_size or i >= n_ics - 1:
            logger.info("Initial condition {} of {}".format(i + 1, n_ics))
            res = autoregressive_inference(params, batch_ics, valid_data_full,
                                           model)
            for key, value in res.items():
                if key in ['seq_real', 'seq_pred']:
                    if i == 0:
                        results[key].append(value)
                else:
                    results[key].append(value)
            batch_ics = []
    for key, value in results.items():
        results[key] = np.concatenate(value, axis=0)

    # average acc and rmse
    rmse_avg = results['rmse'].mean(axis=0)
    acc_avg = results['acc'].mean(axis=0)
    if fld in ['u10', 'v10']:
        for prefix in ['u10', 'v10']:
            idx = idxes[prefix]
            for i in range(rmse_avg.shape[0]):
                log_data = dict(
                    acc=float(acc_avg[i, idx]), rmse=float(rmse_avg[i, idx]))
                log_writer.log_metrics(metrics=log_data, prefix=prefix, step=i)
    else:
        for prefix in ['z500', '2m_temperature', 't850']:
            idx = idxes[prefix]
            for i in range(rmse_avg.shape[0]):
                log_data = dict(
                    acc=float(acc_avg[i, idx]), rmse=float(rmse_avg[i, idx]))
                log_writer.log_metrics(metrics=log_data, prefix=prefix, step=i)
    log_writer.close()

    #save predictions and loss
    save_path = os.path.join(params['experiment_dir'],
                             'autoregressive_predictions_' + fld + '.h5')
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
