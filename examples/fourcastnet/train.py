"""BSD 3-Clause License
Copyright (c) 2022, FourCastNet authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The code was authored by the following people:

Jaideep Pathak - NVIDIA Corporation
Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory
Ashesh Chattopadhyay - Rice University
Morteza Mardani - NVIDIA Corporation
Thorsten Kurth - NVIDIA Corporation
David Hall - NVIDIA Corporation
Zongyi Li - California Institute of Technology, NVIDIA Corporation
Kamyar Azizzadenesheli - Purdue University
Pedram Hassanzadeh - Rice University
Karthik Kashinath - NVIDIA Corporation
Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation
"""

import argparse
import os
import random
import time
from collections import OrderedDict

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn as nn
from networks.afnonet import AFNONet
from networks.afnonet import PrecipNet
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
from utils.darcy_loss import LpLoss
from utils.data_loader_multifiles import get_data_loader
from utils.logging_utils import VDLLogger
from utils.logging_utils import get_logger
from utils.logging_utils import print_dict
from utils.weighted_acc_rmse import unlog_tp_paddle
from utils.weighted_acc_rmse import weighted_rmse_paddle
from utils.YParams import YParams

DECORRELATION_TIME = 36  # 9 days


def set_seed(seed=1024):
    """Set random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


class Trainer(object):
    """The trainer class is a thread that handles training operations"""

    def count_parameters(self):
        """
        Count the number of trainable parameters
        """
        return sum(p.numel() for p in self.model.parameters() if not p.stop_gradient)

    def __init__(self, params, world_rank, world_size, logger, log_writer):
        """ " Initialize a new trainer."""
        self.params = params
        self.world_rank = world_rank
        self.logger = logger
        self.log_writer = log_writer
        self.world_size = world_size
        self.distributed = world_size != 1

        (
            self.train_data_loader,
            self.train_dataset,
            self.train_sampler,
        ) = get_data_loader(
            params, params.train_data_path, self.distributed, train=True
        )
        self.valid_data_loader, self.valid_dataset = get_data_loader(
            params, params.valid_data_path, self.distributed, train=False
        )
        self.loss_obj = LpLoss()
        logger.info("rank {}, data loader initialized".format(world_rank))

        params.crop_size_x = self.valid_dataset.crop_size_x
        params.crop_size_y = self.valid_dataset.crop_size_y
        params.img_shape_x = self.valid_dataset.img_shape_x
        params.img_shape_y = self.valid_dataset.img_shape_y

        # precip models
        self.precip = True if "precip" in params else False

        if self.precip:
            if "model_wind_path" not in params:
                raise Exception("no backbone model weights specified")
            # load a wind model
            # the wind model has out channels = in channels
            out_channels = np.array(params["in_channels"])
            params["N_out_channels"] = len(out_channels)

            if params.nettype_wind == "afno":
                self.model_wind = AFNONet(params)
            else:
                raise ValueError(f"params.nettype({params.nettype}) is not implemented")

            if dist.is_initialized():
                self.model_wind = paddle.DataParallel(self.model_wind)
            self.load_model_wind(params.model_wind_path)
            self.switch_off_grad(self.model_wind)  # no backprop through the wind model

        # reset out_channels for precip models
        if self.precip:
            params["N_out_channels"] = len(params["out_channels"])

        if params.nettype == "afno":
            self.model = AFNONet(params)
        else:
            raise ValueError(f"params.nettype({params.nettype}) is not implemented")

        # precip model
        if self.precip:
            self.model = PrecipNet(params, backbone=self.model)

        self.iters = 0
        self.startEpoch = 0
        if params.scheduler == "ReduceLROnPlateau":
            self.scheduler = paddle.optimizer.lr.ReduceOnPlateau(
                learning_rate=params.lr, factor=0.2, patience=5, mode="min"
            )
        elif params.scheduler == "CosineAnnealingLR":
            self.scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
                learning_rate=params.lr,
                T_max=params.max_epochs,
                last_epoch=self.startEpoch - 1,
            )
        else:
            self.scheduler = None

        self.optimizer = paddle.optimizer.Adam(
            parameters=self.model.parameters(), learning_rate=self.scheduler
        )

        if params.enable_amp == True:
            self.gscaler = paddle.amp.GradScaler()

        if self.distributed:
            self.model = paddle.DataParallel(self.model)

        if params.resuming:
            logger.info("Loading checkpoint %s" % params.checkpoint_path)
            self.restore_checkpoint(params.checkpoint_path)
        if params.two_step_training:
            if params.resuming == False and params.pretrained == True:
                logger.info(
                    "Starting from pretrained one-step afno model at %s"
                    % params.pretrained_ckpt_path
                )
                self.restore_checkpoint(params.pretrained_ckpt_path)
                self.iters = 0
                self.startEpoch = 0
                # logger.info("Pretrained checkpoint was trained for %d epochs"%self.startEpoch)
                # logger.info("Adding %d epochs specified in config file for refining pretrained model"%self.params.max_epochs)
                # self.params.max_epochs += self.startEpoch

        self.epoch = self.startEpoch
        logger.info(
            "Number of trainable model parameters: {}".format(
                self.count_parameters().item()
            )
        )

    def switch_off_grad(self, model):
        """switch off gradient computation for the provided model."""
        for param in model.parameters():
            param.stop_gradient = True

    def train(self):
        """this function is just for training purposes."""
        logger.info("Starting Training Loop...")
        best_valid_loss = 1.0e6
        for epoch in range(self.startEpoch, self.params.max_epochs):
            if self.distributed:
                self.train_sampler.set_epoch(epoch)

            tr_time, data_time, train_logs = self.train_one_epoch()

            run_val = True
            if run_val:
                valid_time, valid_logs = self.validate_one_epoch()

            if self.params.scheduler == "ReduceLROnPlateau":
                self.scheduler.step(valid_logs["valid_loss"])
            elif self.params.scheduler == "CosineAnnealingLR":
                self.scheduler.step()

            if self.world_rank == 0:
                if self.params.save_checkpoint:
                    # checkpoint at the end of every epoch
                    self.save_checkpoint(self.params.checkpoint_path)
                    if epoch >= self.params.max_epochs - 10:
                        # save last 10 epochs
                        self.save_checkpoint(
                            self.params.checkpoint_path[:-4] + str(epoch) + ".tar"
                        )
                    if run_val and valid_logs["valid_loss"] <= best_valid_loss:
                        # logger.info("Val loss improved from {} to {}".format(best_valid_loss, valid_logs["valid_loss"]))
                        self.save_checkpoint(self.params.best_checkpoint_path)
                        best_valid_loss = valid_logs["valid_loss"]

    def train_one_epoch(self):
        """ " Run one epoch. Trains model on training set."""
        self.epoch += 1
        logger.info("Starting training epoch {}".format(self.epoch))
        tr_start = time.time()
        data_time = 0
        self.model.train()
        logger.info("total iters:{}".format(len(self.train_data_loader)))
        cur_data_start = time.time()

        for i, data in enumerate(self.train_data_loader, 0):
            self.iters += 1
            cur_data_time = time.time() - cur_data_start
            data_start = time.time()
            inp, tar = data
            if self.params.orography and self.params.two_step_training:
                orog = inp[:, -2:-1]
            if "residual_field" in self.params.target:
                tar -= inp[:, 0 : tar.size()[1]]
            data_time += time.time() - data_start
            cur_model_start = time.time()
            self.model.clear_gradients()
            if self.params.two_step_training:
                with paddle.amp.auto_cast(self.params.enable_amp):
                    gen_step_one = self.model(inp).cast(paddle.float32)
                    loss_step_one = self.loss_obj(
                        gen_step_one, tar[:, 0 : self.params.N_out_channels]
                    )
                    if self.params.orography:
                        gen_step_two = self.model(
                            paddle.concat((gen_step_one, orog), axis=1)
                        )
                    else:
                        gen_step_two = self.model(gen_step_one).cast(paddle.float32)
                    loss_step_two = self.loss_obj(
                        gen_step_two,
                        tar[
                            :,
                            self.params.N_out_channels : 2 * self.params.N_out_channels,
                        ],
                    )
                    loss = loss_step_one + loss_step_two
            else:
                with paddle.amp.auto_cast(self.params.enable_amp):
                    if (
                        self.precip
                    ):  # use a wind model to predict 17(+n) channels at t+dt
                        with paddle.no_grad():
                            inp = self.model_wind(inp).cast(paddle.float32)
                        gen = self.model(inp.detach()).cast(paddle.float32)
                    else:
                        gen = self.model(inp).cast(paddle.float32)
                    loss = self.loss_obj(gen, tar)

            cur_model_time = time.time() - cur_model_start
            cur_back_start = time.time()
            if self.params.enable_amp:
                self.gscaler.scale(loss).backward()
                self.gscaler.step(self.optimizer)
            else:
                loss.backward()
                self.optimizer.step()
            cur_back_time = time.time() - cur_back_start
            if self.params.enable_amp:
                self.gscaler.update()

            if dist.get_rank() == 0 and (
                i % 10 == 0 or i >= len(self.train_data_loader) - 1
            ):
                if self.params.two_step_training:
                    logger.info(
                        "Train epoch: [{}/{}], iter: [{}/{}], lr: {:.8f}, data_time: {:.5f}, model_time: {:.5f}, back_time: {:.5f}, loss: {:.5f}, loss_step_one: {:.5f}, loss_step_two: {:.5f}".format(
                            self.epoch,
                            self.params.max_epochs,
                            i,
                            len(self.train_data_loader),
                            self.optimizer.get_lr(),
                            cur_data_time,
                            cur_model_time,
                            cur_back_time,
                            loss.item(),
                            loss_step_one.item(),
                            loss_step_two.item(),
                        )
                    )
                else:
                    logger.info(
                        "Train epoch: [{}/{}], iter: [{}/{}], lr: {:.8f}, data_time: {:.5f}, model_time: {:.5f}, back_time: {:.5f}, loss: {:.5f}".format(
                            self.epoch,
                            self.params.max_epochs,
                            i,
                            len(self.train_data_loader),
                            self.optimizer.get_lr(),
                            cur_data_time,
                            cur_model_time,
                            cur_back_time,
                            loss.item(),
                        )
                    )
            if dist.get_rank() == 0:
                if self.params.two_step_training:
                    log_data = dict(
                        lr=self.optimizer.get_lr(),
                        loss=loss.item(),
                        loss_step_one=loss_step_one.item(),
                        loss_step_two=loss_step_two.item(),
                    )
                else:
                    log_data = dict(
                        lr=self.optimizer.get_lr(),
                        loss=loss.item(),
                    )
                self.log_writer.log_metrics(
                    metrics=log_data, prefix="TRAIN", step=self.iters
                )
            cur_data_start = time.time()

        if self.params.two_step_training:
            logs = {
                "loss": loss,
                "loss_step_one": loss_step_one,
                "loss_step_two": loss_step_two,
            }
        else:
            logs = {"loss": loss}

        if dist.is_initialized():
            for key in sorted(logs.keys()):
                dist.all_reduce(logs[key].detach())
                logs[key] = float(logs[key] / dist.get_world_size())
            if dist.get_rank() == 0:
                if self.params.two_step_training:
                    log_data = dict(
                        loss_total=logs["loss"],
                        loss_step_one_total=logs["loss_step_one"],
                        loss_step_two_total=logs["loss_step_two"],
                    )
                else:
                    log_data = dict(
                        loss_total=logs["loss"],
                    )
                self.log_writer.log_metrics(
                    metrics=log_data, prefix="TRAIN", step=self.iters
                )
        tr_time = time.time() - tr_start
        logger.info(
            "Epoch {} training finished!, Time: {} sec".format(self.epoch, tr_time)
        )
        return tr_time, data_time, logs

    def validate_one_epoch(self):
        """ " Validate the model on one epoch."""
        logger.info("Starting eval epoch {}".format(self.epoch))
        self.model.eval()

        if self.params.normalization == "minmax":
            raise Exception("minmax normalization not supported")
        elif self.params.normalization == "zscore":
            mult = paddle.to_tensor(
                np.load(self.params.global_stds_path)[0, self.params.out_channels, 0, 0]
            )

        valid_buff = paddle.zeros([3], dtype=paddle.float32)
        valid_loss = 0
        valid_l1 = 0
        valid_steps = 0
        valid_weighted_rmse = paddle.zeros(
            [self.params.N_out_channels], dtype=paddle.float32
        )
        valid_weighted_acc = paddle.zeros(
            [self.params.N_out_channels], dtype=paddle.float32
        )

        valid_start = time.time()
        sample_idx = np.random.randint(len(self.valid_data_loader))
        with paddle.no_grad():
            for i, data in enumerate(self.valid_data_loader, 0):
                inp, tar = data
                if self.params.orography and self.params.two_step_training:
                    orog = inp[:, -2:-1]

                if self.params.two_step_training:
                    gen_step_one = self.model(inp)
                    loss_step_one = self.loss_obj(
                        gen_step_one, tar[:, 0 : self.params.N_out_channels]
                    )
                    if self.params.orography:
                        gen_step_two = self.model(
                            paddle.concat((gen_step_one, orog), axis=1)
                        )
                    else:
                        gen_step_two = self.model(gen_step_one)
                    loss_step_two = self.loss_obj(
                        gen_step_two,
                        tar[
                            :,
                            self.params.N_out_channels : 2 * self.params.N_out_channels,
                        ],
                    )
                    loss = loss_step_one + loss_step_two
                    valid_loss += loss
                    valid_l1 += nn.functional.l1_loss(
                        gen_step_one, tar[:, 0 : self.params.N_out_channels]
                    )
                else:
                    if self.precip:
                        with paddle.no_grad():
                            inp = self.model_wind(inp)
                        gen = self.model(inp.detach())
                    else:
                        gen = self.model(inp)
                    loss = self.loss_obj(gen, tar)
                    valid_loss += loss
                    valid_l1 += paddle.nn.functional.l1_loss(gen, tar)

                valid_steps += 1.0

                if dist.get_rank() == 0 and i % 10 == 0:
                    if self.params.two_step_training:
                        logger.info(
                            "Eval epoch: [{}/{}], iter: [{}/{}], loss:{:.5f}, loss_step_one:{:.5f}, loss_step_two:{:.5f}".format(
                                self.epoch,
                                self.params.max_epochs,
                                i,
                                len(self.valid_data_loader),
                                loss.item(),
                                loss_step_one.item(),
                                loss_step_two.item(),
                            )
                        )
                    else:
                        logger.info(
                            "Eval epoch: [{}/{}], iter: [{}/{}], loss:{:.5f}".format(
                                self.epoch,
                                self.params.max_epochs,
                                i,
                                len(self.valid_data_loader),
                                loss.item(),
                            )
                        )

                if self.precip:
                    gen = unlog_tp_paddle(gen, self.params.precip_eps)
                    tar = unlog_tp_paddle(tar, self.params.precip_eps)

                # direct prediction weighted rmse
                if self.params.two_step_training:
                    if "residual_field" in self.params.target:
                        valid_weighted_rmse += weighted_rmse_paddle(
                            (gen_step_one + inp),
                            (tar[:, 0 : self.params.N_out_channels] + inp),
                        )
                    else:
                        valid_weighted_rmse += weighted_rmse_paddle(
                            gen_step_one, tar[:, 0 : self.params.N_out_channels]
                        )
                else:
                    if "residual_field" in self.params.target:
                        valid_weighted_rmse += weighted_rmse_paddle(
                            (gen + inp), (tar + inp)
                        )
                    else:
                        valid_weighted_rmse += weighted_rmse_paddle(gen, tar)

        valid_buff[0], valid_buff[1], valid_buff[2] = valid_loss, valid_l1, valid_steps
        if dist.is_initialized():
            dist.all_reduce(valid_buff)
            dist.all_reduce(valid_weighted_rmse)

        # divide by number of steps
        valid_buff[0:2] = valid_buff[0:2] / valid_buff[2]
        valid_weighted_rmse = valid_weighted_rmse / valid_buff[2]
        if not self.precip:
            valid_weighted_rmse *= mult

        # download buffers
        valid_buff_cpu = valid_buff.detach().cpu().numpy()
        valid_weighted_rmse_cpu = valid_weighted_rmse.detach().cpu().numpy()

        valid_time = time.time() - valid_start
        valid_weighted_rmse = mult * paddle.mean(valid_weighted_rmse, axis=0)
        if self.precip:
            logs = {
                "valid_l1": valid_buff_cpu[1],
                "valid_loss": valid_buff_cpu[0],
                "valid_rmse_tp": valid_weighted_rmse_cpu[0],
            }
        else:
            try:
                logs = {
                    "valid_l1": valid_buff_cpu[1],
                    "valid_loss": valid_buff_cpu[0],
                    "valid_rmse_u10": valid_weighted_rmse_cpu[0],
                    "valid_rmse_v10": valid_weighted_rmse_cpu[1],
                }
            except:
                logs = {
                    "valid_l1": valid_buff_cpu[1],
                    "valid_loss": valid_buff_cpu[0],
                    "valid_rmse_u10": valid_weighted_rmse_cpu[0],
                }  # , "valid_rmse_v10": valid_weighted_rmse[1]}

        if self.precip:
            logger.info(
                "Eval epoch: [{}/{}], avg loss: {:.5f}, avg l1: {}, avg rmse tp: {:.5f}".format(
                    self.epoch,
                    self.params.max_epochs,
                    logs["valid_loss"],
                    logs["valid_l1"],
                    logs["valid_rmse_tp"],
                )
            )
        else:
            try:
                logger.info(
                    "Eval epoch: [{}/{}], avg loss: {:.5f}, avg l1: {:.5f}, avg rmse u10: {:.5f}, avg rmse v10: {:.5f}".format(
                        self.epoch,
                        self.params.max_epochs,
                        logs["valid_loss"],
                        logs["valid_l1"],
                        logs["valid_rmse_u10"],
                        logs["valid_rmse_v10"],
                    )
                )
            except:
                logger.info(
                    "Eval epoch: [{}/{}], avg loss: {:.5f}, avg l1: {}, avg rmse u10: {:.5f}".format(
                        self.epoch,
                        self.params.max_epochs,
                        logs["valid_loss"],
                        logs["valid_l1"],
                        logs["valid_rmse_u10"],
                    )
                )
        logger.info(
            "Epoch {} validation finished!, Time: {} sec".format(self.epoch, valid_time)
        )
        return valid_time, logs

    def load_model_wind(self, model_path):
        if self.params.log_to_screen:
            logger.info("Loading the wind model weights from {}".format(model_path))
        checkpoint = paddle.load(model_path)
        if dist.is_initialized():
            self.model_wind.set_state_dict(checkpoint["model_state"])
        else:
            new_model_state = OrderedDict()
            model_key = "model_state" if "model_state" in checkpoint else "state_dict"
            for key in checkpoint[model_key].keys():
                new_model_state[key] = checkpoint[model_key][key]
            self.model_wind.set_state_dict(new_model_state)
        self.model_wind.eval()

    def save_checkpoint(self, checkpoint_path, model=None):
        """We intentionally require a checkpoint_dir to be passed
        in order to allow Ray Tune to use this function"""

        if not model:
            model = self.model

        paddle.save(
            {
                "iters": self.iters,
                "epoch": self.epoch,
                "model_state": model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )

    def restore_checkpoint(self, checkpoint_path):
        """We intentionally require a checkpoint_dir to be passed
        in order to allow Ray Tune to use this function"""
        logger.info("restore checkpoint from {}".format(checkpoint_path))
        checkpoint = paddle.load(checkpoint_path)
        self.model.set_state_dict(checkpoint["model_state"])
        logger.info("restore fininshed!!!")

        self.iters = checkpoint["iters"]
        self.startEpoch = checkpoint["epoch"]
        if (
            self.params.resuming
        ):  # restore checkpoint is used for finetuning as well as resuming. If finetuning (i.e., not resuming), restore checkpoint does not load optimizer state, instead uses config specified lr.
            self.optimizer.set_state_dict(checkpoint["optimizer_state_dict"])


if __name__ == "__main__":
    dist.init_parallel_env()
    world_size = dist.get_world_size()
    world_rank = dist.get_rank()
    set_seed()

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default="00", type=str)
    parser.add_argument("--yaml_config", default="config/AFNO.yaml", type=str)
    parser.add_argument("--config", default="default", type=str)
    parser.add_argument("--enable_amp", default=False, action="store_true")
    parser.add_argument("--epsilon_factor", default=0, type=float)

    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params["epsilon_factor"] = args.epsilon_factor
    params["world_size"] = world_size
    # set device
    device = "gpu:{}".format(dist.ParallelEnv().dev_id)
    device = paddle.set_device(device)

    params["global_batch_size"] = params.batch_size * params["world_size"]
    params["batch_size"] = int(params.batch_size)

    # Set up directory
    expDir = os.path.join(params.exp_dir, args.config + "_paddle", str(args.run_num))

    logger = get_logger(name="FourCastNet", log_file=os.path.join(expDir, "out.log"))
    log_writer = VDLLogger(save_dir=os.path.join(expDir, "vdl/"))

    if dist.get_rank() == 0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir)
            os.makedirs(os.path.join(expDir, "training_checkpoints/"))

    params["experiment_dir"] = os.path.abspath(expDir)
    params["checkpoint_path"] = os.path.join(expDir, "training_checkpoints/ckpt.tar")
    params["best_checkpoint_path"] = os.path.join(
        expDir, "training_checkpoints/best_ckpt.tar"
    )

    # Do not comment this line out please:
    args.resuming = True if os.path.isfile(params.checkpoint_path) else False

    params["resuming"] = args.resuming
    params["enable_amp"] = args.enable_amp

    params["log_to_screen"] = (world_rank == 0) and params["log_to_screen"]

    params["in_channels"] = np.array(params["in_channels"])
    params["out_channels"] = np.array(params["out_channels"])
    if params.orography:
        params["N_in_channels"] = len(params["in_channels"]) + 1
    else:
        params["N_in_channels"] = len(params["in_channels"])

    params["N_out_channels"] = len(params["out_channels"])

    if dist.get_rank() == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(os.path.join(expDir, "hyperparams.yaml"), "w") as hpfile:
            yaml.dump(hparams, hpfile)

    print_dict(params.params, logger)

    trainer = Trainer(params, world_rank, world_size, logger, log_writer)
    trainer.train()
    log_writer.close()
    logger.info("DONE ---- rank %d" % world_rank)
