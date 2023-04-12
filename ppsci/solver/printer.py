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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime

from ppsci.utils import logger
from ppsci.utils import misc


def update_train_loss(trainer, loss_dict, batch_size):
    # update_output_info
    for key in loss_dict:
        if key not in trainer.train_output_info:
            trainer.train_output_info[key] = misc.AverageMeter(key, "7.5f")
        trainer.train_output_info[key].update(float(loss_dict[key]), batch_size)


def update_eval_loss(trainer, loss_dict, batch_size):
    # update_output_info
    for key in loss_dict:
        if key not in trainer.eval_output_info:
            trainer.eval_output_info[key] = misc.AverageMeter(key, "7.5f")
        trainer.eval_output_info[key].update(float(loss_dict[key]), batch_size)


def log_train_info(trainer, batch_size, epoch_id, iter_id):
    lr_msg = f"lr: {trainer.optimizer.get_lr():.8f}"

    metric_msg = ", ".join(
        [
            f"{key}: {trainer.train_output_info[key].avg:.5f}"
            for key in trainer.train_output_info
        ]
    )

    time_msg = ", ".join(
        [trainer.train_time_info[key].mean for key in trainer.train_time_info]
    )

    ips_msg = (
        f"ips: {batch_size / trainer.train_time_info['batch_cost'].avg:.5f} samples/s"
    )

    eta_sec = (
        (trainer.epochs - epoch_id + 1) * trainer.iters_per_epoch - iter_id
    ) * trainer.train_time_info["batch_cost"].avg
    eta_msg = f"eta: {str(datetime.timedelta(seconds=int(eta_sec))):s}"
    logger.info(
        f"[Train][Epoch {epoch_id}/{trainer.epochs}]"
        f"[Iter: {iter_id}/{trainer.iters_per_epoch}] {lr_msg}, "
        f"{metric_msg}, {time_msg}, {ips_msg}, {eta_msg}"
    )

    logger.scaler(
        name="lr",
        value=trainer.optimizer.get_lr(),
        step=trainer.global_step,
        writer=trainer.vdl_writer,
    )

    for key in trainer.train_output_info:
        logger.scaler(
            name=f"train_{key}",
            value=trainer.train_output_info[key].avg,
            step=trainer.global_step,
            writer=trainer.vdl_writer,
        )


def log_eval_info(trainer, batch_size, epoch_id, iters_per_epoch, iter_id):
    metric_msg = ", ".join(
        [
            f"{key}: {trainer.eval_output_info[key].avg:.5f}"
            for key in trainer.eval_output_info
        ]
    )

    time_msg = ", ".join(
        [trainer.eval_time_info[key].mean for key in trainer.eval_time_info]
    )

    ips_msg = (
        f"ips: {batch_size / trainer.eval_time_info['batch_cost'].avg:.5f}" f"samples/s"
    )

    eta_sec = (iters_per_epoch - iter_id) * trainer.eval_time_info["batch_cost"].avg
    eta_msg = f"eta: {str(datetime.timedelta(seconds=int(eta_sec))):s}"
    logger.info(
        f"[Eval][Epoch {epoch_id}][Iter: {iter_id}/{iters_per_epoch}] "
        f"{metric_msg}, {time_msg}, {ips_msg}, {eta_msg}"
    )

    for key in trainer.eval_output_info:
        logger.scaler(
            name=f"eval_{key}",
            value=trainer.eval_output_info[key].avg,
            step=trainer.global_step,
            writer=trainer.vdl_writer,
        )
