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
from typing import TYPE_CHECKING
from typing import Dict

from paddle import device

from ppsci.utils import logger
from ppsci.utils import misc

if TYPE_CHECKING:
    from ppsci import solver


def update_train_loss(
    trainer: "solver.Solver", loss_dict: Dict[str, float], batch_size: int
):
    for key in loss_dict:
        if key not in trainer.train_output_info:
            trainer.train_output_info[key] = misc.AverageMeter(key, "7.5f")
        trainer.train_output_info[key].update(float(loss_dict[key]), batch_size)
        if key not in trainer.train_loss_info:
            trainer.train_loss_info[key] = misc.AverageMeter(key, ".5f")
        trainer.train_loss_info[key].update(float(loss_dict[key]))


def update_eval_loss(
    trainer: "solver.Solver", loss_dict: Dict[str, float], batch_size: int
):
    for key in loss_dict:
        if key not in trainer.eval_output_info:
            trainer.eval_output_info[key] = misc.AverageMeter(key, "7.5f")
        trainer.eval_output_info[key].update(float(loss_dict[key]), batch_size)


def log_train_info(
    trainer: "solver.Solver", batch_size: int, epoch_id: int, iter_id: int
):
    lr_msg = f"lr: {trainer.optimizer.get_lr():.5f}"

    metric_msg = ", ".join(
        [
            f"{key}: {trainer.train_output_info[key].avg:.5f}"
            for key in trainer.train_output_info
        ]
    )

    time_msg = ", ".join(
        [trainer.train_time_info[key].mean for key in trainer.train_time_info]
    )

    ips_msg = f"ips: {batch_size / trainer.train_time_info['batch_cost'].avg:.2f}"
    if trainer.benchmark_flag:
        ips_msg += " samples/s"

    eta_sec = (
        (trainer.epochs - epoch_id + 1) * trainer.iters_per_epoch - iter_id
    ) * trainer.train_time_info["batch_cost"].avg
    eta_msg = f"eta: {str(datetime.timedelta(seconds=int(eta_sec)))}"

    epoch_width = len(str(trainer.epochs))
    iters_width = len(str(trainer.iters_per_epoch))
    log_str = (
        f"[Train][Epoch {epoch_id:>{epoch_width}}/{trainer.epochs}]"
        f"[Iter {iter_id:>{iters_width}}/{trainer.iters_per_epoch}] {lr_msg}, "
        f"{metric_msg}, {time_msg}, {ips_msg}, {eta_msg}"
    )
    if trainer.benchmark_flag:
        max_mem_reserved_msg = (
            f"max_mem_reserved: {device.cuda.max_memory_reserved() // (1 << 20)} MB"
        )
        max_mem_allocated_msg = (
            f"max_mem_allocated: {device.cuda.max_memory_allocated() // (1 << 20)} MB"
        )
        log_str += f", {max_mem_reserved_msg}, {max_mem_allocated_msg}"
    logger.info(log_str)

    logger.scalar(
        {
            "train/lr": trainer.optimizer.get_lr(),
            **{
                f"train/{key}": trainer.train_output_info[key].avg
                for key in trainer.train_output_info
            },
        },
        step=trainer.global_step,
        vdl_writer=trainer.vdl_writer,
        wandb_writer=trainer.wandb_writer,
    )


def log_eval_info(
    trainer: "solver.Solver",
    batch_size: int,
    epoch_id: int,
    iters_per_epoch: int,
    iter_id: int,
):
    metric_msg = ", ".join(
        [
            f"{key}: {trainer.eval_output_info[key].avg:.5f}"
            for key in trainer.eval_output_info
        ]
    )

    time_msg = ", ".join(
        [trainer.eval_time_info[key].mean for key in trainer.eval_time_info]
    )

    ips_msg = f"ips: {batch_size / trainer.eval_time_info['batch_cost'].avg:.2f}"

    eta_sec = (iters_per_epoch - iter_id) * trainer.eval_time_info["batch_cost"].avg
    eta_msg = f"eta: {str(datetime.timedelta(seconds=int(eta_sec)))}"

    epoch_width = len(str(trainer.epochs))
    iters_width = len(str(iters_per_epoch))
    logger.info(
        f"[Eval][Epoch {epoch_id:>{epoch_width}}/{trainer.epochs}]"
        f"[Iter {iter_id:>{iters_width}}/{iters_per_epoch}] "
        f"{metric_msg}, {time_msg}, {ips_msg}, {eta_msg}"
    )

    logger.scalar(
        {
            f"eval/{key}": trainer.eval_output_info[key].avg
            for key in trainer.eval_output_info
        },
        step=trainer.global_step,
        vdl_writer=trainer.vdl_writer,
        wandb_writer=trainer.wandb_writer,
    )
