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

import datetime
from typing import TYPE_CHECKING
from typing import Dict
from typing import Optional

from paddle import device

from ppsci.utils import logger
from ppsci.utils import misc

if TYPE_CHECKING:
    from ppsci import solver


def update_train_loss(
    solver: "solver.Solver", loss_dict: Dict[str, float], batch_size: int
):
    for key in loss_dict:
        if key not in solver.train_output_info:
            solver.train_output_info[key] = misc.AverageMeter(key, "7.5f")
        solver.train_output_info[key].update(float(loss_dict[key]), batch_size)
        if key not in solver.train_loss_info:
            solver.train_loss_info[key] = misc.AverageMeter(key, ".5f")
        solver.train_loss_info[key].update(float(loss_dict[key]))


def update_eval_loss(
    solver: "solver.Solver", loss_dict: Dict[str, float], batch_size: int
):
    for key in loss_dict:
        if key not in solver.eval_output_info:
            solver.eval_output_info[key] = misc.AverageMeter(key, "7.5f")
        solver.eval_output_info[key].update(float(loss_dict[key]), batch_size)


def log_train_info(
    solver: "solver.Solver", batch_size: int, epoch_id: int, iter_id: int
):
    lr_msg = f"lr: {solver.optimizer.get_lr():.5f}"

    metric_msg = ", ".join(
        [
            f"{key}: {solver.train_output_info[key].avg:.5f}"
            for key in solver.train_output_info
        ]
    )

    time_msg = ", ".join(
        [solver.train_time_info[key].mean for key in solver.train_time_info]
    )

    ips_msg = f"ips: {batch_size / solver.train_time_info['batch_cost'].avg:.2f}"
    if solver.benchmark_flag:
        ips_msg += " samples/s"

    eta_sec = (
        (solver.epochs - epoch_id + 1) * solver.iters_per_epoch - iter_id
    ) * solver.train_time_info["batch_cost"].avg
    eta_msg = f"eta: {str(datetime.timedelta(seconds=int(eta_sec)))}"

    epoch_width = len(str(solver.epochs))
    iters_width = len(str(solver.iters_per_epoch))
    log_str = (
        f"[Train][Epoch {epoch_id:>{epoch_width}}/{solver.epochs}]"
        f"[Iter {iter_id:>{iters_width}}/{solver.iters_per_epoch}] {lr_msg}, "
        f"{metric_msg}, {time_msg}, {ips_msg}, {eta_msg}"
    )
    if solver.benchmark_flag:
        max_mem_reserved_msg = (
            f"max_mem_reserved: {device.cuda.max_memory_reserved() // (1 << 20)} MB"
        )
        max_mem_allocated_msg = (
            f"max_mem_allocated: {device.cuda.max_memory_allocated() // (1 << 20)} MB"
        )
        log_str += f", {max_mem_reserved_msg}, {max_mem_allocated_msg}"
    logger.info(log_str)

    # reset time information after printing
    for key in solver.train_time_info:
        solver.train_time_info[key].reset()

    logger.scalar(
        {
            "train/lr": solver.optimizer.get_lr(),
            **{
                f"train/{key}": solver.train_output_info[key].avg
                for key in solver.train_output_info
            },
        },
        step=solver.global_step,
        vdl_writer=solver.vdl_writer,
        wandb_writer=solver.wandb_writer,
        tbd_writer=solver.tbd_writer,
    )


def log_eval_info(
    solver: "solver.Solver",
    batch_size: int,
    epoch_id: Optional[int],
    iters_per_epoch: int,
    iter_id: int,
):
    metric_msg = ", ".join(
        [
            f"{key}: {solver.eval_output_info[key].avg:.5f}"
            for key in solver.eval_output_info
        ]
    )

    time_msg = ", ".join(
        [solver.eval_time_info[key].mean for key in solver.eval_time_info]
    )

    ips_msg = f"ips: {batch_size / solver.eval_time_info['batch_cost'].avg:.2f}"

    eta_sec = (iters_per_epoch - iter_id) * solver.eval_time_info["batch_cost"].avg
    eta_msg = f"eta: {str(datetime.timedelta(seconds=int(eta_sec)))}"

    epoch_width = len(str(solver.epochs))
    iters_width = len(str(iters_per_epoch))
    if isinstance(epoch_id, int):
        logger.info(
            f"[Eval][Epoch {epoch_id:>{epoch_width}}/{solver.epochs}]"
            f"[Iter {iter_id:>{iters_width}}/{iters_per_epoch}] "
            f"{metric_msg}, {time_msg}, {ips_msg}, {eta_msg}"
        )
    else:
        logger.info(
            f"[Eval][Iter {iter_id:>{iters_width}}/{iters_per_epoch}] "
            f"{metric_msg}, {time_msg}, {ips_msg}, {eta_msg}"
        )

    # reset time information after printing
    for key in solver.eval_time_info:
        solver.eval_time_info[key].reset()

    # logger.scalar(
    #     {
    #         f"eval/{key}": solver.eval_output_info[key].avg
    #         for key in solver.eval_output_info
    #     },
    #     step=solver.global_step,
    #     vdl_writer=solver.vdl_writer,
    #     wandb_writer=solver.wandb_writer,
    #     tbd_writer=solver.tbd_writer,
    # )
