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

import sys
import time
from typing import TYPE_CHECKING

import paddle
from paddle.distributed.fleet.utils import hybrid_parallel_util as hpu
from paddle.framework import core

from ppsci.solver import printer
from ppsci.utils import misc

if TYPE_CHECKING:
    from ppsci import solver


def train_epoch_func(solver: "solver.Solver", epoch_id: int, log_freq: int):
    """Train program for one epoch.

    Args:
        solver (solver.Solver): Main solver.
        epoch_id (int): Epoch id.
        log_freq (int): Log training information every `log_freq` steps.
    """
    batch_tic = time.perf_counter()

    for iter_id in range(1, solver.iters_per_epoch + 1):
        if solver.nvtx_flag:  # only for nsight analysis
            core.nvprof_nvtx_push(
                f"Training iteration {solver.global_step + 1}"
            )  # Training iteration

        total_batch_size = 0
        reader_cost = 0.0
        batch_cost = 0.0
        reader_tic = time.perf_counter()

        input_dicts = []
        label_dicts = []
        weight_dicts = []
        for _, _constraint in solver.constraint.items():
            # fetch data from data loader
            try:
                input_dict, label_dict, weight_dict = next(_constraint.data_iter)
            except StopIteration:
                _constraint.data_iter = iter(_constraint.data_loader)
                input_dict, label_dict, weight_dict = next(_constraint.data_iter)
            reader_cost += time.perf_counter() - reader_tic

            # NOTE: eliminate first 5 step for warmup
            if iter_id == 5:
                for key in solver.train_time_info:
                    solver.train_time_info[key].reset()

            for v in input_dict.values():
                if hasattr(v, "stop_gradient"):
                    v.stop_gradient = False

            # gather each constraint's input, label, weight to a list
            input_dicts.append(input_dict)
            label_dicts.append(label_dict)
            weight_dicts.append(weight_dict)
            total_batch_size += next(iter(input_dict.values())).shape[0]
            reader_tic = time.perf_counter()

        loss_dict = misc.Prettydefaultdict(float)
        loss_dict["loss"] = 0.0
        # forward for every constraint, including model and equation expression
        with solver.no_sync_context_manager(solver.world_size > 1, solver.model):
            with solver.autocast_context_manager(solver.use_amp, solver.amp_level):
                if solver.nvtx_flag:  # only for nsight analysis
                    core.nvprof_nvtx_push("Loss computation")

                constraint_losses = solver.forward_helper.train_forward(
                    tuple(
                        _constraint.output_expr
                        for _constraint in solver.constraint.values()
                    ),
                    input_dicts,
                    solver.model,
                    solver.constraint,
                    label_dicts,
                    weight_dicts,
                )

                if solver.nvtx_flag:  # only for nsight analysis
                    core.nvprof_nvtx_pop()  # Loss computation

                # accumulate all losses
                if solver.nvtx_flag:  # only for nsight analysis
                    core.nvprof_nvtx_push("Loss aggregator")

                total_loss = solver.loss_aggregator(
                    constraint_losses, solver.global_step
                )
                if solver.update_freq > 1:
                    total_loss = total_loss / solver.update_freq

                for i, _constraint in enumerate(solver.constraint.values()):
                    loss_dict[_constraint.name] = (
                        float(constraint_losses[i]) / solver.update_freq
                    )
                loss_dict["loss"] = float(total_loss)

                if solver.nvtx_flag:  # only for nsight analysis
                    core.nvprof_nvtx_pop()  # Loss aggregator

            # backward
            if solver.nvtx_flag:  # only for nsight analysis
                core.nvprof_nvtx_push("Loss backward")

            if solver.use_amp:
                total_loss_scaled = solver.scaler.scale(total_loss)
                total_loss_scaled.backward()
            else:
                total_loss.backward()

            if solver.nvtx_flag:  # only for nsight analysis
                core.nvprof_nvtx_pop()  # Loss backward

        # update parameters
        if iter_id % solver.update_freq == 0 or iter_id == solver.iters_per_epoch:
            if solver.nvtx_flag:  # only for nsight analysis
                core.nvprof_nvtx_push("Optimizer update")

            if solver.world_size > 1:
                # fuse + allreduce manually before optimization if use DDP + no_sync
                # details in https://github.com/PaddlePaddle/Paddle/issues/48898#issuecomment-1343838622
                hpu.fused_allreduce_gradients(list(solver.model.parameters()), None)
            if solver.use_amp:
                solver.scaler.minimize(solver.optimizer, total_loss_scaled)
            else:
                solver.optimizer.step()

            if solver.nvtx_flag:  # only for nsight analysis
                core.nvprof_nvtx_pop()  # Optimizer update

            solver.optimizer.clear_grad()

        # update learning rate by step
        if solver.lr_scheduler is not None and not solver.lr_scheduler.by_epoch:
            solver.lr_scheduler.step()

        if solver.benchmark_flag:
            paddle.device.synchronize()
        batch_cost += time.perf_counter() - batch_tic

        # update and log training information
        solver.global_step += 1
        solver.train_time_info["reader_cost"].update(reader_cost)
        solver.train_time_info["batch_cost"].update(batch_cost)
        printer.update_train_loss(solver, loss_dict, total_batch_size)
        if solver.global_step % log_freq == 0 or solver.global_step == 1:
            printer.log_train_info(solver, total_batch_size, epoch_id, iter_id)

        batch_tic = time.perf_counter()

        if solver.nvtx_flag:  # only for nsight analysis
            core.nvprof_nvtx_pop()  # Training iteration
            NVTX_STOP_ITER = 25
            if solver.global_step >= NVTX_STOP_ITER:
                print(
                    f"Only run {NVTX_STOP_ITER} steps when 'NVTX' is set in environment"
                    " for nsight analysis. Exit now ......\n"
                )
                core.nvprof_stop()
                sys.exit(0)


def train_LBFGS_epoch_func(solver: "solver.Solver", epoch_id: int, log_freq: int):
    """Train function for one epoch with L-BFGS optimizer.

    NOTE: L-BFGS training program do not support AMP now.

    Args:
        solver (solver.Solver): Main solver.
        epoch_id (int): Epoch id.
        log_freq (int): Log training information every `log_freq` steps.
    """
    batch_tic = time.perf_counter()

    for iter_id in range(1, solver.iters_per_epoch + 1):
        loss_dict = misc.Prettydefaultdict(float)
        loss_dict["loss"] = 0.0
        total_batch_size = 0
        reader_cost = 0.0
        batch_cost = 0.0
        reader_tic = time.perf_counter()

        input_dicts = []
        label_dicts = []
        weight_dicts = []
        for _, _constraint in solver.constraint.items():
            # fetch data from data loader
            try:
                input_dict, label_dict, weight_dict = next(_constraint.data_iter)
            except StopIteration:
                _constraint.data_iter = iter(_constraint.data_loader)
                input_dict, label_dict, weight_dict = next(_constraint.data_iter)
            reader_cost += time.perf_counter() - reader_tic

            for v in input_dict.values():
                if hasattr(v, "stop_gradient"):
                    v.stop_gradient = False

            # gather each constraint's input, label, weight to a list
            input_dicts.append(input_dict)
            label_dicts.append(label_dict)
            weight_dicts.append(weight_dict)
            total_batch_size += next(iter(input_dict.values())).shape[0]
            reader_tic = time.perf_counter()

        def closure() -> paddle.Tensor:
            """Forward-backward closure function for LBFGS optimizer.

            Returns:
                paddle.Tensor: Computed loss scalar.
            """
            with solver.no_sync_context_manager(solver.world_size > 1, solver.model):
                with solver.autocast_context_manager(solver.use_amp, solver.amp_level):
                    # forward for every constraint, including model and equation expression
                    constraint_losses = solver.forward_helper.train_forward(
                        tuple(
                            _constraint.output_expr
                            for _constraint in solver.constraint.values()
                        ),
                        input_dicts,
                        solver.model,
                        solver.constraint,
                        label_dicts,
                        weight_dicts,
                    )

                    total_loss = solver.loss_aggregator(
                        constraint_losses, solver.global_step
                    )
                    # accumulate all losses
                    for i, _constraint in enumerate(solver.constraint.values()):
                        loss_dict[_constraint.name] = float(constraint_losses[i])
                    loss_dict["loss"] = float(total_loss)

                # backward
                solver.optimizer.clear_grad()
                total_loss.backward()

            if solver.world_size > 1:
                # fuse + allreduce manually before optimization if use DDP model
                # details in https://github.com/PaddlePaddle/Paddle/issues/48898#issuecomment-1343838622
                hpu.fused_allreduce_gradients(list(solver.model.parameters()), None)

            return total_loss

        # update parameters
        solver.optimizer.step(closure)

        # update learning rate by step
        if solver.lr_scheduler is not None and not solver.lr_scheduler.by_epoch:
            solver.lr_scheduler.step()

        if solver.benchmark_flag:
            paddle.device.synchronize()
        batch_cost += time.perf_counter() - batch_tic

        # update and log training information
        solver.global_step += 1
        solver.train_time_info["reader_cost"].update(reader_cost)
        solver.train_time_info["batch_cost"].update(batch_cost)
        printer.update_train_loss(solver, loss_dict, total_batch_size)
        if solver.global_step % log_freq == 0 or solver.global_step == 1:
            printer.log_train_info(solver, total_batch_size, epoch_id, iter_id)

        batch_tic = time.perf_counter()
