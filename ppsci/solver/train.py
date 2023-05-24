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

import time

from paddle.distributed.fleet.utils import hybrid_parallel_util as hpu

from ppsci import solver
from ppsci.solver import printer
from ppsci.utils import expression
from ppsci.utils import misc
from ppsci.utils import profiler


def train_epoch_func(solver: "solver.Solver", epoch_id: int, log_freq: int):
    """Train program for one epoch

    Args:
        solver (solver.Solver): Main solver.
        epoch_id (int): Epoch id.
        log_freq (int): Log training information every `log_freq` steps.
    """
    batch_tic = time.perf_counter()

    for iter_id in range(1, solver.iters_per_epoch + 1):
        total_loss = 0
        loss_dict = misc.Prettydefaultdict(float)
        loss_dict["loss"] = 0.0
        total_batch_size = 0
        reader_cost = 0
        batch_cost = 0
        reader_tic = time.perf_counter()
        with solver.no_sync_context_manager(solver.world_size > 1, solver.model):
            for _, _constraint in solver.constraint.items():
                input_dict, label_dict, weight_dict = next(_constraint.data_iter)

                # profile code below
                # profiler.add_profiler_step(solver.cfg["profiler_options"])
                if iter_id == 5:
                    # 5 step for warmup
                    for key in solver.train_time_info:
                        solver.train_time_info[key].reset()
                reader_cost += time.perf_counter() - reader_tic
                total_batch_size += next(iter(input_dict.values())).shape[0]

                for v in input_dict.values():
                    v.stop_gradient = False
                evaluator = expression.ExpressionSolver(
                    _constraint.input_keys, _constraint.output_keys, solver.model
                )
                for output_name, output_formula in _constraint.output_expr.items():
                    if output_name in label_dict:
                        evaluator.add_target_expr(output_formula, output_name)

                # forward for every constraint
                with solver.autocast_context_manager(solver.use_amp, solver.amp_level):
                    output_dict = evaluator(input_dict)
                    constraint_loss = _constraint.loss(
                        output_dict, label_dict, weight_dict
                    )
                    total_loss += constraint_loss

                loss_dict[_constraint.name] = float(constraint_loss)

                reader_tic = time.perf_counter()

            if solver.update_freq > 1:
                total_loss = total_loss / solver.update_freq
            loss_dict["loss"] = float(total_loss)

            # backward
            if solver.use_amp:
                total_loss_scaled = solver.scaler.scale(total_loss)
                total_loss_scaled.backward()
            else:
                total_loss.backward()

        # update parameters
        if iter_id % solver.update_freq == 0 or iter_id == solver.iters_per_epoch:
            if solver.world_size > 1:
                # fuse + allreduce manually before optimization if use DDP model
                # details in https://github.com/PaddlePaddle/Paddle/issues/48898#issuecomment-1343838622
                hpu.fused_allreduce_gradients(list(solver.model.parameters()), None)
            if solver.use_amp:
                solver.scaler.minimize(solver.optimizer, total_loss_scaled)
            else:
                solver.optimizer.step()
            solver.optimizer.clear_grad()

        # update learning rate by step
        if solver.lr_scheduler is not None and not solver.lr_scheduler.by_epoch:
            solver.lr_scheduler.step()

        batch_cost += time.perf_counter() - batch_tic

        # update and log training information
        solver.global_step += 1
        solver.train_time_info["reader_cost"].update(reader_cost)
        solver.train_time_info["batch_cost"].update(batch_cost)
        printer.update_train_loss(solver, loss_dict, total_batch_size)
        if iter_id == 1 or iter_id % log_freq == 0:
            printer.log_train_info(solver, total_batch_size, epoch_id, iter_id)

        batch_tic = time.perf_counter()


def train_LBFGS_epoch_func(solver: "solver.Solver", epoch_id: int, log_freq: int):
    """Train function for one epoch with L-BFGS optimizer.

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
        reader_cost = 0
        batch_cost = 0
        reader_tic = time.perf_counter()

        input_dicts = []
        label_dicts = []
        weight_dicts = []
        for _, _constraint in solver.constraint.items():
            input_dict, label_dict, weight_dict = next(_constraint.data_iter)
            reader_cost += time.perf_counter() - reader_tic
            for v in input_dict.values():
                v.stop_gradient = False

            # gather all constraint data into list
            input_dicts.append(input_dict)
            label_dicts.append(label_dict)
            weight_dicts.append(weight_dict)
            total_batch_size += next(iter(input_dict.values())).shape[0]
            reader_tic = time.perf_counter()

        def closure():
            """Forward-backward closure function for LBFGS optimizer.

            Returns:
                Tensor: Computed loss.
            """
            total_loss = 0
            with solver.no_sync_context_manager(solver.world_size > 1, solver.model):
                for i, _constraint in enumerate(solver.constraint.values()):
                    evaluator = expression.ExpressionSolver(
                        _constraint.input_keys, _constraint.output_keys, solver.model
                    )
                    for output_name, output_formula in _constraint.output_expr.items():
                        if output_name in label_dict:
                            evaluator.add_target_expr(output_formula, output_name)

                    # forward for every batched data dict
                    with solver.autocast_context_manager(
                        solver.use_amp, solver.amp_level
                    ):
                        output_dict_i = evaluator(input_dicts[i])
                        constraint_loss = _constraint.loss(
                            output_dict_i, label_dicts[i], weight_dicts[i]
                        )
                        total_loss += constraint_loss

                    loss_dict[_constraint.name] = float(constraint_loss)

                solver.optimizer.clear_grad()
                total_loss.backward()
            # fuse + allreduce manually before optimization if use DDP model
            # details in https://github.com/PaddlePaddle/Paddle/issues/48898#issuecomment-1343838622
            if solver.world_size > 1:
                hpu.fused_allreduce_gradients(list(solver.model.parameters()), None)
            loss_dict["loss"] = float(total_loss)

            return total_loss

        solver.optimizer.step(closure)

        # update learning rate by step
        if solver.lr_scheduler is not None and not solver.lr_scheduler.by_epoch:
            solver.lr_scheduler.step()

        batch_cost += time.perf_counter() - batch_tic

        # update and log training information
        solver.global_step += 1
        solver.train_time_info["reader_cost"].update(reader_cost)
        solver.train_time_info["batch_cost"].update(batch_cost)
        printer.update_train_loss(solver, loss_dict, total_batch_size)
        if iter_id == 1 and iter_id % log_freq == 0:
            printer.log_train_info(solver, total_batch_size, epoch_id, iter_id)

        batch_tic = time.perf_counter()
