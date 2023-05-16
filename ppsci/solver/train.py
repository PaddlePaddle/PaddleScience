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

from ppsci.solver import printer
from ppsci.utils import expression
from ppsci.utils import misc
from ppsci.utils import profiler


def train_epoch_func(solver, epoch_id: int, log_freq: int):
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
        total_batch_size = []
        reader_cost = 0
        batch_cost = 0
        reader_tic = time.perf_counter()
        for _, _constraint in solver.constraint.items():
            input_dict, label_dict, weight_dict = next(_constraint.data_iter)

            # profile code below
            # profiler.add_profiler_step(solver.cfg["profiler_options"])
            if iter_id == 5:
                # 5 step for warmup
                for key in solver.train_time_info:
                    solver.train_time_info[key].reset()
            reader_cost += time.perf_counter() - reader_tic
            total_batch_size.append(next(iter(input_dict.values())).shape[0])

            for v in input_dict.values():
                v.stop_gradient = False
            evaluator = expression.ExpressionSolver(
                _constraint.input_keys, _constraint.output_keys, solver.model
            )
            for output_name, output_formula in _constraint.output_expr.items():
                evaluator.add_target_expr(output_formula, output_name)

            # forward for every constraint
            with solver.autocast_context_manager():
                output_dict = evaluator(input_dict)
                constraint_loss = _constraint.loss(output_dict, label_dict, weight_dict)
                total_loss += constraint_loss

            loss_dict[_constraint.name] += float(constraint_loss)

            reader_tic = time.perf_counter()

        if solver.update_freq > 1:
            total_loss = total_loss / solver.update_freq
        loss_dict["loss"] = float(total_loss)
        # backward
        if solver.use_amp:
            total_loss_scaled = solver.scaler.scale(total_loss)
            total_loss_scaled.backward()
            if iter_id % solver.update_freq == 0:
                solver.scaler.minimize(solver.optimizer, total_loss_scaled)
                solver.optimizer.clear_grad()
                if solver.lr_scheduler is not None and not solver.lr_scheduler.by_epoch:
                    solver.lr_scheduler.step()
        else:
            total_loss.backward()
            # import numpy as np
            # # import os
            # bias_u = np.load("./data/net2_params" + f"/bias_{epoch_id - 1}_{iter_id - 1}.npz")
            # bias_v = np.load("./data/net3_params" + f"/bias_{epoch_id - 1}_{iter_id - 1}.npz")
            # bias_p = np.load("./data/net4_params" + f"/bias_{epoch_id - 1}_{iter_id - 1}.npz")
            # weight_u = np.load("./data/net2_params" + f"/weight_{epoch_id - 1}_{iter_id - 1}.npz")
            # weight_v = np.load("./data/net3_params" + f"/weight_{epoch_id - 1}_{iter_id - 1}.npz")
            # weight_p = np.load("./data/net4_params" + f"/weight_{epoch_id - 1}_{iter_id - 1}.npz")

            # weight_grad_u = np.load("./data/net2_params" + f"/weight_grad_{epoch_id - 1}_{iter_id - 1}.npz")
            # weight_grad_v = np.load("./data/net3_params" + f"/weight_grad_{epoch_id - 1}_{iter_id - 1}.npz")
            # weight_grad_p = np.load("./data/net4_params" + f"/weight_grad_{epoch_id - 1}_{iter_id - 1}.npz")
            # # u
            # print("w0_u is close:", np.allclose(solver.model.model_list[0].linears[0].weight.numpy(), weight_u[f"w_0"]))
            # print("w1_u is close:", np.allclose(solver.model.model_list[0].linears[1].weight.numpy(), weight_u[f"w_1"]))
            # print("w2_u is close:", np.allclose(solver.model.model_list[0].linears[2].weight.numpy(), weight_u[f"w_2"]))
            # print("w3_u is close:", np.allclose(solver.model.model_list[0].last_fc.weight.numpy(), weight_u[f"w_3"]))

            # print("b0_u is close:", np.allclose(solver.model.model_list[0].linears[0].bias.numpy(), bias_u[f"b_0"]))
            # print("b1_u is close:", np.allclose(solver.model.model_list[0].linears[1].bias.numpy(), bias_u[f"b_1"]))
            # print("b2_u is close:", np.allclose(solver.model.model_list[0].linears[2].bias.numpy(), bias_u[f"b_2"]))
            # print("b3_u is close:", np.allclose(solver.model.model_list[0].last_fc.bias.numpy(), bias_u[f"b_3"]))

            # print("w0_u grad is close:", np.allclose(solver.model.model_list[0].linears[0].weight.grad.numpy(), weight_grad_u[f"w_0"]))
            # print("w1_u grad is close:", np.allclose(solver.model.model_list[0].linears[1].weight.grad.numpy(), weight_grad_u[f"w_1"]))
            # print("w2_u grad is close:", np.allclose(solver.model.model_list[0].linears[2].weight.grad.numpy(), weight_grad_u[f"w_2"]))
            # print("w3_u grad is close:", np.allclose(solver.model.model_list[0].last_fc.weight.grad.numpy(), weight_grad_u[f"w_3"]))

            # #v
            # print("w0_v is close:", np.allclose(solver.model.model_list[1].linears[0].weight.numpy(), weight_v[f"w_0"]))
            # print("w1_v is close:", np.allclose(solver.model.model_list[1].linears[1].weight.numpy(), weight_v[f"w_1"]))
            # print("w2_v is close:", np.allclose(solver.model.model_list[1].linears[2].weight.numpy(), weight_v[f"w_2"]))
            # print("w3_v is close:", np.allclose(solver.model.model_list[1].last_fc.weight.numpy(), weight_v[f"w_3"]))

            # print("b0_v is close:", np.allclose(solver.model.model_list[1].linears[0].bias.numpy(), bias_v[f"b_0"]))
            # print("b1_v is close:", np.allclose(solver.model.model_list[1].linears[1].bias.numpy(), bias_v[f"b_1"]))
            # print("b2_v is close:", np.allclose(solver.model.model_list[1].linears[2].bias.numpy(), bias_v[f"b_2"]))
            # print("b3_v is close:", np.allclose(solver.model.model_list[1].last_fc.bias.numpy(), bias_v[f"b_3"]))

            # print("w0_v grad is close:", np.allclose(solver.model.model_list[1].linears[0].weight.grad.numpy(), weight_grad_v[f"w_0"]))
            # print("w1_v grad is close:", np.allclose(solver.model.model_list[1].linears[1].weight.grad.numpy(), weight_grad_v[f"w_1"]))
            # print("w2_v grad is close:", np.allclose(solver.model.model_list[1].linears[2].weight.grad.numpy(), weight_grad_v[f"w_2"]))
            # print("w3_v grad is close:", np.allclose(solver.model.model_list[1].last_fc.weight.grad.numpy(), weight_grad_v[f"w_3"]))

            # #p
            # print("w0_p is close:", np.allclose(solver.model.model_list[2].linears[0].weight.numpy(), weight_p[f"w_0"]))
            # print("w1_p is close:", np.allclose(solver.model.model_list[2].linears[1].weight.numpy(), weight_p[f"w_1"]))
            # print("w2_p is close:", np.allclose(solver.model.model_list[2].linears[2].weight.numpy(), weight_p[f"w_2"]))
            # print("w3_p is close:", np.allclose(solver.model.model_list[2].last_fc.weight.numpy(), weight_p[f"w_3"]))

            # print("b0_p is close:", np.allclose(solver.model.model_list[2].linears[0].bias.numpy(), bias_p[f"b_0"]))
            # print("b1_p is close:", np.allclose(solver.model.model_list[2].linears[1].bias.numpy(), bias_p[f"b_1"]))
            # print("b2_p is close:", np.allclose(solver.model.model_list[2].linears[2].bias.numpy(), bias_p[f"b_2"]))
            # print("b3_p is close:", np.allclose(solver.model.model_list[2].last_fc.bias.numpy(), bias_p[f"b_3"]))

            # print("w0_p grad is close:", np.allclose(solver.model.model_list[2].linears[0].weight.grad.numpy(), weight_grad_p[f"w_0"]))
            # print("w1_p grad is close:", np.allclose(solver.model.model_list[2].linears[1].weight.grad.numpy(), weight_grad_p[f"w_1"]))
            # print("w2_p grad is close:", np.allclose(solver.model.model_list[2].linears[2].weight.grad.numpy(), weight_grad_p[f"w_2"]))
            # print("w3_p grad is close:", np.allclose(solver.model.model_list[2].last_fc.weight.grad.numpy(), weight_grad_p[f"w_3"]))

            if iter_id % solver.update_freq == 0:
                solver.optimizer.step()
                solver.optimizer.clear_grad()
                if solver.lr_scheduler is not None and not solver.lr_scheduler.by_epoch:
                    solver.lr_scheduler.step()

        batch_cost += time.perf_counter() - batch_tic

        # update and log training information
        solver.global_step += 1
        total_batch_size = sum(total_batch_size)
        solver.train_time_info["reader_cost"].update(reader_cost)
        solver.train_time_info["batch_cost"].update(batch_cost)
        printer.update_train_loss(solver, loss_dict, total_batch_size)
        if iter_id == 1 or iter_id % log_freq == 0:
            printer.log_train_info(solver, total_batch_size, epoch_id, iter_id)

        # real_loss = float(loss_dict["loss"])
        # print(f"{(epoch_id - 1) * 195 + iter_id} step: loss is {real_loss}")
        batch_tic = time.perf_counter()


def train_LBFGS_epoch_func(solver, epoch_id: int, log_freq: int):
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
        total_batch_size = []
        reader_cost = 0
        batch_cost = 0
        reader_tic = time.perf_counter()

        input_dict_list = []
        label_dict_list = []
        weight_dict_list = []
        for _, _constraint in solver.constraint.items():
            input_dict, label_dict, weight_dict = next(_constraint.data_iter)
            reader_cost += time.perf_counter() - reader_tic
            for v in input_dict.values():
                v.stop_gradient = False
            input_dict_list.append(input_dict)
            label_dict_list.append(label_dict)
            weight_dict_list.append(weight_dict)
            total_batch_size.append(next(iter(input_dict.values())).shape[0])

        def closure():
            """Forward-backward closure function for LBFGS optimizer.

            Returns:
                Tensor: Computed loss.
            """
            total_loss = 0
            for i, _constraint in enumerate(solver.constraint.values()):
                evaluator = expression.ExpressionSolver(
                    _constraint.input_keys, _constraint.output_keys, solver.model
                )
                for output_name, output_formula in _constraint.output_expr.items():
                    evaluator.add_target_expr(output_formula, output_name)

                # forward for every batched data dict
                with solver.autocast_context_manager():
                    output_dict_i = evaluator(input_dict_list[i])
                    constraint_loss = _constraint.loss(
                        output_dict_i, label_dict_list[i], weight_dict_list[i]
                    )
                    total_loss += constraint_loss

                loss_dict[_constraint.name] += float(constraint_loss)

            total_loss.backward()
            loss_dict["loss"] = float(total_loss)

            return total_loss

        reader_tic = time.perf_counter()

        solver.optimizer.step(closure)
        if solver.lr_scheduler is not None and not solver.lr_scheduler.by_epoch:
            solver.lr_scheduler.step()

        batch_cost += time.perf_counter() - batch_tic

        # update and log training information
        solver.global_step += 1
        total_batch_size = sum(total_batch_size)
        solver.train_time_info["reader_cost"].update(reader_cost)
        solver.train_time_info["batch_cost"].update(batch_cost)
        printer.update_train_loss(solver, loss_dict, total_batch_size)
        if iter_id == 1 and iter_id % log_freq == 0:
            printer.log_train_info(solver, total_batch_size, epoch_id, iter_id)

        batch_tic = time.perf_counter()
