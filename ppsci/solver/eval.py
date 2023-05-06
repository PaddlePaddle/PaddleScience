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

import paddle
import paddle.amp as amp
import paddle.io as io

from ppsci import solver
from ppsci.solver import printer
from ppsci.utils import expression
from ppsci.utils import misc
from ppsci.utils import profiler


def eval_func(solver: solver.Solver, epoch_id: int, log_freq: int) -> float:
    """Evaluation program

    Args:
        solver (Solver): Main Solver.
        epoch_id (int): Epoch id.
        log_freq (int): Log evaluation information every `log_freq` steps.

    Returns:
        float: Target metric computed during evaluation.
    """
    target_metric: float = None
    for _, _validator in solver.validator.items():
        all_input = misc.Prettydefaultdict(list)
        all_output = misc.Prettydefaultdict(list)
        all_label = misc.Prettydefaultdict(list)
        if isinstance(_validator.data_loader, io.DataLoader):
            num_samples = len(_validator.data_loader.dataset)
        else:
            num_samples = _validator.data_loader.num_samples

        loss_dict = misc.Prettydefaultdict(float)
        reader_tic = time.perf_counter()
        batch_tic = time.perf_counter()
        for iter_id, batch in enumerate(_validator.data_loader, start=1):
            input_dict, label_dict, weight_dict = batch
            # profile code
            # profiler.add_profiler_step(solver.cfg["profiler_options"])
            if iter_id == 5:
                # 5 step for warmup
                for key in solver.eval_time_info:
                    solver.eval_time_info[key].reset()
            reader_cost = time.perf_counter() - reader_tic

            for v in input_dict.values():
                v.stop_gradient = False
            evaluator = expression.ExpressionSolver(
                _validator.input_keys, _validator.output_keys, solver.model
            )
            for output_name, output_formula in _validator.output_expr.items():
                evaluator.add_target_expr(output_formula, output_name)

            # forward
            if solver.use_amp:
                with amp.auto_cast(level=solver.amp_level):
                    output_dict = evaluator(input_dict)
                    validator_loss = _validator.loss(
                        output_dict, label_dict, weight_dict
                    )
                    loss_dict[f"loss({_validator.name})"] = float(validator_loss)
            else:
                output_dict = evaluator(input_dict)
                validator_loss = _validator.loss(output_dict, label_dict, weight_dict)
                loss_dict[f"loss({_validator.name})"] = float(validator_loss)

            # collect batch data
            for key, input in input_dict.items():
                all_input[key].append(
                    input.detach()
                    if solver.world_size == 1
                    else misc.all_gather(input.detach())
                )
            for key, output in output_dict.items():
                all_output[key].append(
                    output.detach()
                    if solver.world_size == 1
                    else misc.all_gather(output.detach())
                )
            for key, label in label_dict.items():
                all_label[key].append(
                    label.detach()
                    if solver.world_size == 1
                    else misc.all_gather(label.detach())
                )

            batch_cost = time.perf_counter() - batch_tic
            solver.eval_time_info["reader_cost"].update(reader_cost)
            solver.eval_time_info["batch_cost"].update(batch_cost)
            total_batch_size = sum([v.shape[0] for v in input_dict.values()])
            printer.update_eval_loss(solver, loss_dict, total_batch_size)
            if iter_id == 1 or iter_id % log_freq == 0:
                printer.log_eval_info(
                    solver,
                    total_batch_size,
                    epoch_id,
                    len(_validator.data_loader),
                    iter_id,
                )

            reader_tic = time.perf_counter()
            batch_tic = time.perf_counter()

        # gather all data
        for key in all_input:
            all_input[key] = paddle.concat(all_input[key])
            if len(all_input[key]) > num_samples:
                all_input[key] = all_input[key][:num_samples]
        for key in all_output:
            all_output[key] = paddle.concat(all_output[key])
            if len(all_output[key]) > num_samples:
                all_output[key] = all_output[key][:num_samples]
        for key in all_label:
            all_label[key] = paddle.concat(all_label[key])
            if len(all_label[key]) > num_samples:
                all_label[key] = all_label[key][:num_samples]

        metric = misc.PrettyOrderedDict()
        for metric_name, metric_func in _validator.metric.items():
            metric_dict = metric_func(all_output, all_label)
            metric[metric_name] = metric_dict
            for var_name, metric_value in metric_dict.items():
                metric_str = f"{metric_name}.{var_name}({_validator.name})"
                if metric_str not in solver.eval_output_info:
                    solver.eval_output_info[metric_str] = misc.AverageMeter(
                        metric_str, ".5f"
                    )
                solver.eval_output_info[metric_str].update(metric_value, num_samples)

        if target_metric is None:
            tmp = metric
            while isinstance(tmp, dict):
                tmp = next(iter(tmp.values()))
            assert isinstance(
                tmp, (int, float)
            ), f"Target metric({type(tmp)}) should be a number"
            target_metric = tmp

    return target_metric
