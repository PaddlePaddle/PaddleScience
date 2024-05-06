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

import time
from typing import TYPE_CHECKING
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import paddle
from paddle import io

from ppsci.solver import printer
from ppsci.utils import misc

if TYPE_CHECKING:
    from pgl.utils import data as pgl_data

    from ppsci import solver


def _get_dataset_length(
    data_loader: Union["io.DataLoader", "pgl_data.Dataloader", "io.IterableDataset"]
) -> int:
    """Get full dataset length of given dataloader.

    Args:
        data_loader (Union[io.DataLoader, pgl_data.Dataloader, io.IterableDataset]):
            Given dataloader.

    Returns:
        int: Length of full dataset.
    """
    if isinstance(data_loader, io.DataLoader):
        num_samples = len(data_loader.dataset)
    elif isinstance(data_loader, io.IterableDataset):
        num_samples = data_loader.num_samples
    elif str(type(data_loader)) == "<class 'pgl.utils.data.dataloader.Dataloader'>":
        num_samples = len(data_loader.dataset)
    else:
        raise NotImplementedError(
            f"Can not fetch the length of given dataset({type(data_loader)})."
        )

    return num_samples


def _eval_by_dataset(
    solver: "solver.Solver", epoch_id: Optional[int], log_freq: int
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """Evaluate with computing metric on total samples(default process).

    NOTE: This is the default evaluation method as general for most cases, but may not
    memory-efficiency for large dataset or large output.

    Args:
        solver (solver.Solver): Main Solver.
        epoch_id (Optional[int]): Epoch id.
        log_freq (int): Log evaluation information every `log_freq` steps.

    Returns:
        Tuple[float, Dict[str, Dict[str, float]]]: Target metric and all metric dicts
            computed during evaluation.
    """
    target_metric: float = float("inf")
    for _, _validator in solver.validator.items():
        all_output = misc.Prettydefaultdict(list)
        all_label = misc.Prettydefaultdict(list)
        num_samples = _get_dataset_length(_validator.data_loader)

        loss_dict = misc.Prettydefaultdict(float)
        reader_tic = time.perf_counter()
        batch_tic = time.perf_counter()
        for iter_id, batch in enumerate(_validator.data_loader, start=1):
            input_dict, label_dict, weight_dict = batch
            reader_cost = time.perf_counter() - reader_tic

            # NOTE: eliminate first 5 step for warmup
            if iter_id == 5:
                for key in solver.eval_time_info:
                    solver.eval_time_info[key].reset()

            for v in input_dict.values():
                if hasattr(v, "stop_gradient"):
                    v.stop_gradient = False

            # forward
            with solver.autocast_context_manager(
                solver.use_amp, solver.amp_level
            ), solver.no_grad_context_manager(solver.eval_with_no_grad):
                output_dict, validator_loss = solver.forward_helper.eval_forward(
                    _validator.output_expr,
                    input_dict,
                    solver.model,
                    _validator,
                    label_dict,
                    weight_dict,
                )

            loss_dict[f"{_validator.name}/loss"] = float(validator_loss)

            for key, output in output_dict.items():
                all_output[key].append(
                    (output.detach() if hasattr(output, "detach") else output)
                    if solver.world_size == 1
                    else misc.all_gather(output.detach())
                )

            for key, label in label_dict.items():
                all_label[key].append(
                    (label.detach() if hasattr(label, "detach") else label)
                    if solver.world_size == 1
                    else misc.all_gather(label.detach())
                )

            batch_cost = time.perf_counter() - batch_tic
            solver.eval_time_info["reader_cost"].update(reader_cost)
            solver.eval_time_info["batch_cost"].update(batch_cost)
            batch_size = next(iter(input_dict.values())).shape[0]
            printer.update_eval_loss(solver, loss_dict, batch_size)
            if (
                iter_id == 1
                or iter_id % log_freq == 0
                or iter_id == len(_validator.data_loader)
            ):
                printer.log_eval_info(
                    solver,
                    batch_size,
                    epoch_id,
                    len(_validator.data_loader),
                    iter_id,
                )

            reader_tic = time.perf_counter()
            batch_tic = time.perf_counter()

        # concatenate all data and discard padded sample(s)
        for key in all_output:
            if paddle.is_tensor(all_output[key][0]):
                all_output[key] = paddle.concat(all_output[key])
            if len(all_output[key]) > num_samples:
                all_output[key] = all_output[key][:num_samples]

        for key in all_label:
            if paddle.is_tensor(all_label[key][0]):
                all_label[key] = paddle.concat(all_label[key])
            if len(all_label[key]) > num_samples:
                all_label[key] = all_label[key][:num_samples]

        metric_dict_group: Dict[str, Dict[str, float]] = misc.PrettyOrderedDict()
        for metric_name, metric_func in _validator.metric.items():
            # NOTE: compute metric with entire output and label
            metric_dict = metric_func(all_output, all_label)
            metric_dict_group[metric_name] = {
                k: float(v) for k, v in metric_dict.items()
            }
            for var_name, metric_value in metric_dict.items():
                metric_str = f"{_validator.name}/{metric_name}.{var_name}"
                if metric_str not in solver.eval_output_info:
                    solver.eval_output_info[metric_str] = misc.AverageMeter(
                        metric_str, ".5f"
                    )
                solver.eval_output_info[metric_str].update(
                    float(metric_value), num_samples
                )

        # use the first metric for return value
        tmp = metric_dict_group
        while isinstance(tmp, dict):
            tmp = next(iter(tmp.values()))
        # avoid that none of metric is set
        if isinstance(tmp, float):
            target_metric = float(tmp)

    return target_metric, metric_dict_group


def _eval_by_batch(
    solver: "solver.Solver", epoch_id: Optional[int], log_freq: int
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """Evaluate with computing metric by batch, which is memory-efficient.

    NOTE: This is a evaluation function for large dataset or large output, as is more
    memory-efficiency than evaluating by dataset, but less general because some metric
    is not independent among samples, e.g. L2 relative error.

    Args:
        solver (solver.Solver): Main Solver.
        epoch_id (Optional[int]): Epoch id.
        log_freq (int): Log evaluation information every `log_freq` steps.

    Returns:
        Tuple[float, Dict[str, Dict[str, float]]]: Target metric and all metric dicts
            computed during evaluation.
    """
    target_metric: float = float("inf")
    for _, _validator in solver.validator.items():
        num_samples = _get_dataset_length(_validator.data_loader)

        loss_dict = misc.Prettydefaultdict(float)
        metric_dict_group: Dict[str, Dict[str, float]] = misc.PrettyOrderedDict()
        reader_tic = time.perf_counter()
        batch_tic = time.perf_counter()
        for iter_id, batch in enumerate(_validator.data_loader, start=1):
            input_dict, label_dict, weight_dict = batch
            reader_cost = time.perf_counter() - reader_tic

            # NOTE: eliminate first 5 step for warmup
            if iter_id == 5:
                for key in solver.eval_time_info:
                    solver.eval_time_info[key].reset()

            batch_size = next(iter(input_dict.values())).shape[0]
            for v in input_dict.values():
                if hasattr(v, "stop_gradient"):
                    v.stop_gradient = False

            # forward
            with solver.autocast_context_manager(
                solver.use_amp, solver.amp_level
            ), solver.no_grad_context_manager(solver.eval_with_no_grad):
                output_dict, validator_loss = solver.forward_helper.eval_forward(
                    _validator.output_expr,
                    input_dict,
                    solver.model,
                    _validator,
                    label_dict,
                    weight_dict,
                )

            loss_dict[f"{_validator.name}/loss"] = float(validator_loss)

            # collect batch metric
            for metric_name, metric_func in _validator.metric.items():
                metric_dict = metric_func(output_dict, label_dict)
                if metric_name not in metric_dict_group:
                    metric_dict_group[metric_name] = misc.Prettydefaultdict(list)
                for var_name, metric_value in metric_dict.items():
                    metric_dict_group[metric_name][var_name].append(
                        metric_value
                        if solver.world_size == 1
                        else misc.all_gather(metric_value)
                    )

            batch_cost = time.perf_counter() - batch_tic
            solver.eval_time_info["reader_cost"].update(reader_cost)
            solver.eval_time_info["batch_cost"].update(batch_cost)
            printer.update_eval_loss(solver, loss_dict, batch_size)
            if (
                iter_id == 1
                or iter_id % log_freq == 0
                or iter_id == len(_validator.data_loader)
            ):
                printer.log_eval_info(
                    solver,
                    batch_size,
                    epoch_id,
                    len(_validator.data_loader),
                    iter_id,
                )

            reader_tic = time.perf_counter()
            batch_tic = time.perf_counter()

        # concatenate all metric and discard metric of padded sample(s)
        for metric_name, metric_dict in metric_dict_group.items():
            for var_name, metric_value in metric_dict.items():
                # NOTE: concat all metric(scalars) into metric vector
                metric_value = paddle.concat(metric_value)[:num_samples]
                # NOTE: compute metric via averaging metric vector,
                # this might be not general for certain evaluation case
                metric_value = float(metric_value.mean())
                metric_dict_group[metric_name][var_name] = metric_value
                metric_str = f"{_validator.name}/{metric_name}.{var_name}"
                if metric_str not in solver.eval_output_info:
                    solver.eval_output_info[metric_str] = misc.AverageMeter(
                        metric_str, ".5f"
                    )
                solver.eval_output_info[metric_str].update(metric_value, num_samples)

        # use the first metric for return value
        tmp = metric_dict_group
        while isinstance(tmp, dict):
            tmp = next(iter(tmp.values()))
        # avoid that none of metric is set
        if isinstance(tmp, float):
            target_metric = tmp

    return target_metric, metric_dict_group


def eval_func(
    solver: "solver.Solver", epoch_id: Optional[int], log_freq: int
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """Evaluation function.

    Args:
        solver (solver.Solver): Main Solver.
        epoch_id (Optional[int]): Epoch id.
        log_freq (int): Log evaluation information every `log_freq` steps.

    Returns:
        Tuple[float, Dict[str, Dict[str, float]]]: Target metric and all metric dicts
            computed during evaluation.
    """
    if solver.compute_metric_by_batch:
        return _eval_by_batch(solver, epoch_id, log_freq)
    return _eval_by_dataset(solver, epoch_id, log_freq)
