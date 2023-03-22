"""Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import random

import numpy as np
import paddle
import paddle.amp as amp
import paddle.distributed as dist
from packaging.version import Version
from paddle.distributed import fleet

import ppsci
from ppsci import gradient
from ppsci.utils import logger
from ppsci.utils import misc
from ppsci.utils import save_load


def train(solver: ppsci.solver.Solver):
    # init hyper-parameters
    epochs = 20000
    solver.cfg = {
        "profiler_options": None,
        "Global": {"epochs": epochs},
        "Arch": {"name": "MLP"},
    }
    solver.iters_per_epoch = 10
    save_freq = 10000
    eval_during_train = True
    eval_freq = 200
    start_eval_epoch = 1

    # init gradient accumulation config
    solver.update_freq = 1

    best_metric = {
        "metric": float("inf"),
        "epoch": 0,
    }

    # init optimizer and lr scheduler
    solver.lr_scheduler = ppsci.optimizer.lr_scheduler.ConstLR(
        epochs, solver.iters_per_epoch, 0.001, by_epoch=False
    )()
    solver.optimizer = ppsci.optimizer.Adam(
        solver.lr_scheduler,
    )([solver.model])

    # load checkpoint for resume training
    checkpoint_path = None
    if checkpoint_path is not None:
        loaded_metric = save_load.load_checkpoint(
            "checkpoint_path", solver.model, solver.optimizer
        )
        if isinstance(loaded_metric, dict):
            best_metric.update(loaded_metric)

    # init constraint(s)
    eq_dataloader_cfg = {
        "dataset": "NamedArrayDataset",
        "sampler": {
            "name": "BatchSampler",
            "batch_size": 980,
            "drop_last": False,
            "shuffle": False,
        },
        "num_workers": 0,
        "seed": 42,
        "use_shared_memory": False,
        "iters_per_epoch": solver.iters_per_epoch,
    }

    # init constraint(s)
    bc_dataloader_cfg = {
        "dataset": "NamedArrayDataset",
        "sampler": {
            "name": "BatchSampler",
            "batch_size": 40,
            "drop_last": False,
            "shuffle": False,
        },
        "num_workers": 0,
        "seed": 42,
        "use_shared_memory": False,
        "iters_per_epoch": solver.iters_per_epoch,
    }

    # maunally build constraint(s)
    def u_compute_func(d):
        x, y = d["x"], d["y"]
        return np.cos(x) * np.cosh(y)

    solver.constraints = {
        "EQ": ppsci.constraint.InteriorConstraint(
            solver.equation["laplace"].equations,
            {"laplace": lambda d: 0.0},
            solver.geom["rect"],
            eq_dataloader_cfg,
            ppsci.loss.MSELoss("sum"),
            evenly=True,
            name="EQ",
        ),
        "BC": ppsci.constraint.BoundaryConstraint(
            {"u": lambda d: d["u"]},
            {"u": u_compute_func},
            solver.geom["rect"],
            bc_dataloader_cfg,
            ppsci.loss.MSELoss("sum"),
            criteria=lambda x, y: np.isclose(x, 0.0)
            | np.isclose(x, 1.0)
            | np.isclose(y, 0.0)
            | np.isclose(y, 1),
            name="BC",
        ),
    }

    # init train output infor object
    solver.train_output_info = {}
    solver.train_time_info = {
        "batch_cost": misc.AverageMeter("batch_cost", ".5f", postfix="s"),
        "reader_cost": misc.AverageMeter("reader_cost", ".5f", postfix="s"),
    }

    # init train func
    solver.train_mode = None
    if solver.train_mode is None:
        solver.train_epoch_func = ppsci.solver.train.train_epoch_func
    else:
        solver.train_epoch_func = ppsci.solver.train.train_LBFGS_epoch_func

    # init distributed environment
    if solver.world_size > 1:
        # TODO(sensen): support different kind of DistributedStrategy
        fleet.init(is_collective=True)
        solver.model = fleet.distributed_model(solver.model)
        solver.optimizer = fleet.distributed_optimizer(solver.optimizer)

    # train epochs
    solver.global_step = 0
    for epoch_id in range(1, epochs + 1):
        solver.train_epoch_func(solver, epoch_id, solver.log_freq)

        # log training summation at end of a epoch
        metric_msg = ", ".join(
            [solver.train_output_info[key].avg_info for key in solver.train_output_info]
        )
        logger.info(f"[Train][Epoch {epoch_id}/{epochs}][Avg] {metric_msg}")
        solver.train_output_info.clear()

        cur_metric = float("inf")
        # evaluate during training
        if (
            eval_during_train
            and epoch_id % eval_freq == 0
            and epoch_id >= start_eval_epoch
        ):
            cur_metric = eval(solver, epoch_id)
            if cur_metric < best_metric["metric"]:
                best_metric["metric"] = cur_metric
                best_metric["epoch"] = epoch_id
                save_load.save_checkpoint(
                    solver.model,
                    solver.optimizer,
                    best_metric,
                    solver.output_dir,
                    solver.model.__class__.__name__,
                    "best_model",
                )
        logger.info(f"[Eval][Epoch {epoch_id}][best metric: {best_metric['metric']}]")
        logger.scaler("eval_metric", cur_metric, epoch_id, solver.vdl_writer)

        # update learning rate by epoch
        if solver.lr_scheduler.by_epoch:
            solver.lr_scheduler.step()

        # save epoch model every `save_freq`
        if save_freq > 0 and epoch_id % save_freq == 0:
            save_load.save_checkpoint(
                solver.model,
                solver.optimizer,
                {"metric": cur_metric, "epoch": epoch_id},
                solver.output_dir,
                solver.model.__class__.__name__,
                f"epoch_{epoch_id}",
            )

        # always save the latest model for convenient resume training
        save_load.save_checkpoint(
            solver.model,
            solver.optimizer,
            {"metric": cur_metric, "epoch": epoch_id},
            solver.output_dir,
            solver.model.__class__.__name__,
            "latest",
        )

    # close VisualDL
    if solver.vdl_writer is not None:
        solver.vdl_writer.close()


def eval(solver: ppsci.solver.Solver, epoch_id):
    """Evaluation"""
    # load pretrained model if specified
    if not hasattr(solver, "cfg"):
        solver.cfg = {
            "profiler_options": None,
            "Global": {"epochs": 0},
            "Arch": {"name": "MLP"},
        }
    solver.global_step = 0
    pretrained_model_path = None
    if pretrained_model_path is not None:
        save_load.load_pretrain(solver.model, pretrained_model_path)

    solver.model.eval()

    # init train func
    solver.eval_func = ppsci.solver.eval.eval_func

    # init validator(s) at the first time
    global_dataloader_cfg = {
        "dataset": "NamedArrayDataset",
        "sampler": {
            "name": "BatchSampler",
            "batch_size": 40,
            "drop_last": False,
            "shuffle": False,
        },
        "num_workers": 0,
        "seed": 42,
        "use_shared_memory": False,
    }

    def u_compute_func(d):
        x, y = d["x"], d["y"]
        return np.cos(x) * np.cosh(y)

    if not hasattr(solver, "validator"):
        solver.validator = {
            "Residual": ppsci.validate.GeometryValidator(
                {"u": lambda d: d["u"]},
                {"u": u_compute_func},
                solver.geom["rect"],
                {**global_dataloader_cfg, **{"total_size": 9800}},
                ppsci.loss.MSELoss("mean"),
                evenly=True,
                metric={"MSE": ppsci.metric.MSE()},
                with_initial=True,
                name="Residual",
            )
        }

    solver.eval_output_info = {}
    solver.eval_time_info = {
        "batch_cost": misc.AverageMeter("batch_cost", ".5f", postfix="s"),
        "reader_cost": misc.AverageMeter("reader_cost", ".5f", postfix="s"),
    }

    result = solver.eval_func(solver, epoch_id, solver.log_freq)
    metric_msg = ", ".join(
        [solver.eval_output_info[key].avg_info for key in solver.eval_output_info]
    )
    logger.info(f"[Eval][Epoch {epoch_id}] {metric_msg}")
    solver.eval_output_info.clear()

    solver.model.train()
    return result


if __name__ == "__main__":

    # initialzie hyper-parameter, geometry, model, equation and AMP settings
    solver = ppsci.solver.Solver()
    solver.mode = "train"
    solver.rank = dist.get_rank()
    solver.vdl_writer = None

    paddle.seed(42 + solver.rank)
    np.random.seed(42 + solver.rank)
    random.seed(42 + solver.rank)

    solver.output_dir = "./output"

    solver.log_freq = 20
    logger.init_logger(log_file=f"./{solver.output_dir}/{solver.mode}.log")

    solver.device = paddle.set_device("gpu")

    version = paddle.__version__
    if Version(version) == Version("0.0.0"):
        version = f"develop({paddle.version.commit[:7]})"
    logger.info(f"Using paddlepaddle {version} on device {solver.device}")

    # manually init geometry(ies)
    solver.geom = {"rect": ppsci.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])}

    # manually init model
    solver.model = ppsci.arch.MLP(["x", "y"], ["u"], 5, 20, "tanh")

    # manually init equation(s)
    def laplace_compute_func(d):
        x, y = d["x"], d["y"]
        u = d["u"]
        laplace = gradient.hessian(u, x) + gradient.hessian(u, y)
        return laplace

    laplace_equation = ppsci.equation.pde.PDE()
    laplace_equation.add_equation("laplace", laplace_compute_func)
    solver.equation = {"laplace": laplace_equation}

    # init AMP
    solver.use_amp = False
    if solver.use_amp:
        solver.amp_level = "O1"
        solver.scaler = amp.GradScaler(True, 2**16)
    else:
        solver.amp_level = "O0"

    solver.world_size = dist.get_world_size()

    # manually start training
    train(solver)

    # manually start evaluation
    # eval(solver, 0)
