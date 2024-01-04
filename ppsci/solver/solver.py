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

import contextlib
import itertools
import os
import sys
from os import path as osp
from typing import Callable
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import paddle.distributed as dist
import sympy as sp
import visualdl as vdl
from packaging import version
from paddle import amp
from paddle import jit
from paddle import nn
from paddle import optimizer as optim
from paddle.distributed import fleet
from typing_extensions import Literal

import ppsci
from ppsci.loss import mtl
from ppsci.utils import expression
from ppsci.utils import logger
from ppsci.utils import misc
from ppsci.utils import save_load


class Solver:
    """Class for solver.

    Args:
        model (nn.Layer): Model.
        constraint (Optional[Dict[str, ppsci.constraint.Constraint]]): Constraint(s) applied on model. Defaults to None.
        output_dir (Optional[str]): Output directory. Defaults to "./output/".
        optimizer (Optional[optimizer.Optimizer]): Optimizer object. Defaults to None.
        lr_scheduler (Optional[optimizer.lr.LRScheduler]): Learning rate scheduler. Defaults to None.
        epochs (int, optional): Training epoch(s). Defaults to 5.
        iters_per_epoch (int, optional): Number of iterations within an epoch. Defaults to 20.
        update_freq (int, optional): Update frequency of parameters. Defaults to 1.
        save_freq (int, optional): Saving frequency for checkpoint. Defaults to 0.
        log_freq (int, optional): Logging frequency. Defaults to 10.
        eval_during_train (bool, optional): Whether evaluate model during training. Defaults to False.
        start_eval_epoch (int, optional): Epoch number evaluation applied begin after. Defaults to 1.
        eval_freq (int, optional): Evaluation frequency. Defaults to 1.
        seed (int, optional): Random seed. Defaults to 42.
        use_vdl (Optional[bool]): Whether use VisualDL to log scalars. Defaults to False.
        use_wandb (Optional[bool]): Whether use wandb to log data. Defaults to False.
        wandb_config (Optional[Dict[str, str]]): Config dict of WandB. Defaults to None.
        device (Literal["cpu", "gpu", "xpu"], optional): Runtime device. Defaults to "gpu".
        equation (Optional[Dict[str, ppsci.equation.PDE]]): Equation dict. Defaults to None.
        geom (Optional[Dict[str, ppsci.geometry.Geometry]]): Geometry dict. Defaults to None.
        validator (Optional[Dict[str, ppsci.validate.Validator]]): Validator dict. Defaults to None.
        visualizer (Optional[Dict[str, ppsci.visualize.Visualizer]]): Visualizer dict. Defaults to None.
        use_amp (bool, optional): Whether use AMP. Defaults to False.
        amp_level (Literal["O1", "O2", "O0"], optional): AMP level. Defaults to "O0".
        pretrained_model_path (Optional[str]): Pretrained model path. Defaults to None.
        checkpoint_path (Optional[str]): Checkpoint path. Defaults to None.
        compute_metric_by_batch (bool, optional): Whether calculate metrics after each batch during evaluation. Defaults to False.
        eval_with_no_grad (bool, optional): Whether set `stop_gradient=True` for every Tensor if no differentiation
            involved during computation, generally for save GPU memory and accelerate computing. Defaults to False.
        to_static (bool, optional): Whether enable to_static for forward pass. Defaults to False.
        loss_aggregator (Optional[mtl.LossAggregator]): Loss aggregator, such as a multi-task learning loss aggregator. Defaults to None.

    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.MLP(("x",), ("u",), 5, 20)
        >>> opt = ppsci.optimizer.AdamW(1e-3)((model,))
        >>> geom = ppsci.geometry.Rectangle((0, 0), (1, 1))
        >>> pde_constraint = ppsci.constraint.InteriorConstraint(
        ...     {"u": lambda out: out["u"]},
        ...     {"u": 0},
        ...     geom,
        ...     {
        ...         "dataset": "IterableNamedArrayDataset",
        ...         "iters_per_epoch": 1,
        ...         "batch_size": 16,
        ...     },
        ...     ppsci.loss.MSELoss("mean"),
        ...     name="EQ",
        ... )
        >>> solver = ppsci.solver.Solver(
        ...     model,
        ...     {"EQ": pde_constraint},
        ...     "./output",
        ...     opt,
        ...     None,
        ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        model: nn.Layer,
        constraint: Optional[Dict[str, ppsci.constraint.Constraint]] = None,
        output_dir: Optional[str] = "./output/",
        optimizer: Optional[optim.Optimizer] = None,
        lr_scheduler: Optional[optim.lr.LRScheduler] = None,
        epochs: int = 5,
        iters_per_epoch: int = 20,
        update_freq: int = 1,
        save_freq: int = 0,
        log_freq: int = 10,
        eval_during_train: bool = False,
        start_eval_epoch: int = 1,
        eval_freq: int = 1,
        seed: int = 42,
        use_vdl: bool = False,
        use_wandb: bool = False,
        wandb_config: Optional[Mapping] = None,
        device: Literal["cpu", "gpu", "xpu"] = "gpu",
        equation: Optional[Dict[str, ppsci.equation.PDE]] = None,
        geom: Optional[Dict[str, ppsci.geometry.Geometry]] = None,
        validator: Optional[Dict[str, ppsci.validate.Validator]] = None,
        visualizer: Optional[Dict[str, ppsci.visualize.Visualizer]] = None,
        use_amp: bool = False,
        amp_level: Literal["O1", "O2", "O0"] = "O0",
        pretrained_model_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        compute_metric_by_batch: bool = False,
        eval_with_no_grad: bool = False,
        to_static: bool = False,
        loss_aggregator: Optional[mtl.LossAggregator] = None,
    ):
        # set model
        self.model = model
        # set constraint
        self.constraint = constraint
        # set output directory
        self.output_dir = output_dir

        # set optimizer
        self.optimizer = optimizer
        # set learning rate scheduler
        self.lr_scheduler = lr_scheduler

        # set training hyper-parameter
        self.epochs = epochs
        self.iters_per_epoch = iters_per_epoch
        # set update_freq for gradient accumulation
        self.update_freq = update_freq
        # set checkpoint saving frequency
        self.save_freq = save_freq
        # set logging frequency
        self.log_freq = log_freq

        # set evaluation hyper-parameter
        self.eval_during_train = eval_during_train
        self.start_eval_epoch = start_eval_epoch
        self.eval_freq = eval_freq

        # initialize training log(training loss, time cost, etc.) recorder during one epoch
        self.train_output_info: Dict[str, misc.AverageMeter] = {}
        self.train_time_info = {
            "batch_cost": misc.AverageMeter("batch_cost", ".5f", postfix="s"),
            "reader_cost": misc.AverageMeter("reader_cost", ".5f", postfix="s"),
        }
        self.train_loss_info: Dict[str, misc.AverageMeter] = {}

        # initialize evaluation log(evaluation loss, metric, etc.) recorder.
        self.eval_output_info: Dict[str, misc.AverageMeter] = {}
        self.eval_time_info = {
            "batch_cost": misc.AverageMeter("batch_cost", ".5f", postfix="s"),
            "reader_cost": misc.AverageMeter("reader_cost", ".5f", postfix="s"),
        }

        # fix seed for reproducibility
        self.seed = seed

        # set running device
        if device != "cpu" and paddle.device.get_device() == "cpu":
            logger.warning(f"Set device({device}) to 'cpu' for only cpu available.")
            device = "cpu"
        self.device = paddle.set_device(device)

        # set equations for physics-driven or data-physics hybrid driven task, such as PINN
        self.equation = equation

        # set geometry for generating data
        self.geom = {} if geom is None else geom

        # set validator
        self.validator = validator

        # set visualizer
        self.visualizer = visualizer

        # set automatic mixed precision(AMP) configuration
        self.use_amp = use_amp
        self.amp_level = amp_level
        self.scaler = amp.GradScaler(True) if self.use_amp else None

        # whether calculate metrics by each batch during evaluation, mainly for memory efficiency
        self.compute_metric_by_batch = compute_metric_by_batch
        if validator is not None:
            for metric in itertools.chain(
                *[_v.metric.values() for _v in self.validator.values()]
            ):
                if metric.keep_batch ^ compute_metric_by_batch:
                    raise ValueError(
                        f"{misc.typename(metric)}.keep_batch should be "
                        f"{compute_metric_by_batch} when compute_metric_by_batch="
                        f"{compute_metric_by_batch}."
                    )
        # whether set `stop_gradient=True` for every Tensor if no differentiation involved during evaluation
        self.eval_with_no_grad = eval_with_no_grad

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # initialize distributed environment
        if self.world_size > 1:
            # TODO(sensen): support different kind of DistributedStrategy
            fleet.init(is_collective=True)
            logger.warning(
                f"Detected 'world_size'({self.world_size}) > 1, it is recommended to "
                "scale up the learning rate and reduce the 'epochs' or "
                "'iters_per_epoch' according to the 'world_size' both linearly if you "
                "are training model."
            )

        # load pretrained model, usually used for transfer learning
        if pretrained_model_path is not None:
            save_load.load_pretrain(self.model, pretrained_model_path, self.equation)

        # initialize an dict for tracking best metric during training
        self.best_metric = {
            "metric": float("inf"),
            "epoch": 0,
        }
        # load model checkpoint, usually used for resume training
        if checkpoint_path is not None:
            if pretrained_model_path is not None:
                logger.warning(
                    "Detected 'pretrained_model_path' is given, weights in which might be"
                    "overridden by weights loaded from given 'checkpoint_path'."
                )
            loaded_metric = save_load.load_checkpoint(
                checkpoint_path, self.model, self.optimizer, self.scaler, self.equation
            )
            if isinstance(loaded_metric, dict):
                self.best_metric.update(loaded_metric)

        # decorate model(s) and optimizer(s) for AMP
        if self.use_amp:
            self.model, self.optimizer = amp.decorate(
                self.model,
                self.optimizer,
                self.amp_level,
                save_dtype="float32",
            )

        # choosing an appropriate training function for different optimizers
        if misc.typename(self.optimizer) == "LBFGS":
            self.train_epoch_func = ppsci.solver.train.train_LBFGS_epoch_func
            if self.update_freq != 1:
                self.update_freq = 1
                logger.warning("Set 'update_freq' to to 1 when using L-BFGS optimizer.")
        else:
            self.train_epoch_func = ppsci.solver.train.train_epoch_func

        # wrap model and optimizer to parallel object
        if self.world_size > 1:
            if isinstance(self.model, paddle.DataParallel):
                raise ValueError(
                    "Given model is already wrapped by paddle.DataParallel."
                    "Please do not wrap your model with DataParallel "
                    "before 'Solver.__init__' and keep it's type as 'nn.Layer'."
                )

            def dist_wrapper(model: nn.Layer) -> paddle.DataParallel:
                dist_model = fleet.distributed_model(model)
                if hasattr(model, "input_keys"):
                    dist_model.input_keys = dist_model._layers.input_keys
                if hasattr(model, "output_keys"):
                    dist_model.output_keys = dist_model._layers.output_keys
                return dist_model

            if isinstance(self.model, ppsci.arch.ModelList):
                for i in range(len(self.model.model_list)):
                    # NOTE: Convert each model in model_list to DataParallel
                    self.model.model_list[i] = dist_wrapper(self.model.model_list[i])
            else:
                self.model = dist_wrapper(self.model)

            if self.optimizer is not None:
                self.optimizer = fleet.distributed_optimizer(self.optimizer)

        # set VisualDL tool
        self.vdl_writer = None
        if use_vdl:
            with misc.RankZeroOnly(self.rank) as is_master:
                if is_master:
                    self.vdl_writer = vdl.LogWriter(osp.join(output_dir, "vdl"))
            logger.info(
                "VisualDL tool is enabled for logging, you can view it by "
                f"running: 'visualdl --logdir {self.vdl_writer._logdir} --port 8080'."
            )

        # set WandB tool
        self.wandb_writer = None
        if use_wandb:
            try:
                import wandb
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "Please install 'wandb' with `pip install wandb` first."
                )
            with misc.RankZeroOnly(self.rank) as is_master:
                if is_master:
                    self.wandb_writer = wandb.init(**wandb_config)

        self.global_step = 0

        # log paddlepaddle's version
        if version.Version(paddle.__version__) != version.Version("0.0.0"):
            paddle_version = paddle.__version__
            if version.Version(paddle.__version__) < version.Version("2.6.0"):
                logger.warning(
                    f"Detected paddlepaddle version is '{paddle_version}', "
                    "currently it is recommended to use 2.6 or develop version."
                )
        else:
            paddle_version = f"develop({paddle.version.commit[:7]})"

        logger.info(f"Using paddlepaddle {paddle_version} on device {self.device}")

        self.forward_helper = expression.ExpressionSolver()

        # whether enable static for forward pass, defaults to False
        jit.enable_to_static(to_static)
        logger.info(f"Set to_static={to_static} for computational optimization.")

        # use loss aggregator, use summation if None
        self.loss_aggregator = loss_aggregator

        # convert sympy to callable object if exist
        extra_parameters = []
        if self.equation:
            for equation in self.equation.values():
                extra_parameters += list(equation.learnable_parameters)

        def convert_expr(
            container_dict: Union[
                Dict[str, ppsci.constraint.Constraint],
                Dict[str, ppsci.validate.Validator],
                Dict[str, ppsci.visualize.Visualizer],
            ]
        ) -> None:
            for container in container_dict.values():
                for name, expr in container.output_expr.items():
                    if isinstance(expr, sp.Basic):
                        container.output_expr[name] = ppsci.lambdify(
                            expr,
                            self.model,
                            extra_parameters,
                            # osp.join(self.output_dir, "symbolic_graph_visual", container.name, name), # HACK: Activate it for DEBUG.
                        )

        if self.constraint:
            convert_expr(self.constraint)

        if self.validator:
            convert_expr(self.validator)

        if self.visualizer:
            convert_expr(self.visualizer)

        # set up benchmark flag, will print memory stat if enabled
        self.benchmark_flag: bool = os.getenv("BENCHMARK_ROOT", None) is not None

    def train(self):
        """Training."""
        self.global_step = self.best_metric["epoch"] * self.iters_per_epoch
        start_epoch = self.best_metric["epoch"] + 1

        for epoch_id in range(start_epoch, self.epochs + 1):
            self.train_epoch_func(self, epoch_id, self.log_freq)
            self.train_output_info.clear()

            cur_metric = float("inf")
            # evaluate during training
            if (
                self.eval_during_train
                and epoch_id % self.eval_freq == 0
                and epoch_id >= self.start_eval_epoch
            ):
                cur_metric, metric_dict_group = self.eval(epoch_id)
                if cur_metric < self.best_metric["metric"]:
                    self.best_metric["metric"] = cur_metric
                    self.best_metric["epoch"] = epoch_id
                    save_load.save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.best_metric,
                        self.scaler,
                        self.output_dir,
                        "best_model",
                        self.equation,
                    )
                logger.info(
                    f"[Eval][Epoch {epoch_id}]"
                    f"[best metric: {self.best_metric['metric']}]"
                )
                for metric_dict in metric_dict_group.values():
                    logger.scaler(
                        {f"eval/{k}": v for k, v in metric_dict.items()},
                        epoch_id,
                        self.vdl_writer,
                        self.wandb_writer,
                    )

                # visualize after evaluation
                if self.visualizer is not None:
                    self.visualize(epoch_id)

            # update learning rate by epoch
            if self.lr_scheduler is not None and self.lr_scheduler.by_epoch:
                self.lr_scheduler.step()

            # save epoch model every save_freq epochs
            if self.save_freq > 0 and epoch_id % self.save_freq == 0:
                save_load.save_checkpoint(
                    self.model,
                    self.optimizer,
                    {"metric": cur_metric, "epoch": epoch_id},
                    self.scaler,
                    self.output_dir,
                    f"epoch_{epoch_id}",
                    self.equation,
                )

            # save the latest model for convenient resume training
            save_load.save_checkpoint(
                self.model,
                self.optimizer,
                {"metric": cur_metric, "epoch": epoch_id},
                self.scaler,
                self.output_dir,
                "latest",
                self.equation,
                print_log=(epoch_id == start_epoch),
            )

    @misc.run_on_eval_mode
    def eval(self, epoch_id: int = 0) -> Tuple[float, Dict[str, Dict[str, float]]]:
        """Evaluation.

        Args:
            epoch_id (int, optional): Epoch id. Defaults to 0.

        Returns:
            Tuple[float, Dict[str, Dict[str, float]]]: A targe metric value(float) and
                all metric(s)(dict) of evaluation, used to judge the quality of the model.
        """
        # set eval func
        self.eval_func = ppsci.solver.eval.eval_func

        result = self.eval_func(self, epoch_id, self.log_freq)
        metric_msg = ", ".join(
            [self.eval_output_info[key].avg_info for key in self.eval_output_info]
        )
        logger.info(f"[Eval][Epoch {epoch_id}][Avg] {metric_msg}")
        self.eval_output_info.clear()

        return result

    @misc.run_on_eval_mode
    def visualize(self, epoch_id: int = 0):
        """Visualization.

        Args:
            epoch_id (int, optional): Epoch id. Defaults to 0.
        """
        # set visualize func
        self.visu_func = ppsci.solver.visu.visualize_func

        self.visu_func(self, epoch_id)
        logger.info(f"[Visualize][Epoch {epoch_id}] Finish visualization")

    @misc.run_on_eval_mode
    def predict(
        self,
        input_dict: Dict[str, Union[np.ndarray, paddle.Tensor]],
        expr_dict: Optional[Dict[str, Callable]] = None,
        batch_size: int = 64,
        no_grad: bool = True,
        return_numpy: bool = False,
    ) -> Dict[str, Union[paddle.Tensor, np.ndarray]]:
        """Pure prediction using model.forward(...) and expression(optional, if given).

        Args:
            input_dict (Dict[str, Union[np.ndarray, paddle.Tensor]]): Input data in dict.
            expr_dict (Optional[Dict[str, Callable]]): Expression dict, which guide to
                compute equation variable with callable function. Defaults to None.
            batch_size (int, optional): Predicting by batch size. Defaults to 64.
            no_grad (bool): Whether set stop_gradient=True for entire prediction, mainly
                for memory-efficiency. Defaults to True.
            return_numpy (bool): Whether convert result from Tensor to numpy ndarray.
                Defaults to False.

        Returns:
            Dict[str, Union[paddle.Tensor, np.ndarray]]: Prediction in dict.

        Examples:
            >>> import paddle
            >>> import ppsci
            >>> paddle.seed(42)  # doctest: +SKIP
            >>> model = ppsci.arch.MLP(('x', 'y'), ('u', 'v'), num_layers=None, hidden_size=[32, 8])
            >>> solver = ppsci.solver.Solver(model)  # doctest: +SKIP
            >>> input_dict = {'x': paddle.rand((2, 1)),
            ...               'y': paddle.rand((2, 1))}
            >>> solver.predict(input_dict) # doctest: +SKIP
            {'u': Tensor(shape=[2, 1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                   [[-0.17509711],
                    [-0.03884222]]), 'v': Tensor(shape=[2, 1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                   [[0.27433380],
                    [0.42387512]])}
        """
        num_samples = len(next(iter(input_dict.values())))
        num_pad = (self.world_size - num_samples % self.world_size) % self.world_size
        # pad with last element if `num_samples` is not divisible by `world_size`
        # ensuring every device get same number of data.
        if num_pad > 0:
            for k, v in input_dict.items():
                repeat_times = (num_pad, *(1 for _ in range(v.ndim - 1)))
                if isinstance(v, np.ndarray):
                    input_dict[k] = np.concatenate(
                        (
                            v,
                            np.tile(v[num_samples - 1 : num_samples], repeat_times),
                        ),
                    )
                elif isinstance(v, paddle.Tensor):
                    input_dict[k] = paddle.concat(
                        (
                            v,
                            paddle.tile(v[num_samples - 1 : num_samples], repeat_times),
                        ),
                    )
                else:
                    raise ValueError(f"Unsupported data type {type(v)}.")

        num_samples_pad = num_samples + num_pad
        local_num_samples_pad = num_samples_pad // self.world_size
        local_input_dict = (
            {k: v[self.rank :: self.world_size] for k, v in input_dict.items()}
            if self.world_size > 1
            else input_dict
        )
        local_batch_num = (local_num_samples_pad + (batch_size - 1)) // batch_size

        pred_dict = misc.Prettydefaultdict(list)
        with self.no_grad_context_manager(no_grad), self.no_sync_context_manager(
            self.world_size > 1, self.model
        ):
            for batch_id in range(local_batch_num):
                batch_input_dict = {}
                st = batch_id * batch_size
                ed = min(local_num_samples_pad, (batch_id + 1) * batch_size)

                # prepare batch input dict
                for key in local_input_dict:
                    if not paddle.is_tensor(local_input_dict[key]):
                        batch_input_dict[key] = paddle.to_tensor(
                            local_input_dict[key][st:ed], paddle.get_default_dtype()
                        )
                    else:
                        batch_input_dict[key] = local_input_dict[key][st:ed]
                    batch_input_dict[key].stop_gradient = no_grad

                # forward
                with self.autocast_context_manager(self.use_amp, self.amp_level):
                    batch_output_dict = self.forward_helper.visu_forward(
                        expr_dict, batch_input_dict, self.model
                    )

                # collect batch data
                for key, batch_output in batch_output_dict.items():
                    pred_dict[key].append(
                        batch_output.detach() if no_grad else batch_output
                    )

            # concatenate local predictions
            pred_dict = {key: paddle.concat(value) for key, value in pred_dict.items()}

            if self.world_size > 1:
                # gather global predictions from all devices if world_size > 1
                pred_dict = {
                    key: misc.all_gather(value) for key, value in pred_dict.items()
                }
                # rearrange predictions as the same order of input_dict according
                # to inverse permutation
                perm = np.arange(num_samples_pad, dtype="int64")
                perm = np.concatenate(
                    [perm[rank :: self.world_size] for rank in range(self.world_size)],
                    axis=0,
                )
                perm_inv = np.empty_like(perm)
                perm_inv[perm] = np.arange(num_samples_pad, dtype="int64")
                perm_inv = paddle.to_tensor(perm_inv)
                pred_dict = {key: value[perm_inv] for key, value in pred_dict.items()}
                # then discard predictions of padding data at the end if num_pad > 0
                if num_pad > 0:
                    pred_dict = {
                        key: value[:num_samples] for key, value in pred_dict.items()
                    }
                    # NOTE: Discard padding data in input_dict for consistency
                    for k in input_dict:
                        input_dict[k] = input_dict[k][:num_samples]

        # convert to numpy ndarray if specified
        if return_numpy:
            pred_dict = {
                k: (v.numpy() if paddle.is_tensor(v) else v)
                for k, v in pred_dict.items()
            }

        return pred_dict

    @misc.run_on_eval_mode
    def export(self):
        """Export to inference model."""
        raise NotImplementedError("model export is not supported yet.")

    def autocast_context_manager(
        self, enable: bool, level: Literal["O0", "O1", "O2"] = "O1"
    ) -> contextlib.AbstractContextManager:
        """Smart autocast context manager for Auto Mix Precision.

        Args:
            enable (bool): Enable autocast.
            level (Literal["O0", "O1", "O2"]): Autocast level.

        Returns:
            contextlib.AbstractContextManager: Smart autocast context manager.
        """
        if enable:
            ctx_manager = amp.auto_cast(level=level)
        else:
            ctx_manager = (
                contextlib.nullcontext()
                if sys.version_info >= (3, 7)
                else contextlib.suppress()
            )
        return ctx_manager

    def no_grad_context_manager(
        self, enable: bool
    ) -> contextlib.AbstractContextManager:
        """Smart no_grad context manager.

        Args:
            enable (bool): Enable no_grad.

        Returns:
            contextlib.AbstractContextManager: Smart no_grad context manager.
        """
        if enable:
            ctx_manager = paddle.no_grad()
        else:
            ctx_manager = (
                contextlib.nullcontext()
                if sys.version_info >= (3, 7)
                else contextlib.suppress()
            )
        return ctx_manager

    def no_sync_context_manager(
        self,
        enable: bool,
        ddp_model: paddle.DataParallel,
    ) -> contextlib.AbstractContextManager:
        """Smart no_sync context manager for given model.
        NOTE: Only `paddle.DataParallel` object has `no_sync` interface.

        Args:
            enable (bool): Enable no_sync.

        Returns:
            contextlib.AbstractContextManager: Smart no_sync context manager.
        """
        if enable:
            if isinstance(self.model, ppsci.arch.ModelList):
                for model in self.model.model_list:
                    if not isinstance(model, paddle.DataParallel):
                        raise TypeError(
                            "no_sync interface is only for model with type "
                            "paddle.DataParallel, but got type "
                            f"{misc.typename(model)}"
                        )
                ctx_manager = contextlib.ExitStack()
                for model in self.model.model_list:
                    ctx_manager.enter_context(model.no_sync())
            else:
                if not isinstance(self.model, paddle.DataParallel):
                    raise TypeError(
                        "no_sync interface is only for model with type "
                        f"paddle.DataParallel, but got type {misc.typename(ddp_model)}"
                    )
                ctx_manager = ddp_model.no_sync()
        else:
            ctx_manager = (
                contextlib.nullcontext()
                if sys.version_info >= (3, 7)
                else contextlib.suppress()
            )
        return ctx_manager

    def plot_loss_history(
        self,
        by_epoch: bool = False,
        smooth_step: int = 1,
        use_semilogy: bool = True,
    ) -> None:
        """Plotting iteration/epoch-loss curve.

        Args:
            by_epoch (bool, optional): Whether the abscissa axis of the curve is epoch or iteration. Defaults to False.
            smooth_step (int, optional): How many steps of loss are squeezed to one point to smooth the curve. Defaults to 1.
            use_semilogy (bool, optional): Whether to set non-uniform coordinates for the y-axis. Defaults to True.
        """
        loss_dict = {}
        for key in self.train_loss_info:
            loss_arr = np.asarray(self.train_loss_info[key].history)
            if by_epoch:
                loss_arr = np.mean(
                    np.reshape(loss_arr, (-1, self.iters_per_epoch)),
                    axis=1,
                )
            loss_dict[key] = list(loss_arr)

        misc.plot_curve(
            data=loss_dict,
            xlabel="Epoch" if by_epoch else "Iteration",
            ylabel="Loss",
            output_dir=self.output_dir,
            smooth_step=smooth_step,
            use_semilogy=use_semilogy,
        )
