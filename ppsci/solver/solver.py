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

import copy
import os
from typing import Any
from typing import Dict
from typing import Optional

import paddle
import paddle.amp as amp
import paddle.distributed as dist
import paddle.incubate as incubate
import paddle.nn as nn
import paddle.optimizer as optimizer
import visualdl as vdl
from packaging import version
from paddle.distributed import fleet
from typing_extensions import Literal

import ppsci
from ppsci.utils import config
from ppsci.utils import logger
from ppsci.utils import misc
from ppsci.utils import save_load


class Solver:
    """Class for solver.

    Args:
        model (nn.Layer): Model.
        constraint (Optional[Dict[str, ppsci.constraint.Constraint]]): Constraint(s) applied on model. Defaults to None.
        output_dir (str, optional): Output directory. Defaults to "./output/".
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
        vdl_writer (Optional[vdl.LogWriter]): VisualDL writer object. Defaults to None.
        device (Literal["cpu", "gpu", "xpu"], optional): _description_. Defaults to "gpu".
        equation (Optional[Dict[str, ppsci.equation.PDE]]): Equation dict. Defaults to None.
        geom (Optional[Dict[str, ppsci.geometry.Geometry]]): Geometry dict. Defaults to None.
        validator (Optional[Dict[str, ppsci.validate.Validator]]): Validator dict. Defaults to None.
        visualizer (Optional[Dict[str, ppsci.visualize.Visualizer]]): Visualizer dict. Defaults to None.
        use_amp (bool, optional): Whether use AMP. Defaults to False.
        amp_level (Literal["O1", "O2", "O0"], optional): AMP level. Defaults to "O0".
        pretrained_model_path (Optional[str]): Pretrained model path. Defaults to None.
        checkpoint_path (Optional[str]): Checkpoint path. Defaults to None.
    """

    def __init__(
        self,
        model: nn.Layer,
        constraint: Optional[Dict[str, ppsci.constraint.Constraint]] = None,
        output_dir: str = "./output/",
        optimizer: Optional[optimizer.Optimizer] = None,
        lr_scheduler: Optional[optimizer.lr.LRScheduler] = None,
        epochs: int = 5,
        iters_per_epoch: int = 20,
        update_freq: int = 1,
        save_freq: int = 0,
        log_freq: int = 10,
        eval_during_train: bool = False,
        start_eval_epoch: int = 1,
        eval_freq: int = 1,
        seed: int = 42,
        vdl_writer: Optional[vdl.LogWriter] = None,
        device: Literal["cpu", "gpu", "xpu"] = "gpu",
        equation: Optional[Dict[str, ppsci.equation.PDE]] = None,
        geom: Optional[Dict[str, ppsci.geometry.Geometry]] = None,
        validator: Optional[Dict[str, ppsci.validate.Validator]] = None,
        visualizer: Optional[Dict[str, ppsci.visualize.Visualizer]] = None,
        use_amp: bool = False,
        amp_level: Literal["O1", "O2", "O0"] = "O0",
        pretrained_model_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
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

        # initialize traning log recorder for loss, time cost, metric, etc.
        self.train_output_info = {}
        self.train_time_info = {
            "batch_cost": misc.AverageMeter("batch_cost", ".5f", postfix="s"),
            "reader_cost": misc.AverageMeter("reader_cost", ".5f", postfix="s"),
        }

        # initialize evaluation log recorder for loss, time cost, metric, etc.
        self.eval_output_info = {}
        self.eval_time_info = {
            "batch_cost": misc.AverageMeter("batch_cost", ".5f", postfix="s"),
            "reader_cost": misc.AverageMeter("reader_cost", ".5f", postfix="s"),
        }

        # fix seed for reproducibility
        self.seed = seed

        # set VisualDL tool
        self.vdl_writer = vdl_writer

        # set running device
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
            loaded_metric = save_load.load_checkpoint(
                checkpoint_path, self.model, self.optimizer, self.scaler, self.equation
            )
            if isinstance(loaded_metric, dict):
                self.best_metric.update(loaded_metric)

        # choosing an appropriate training function for different optimizers
        if not isinstance(self.optimizer, incubate.optimizer.LBFGS):
            self.train_epoch_func = ppsci.solver.train.train_epoch_func
        else:
            self.train_epoch_func = ppsci.solver.train.train_LBFGS_epoch_func

        # decorate model(s) and optimizer(s) for AMP
        if self.use_amp:
            self.model = amp.decorate(self.model, self.optimizer, self.amp_level)

        # wrap model and optimizer to parallel object
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        if self.world_size > 1:
            # TODO(sensen): support different kind of DistributedStrategy
            fleet.init(is_collective=True)
            self.model = fleet.distributed_model(self.model)
            if self.optimizer is not None:
                self.optimizer = fleet.distributed_optimizer(self.optimizer)

        self.global_step = 0

        # log paddlepaddle's version
        paddle_version = (
            paddle.__version__
            if version.Version(paddle.__version__) != version.Version("0.0.0")
            else f"develop({paddle.version.commit[:7]})"
        )
        logger.info(f"Using paddlepaddle {paddle_version} on device {self.device}")

    @staticmethod
    def from_config(cfg: Dict[str, Any]) -> Solver:
        """Initialize solver from given config.

        Args:
            cfg (Dict[str, Any]): Dict config, e.g. AttrDict parsed from yaml.

        Returns:
            Solver: Initialized solver object.
        """
        config.print_config(cfg)
        # TODO(sensen): sanity check for config
        output_dir = cfg["Global"]["output_dir"]
        epochs = cfg["Global"]["epochs"]
        iters_per_epoch = cfg["Global"]["iters_per_epoch"]
        save_freq = cfg["Global"]["save_freq"]
        eval_during_train = cfg["Global"]["eval_during_train"]
        eval_freq = cfg["Global"]["eval_freq"]

        seed = cfg["Global"].get("seed", 42)
        rank = dist.get_rank()
        misc.set_random_seed(seed + rank)

        model = ppsci.arch.build_model(cfg["Arch"])
        geom = ppsci.geometry.build_geometry(cfg.get("Geometry", None))
        equation = ppsci.equation.build_equation(cfg.get("Equation", None))
        constraint = ppsci.constraint.build_constraint(
            cfg["Global"].get("Constraint", None),
            equation,
            geom,
        )
        optimizer, lr_scheduler = ppsci.optimizer.build_optimizer(
            cfg["Global"]["Optimizer"],
            model + ([eq for eq in equation.values()] if equation is not None else []),
            epochs,
            iters_per_epoch,
        )

        vdl_writer = None
        if cfg["Global"].get("vdl_writer", False):
            vdl_writer_path = os.path.join(output_dir, "vdl")
            os.makedirs(vdl_writer_path, exist_ok=True)
            vdl_writer = vdl.LogWriter(vdl_writer_path)

        log_freq = cfg["Global"].get("log_freq", 10)
        device = cfg["Global"].get("device", "gpu")
        validator = ppsci.validate.build_validator(
            cfg.get("Validator", None), equation, geom
        )
        visualizer = ppsci.visualize.build_visualizer(cfg.get("Visualizer", None))
        use_amp = "AMP" in cfg
        amp_level = cfg["AMP"].pop("level", "O1").upper() if use_amp else "O0"

        start_eval_epoch = cfg["Global"].get("start_eval_epoch", 1)
        update_freq = cfg["Global"].get("update_freq", 1)
        pretrained_model_path = cfg["Global"].get("pretrained_model_path", None)
        checkpoint_path = cfg["Global"].get("checkpoint_path", None)

        return Solver(
            model,
            constraint,
            output_dir,
            optimizer,
            lr_scheduler,
            epochs,
            iters_per_epoch,
            update_freq,
            save_freq,
            log_freq,
            eval_during_train,
            start_eval_epoch,
            eval_freq,
            seed,
            vdl_writer,
            device,
            equation,
            geom,
            validator,
            visualizer,
            use_amp,
            amp_level,
            pretrained_model_path,
            checkpoint_path,
        )

    def train(self):
        """Training"""
        self.global_step = self.best_metric["epoch"] * self.iters_per_epoch + 1

        for epoch_id in range(self.best_metric["epoch"] + 1, self.epochs + 1):
            self.train_epoch_func(self, epoch_id, self.log_freq)

            # log training summation at end of a epoch
            metric_msg = ", ".join(
                [self.train_output_info[key].avg_info for key in self.train_output_info]
            )
            logger.info(f"[Train][Epoch {epoch_id}/{self.epochs}][Avg] {metric_msg}")
            self.train_output_info.clear()

            cur_metric = float("inf")
            # evaluate during training
            if (
                self.eval_during_train
                and epoch_id % self.eval_freq == 0
                and epoch_id >= self.start_eval_epoch
            ):
                cur_metric = self.eval(epoch_id)
                if cur_metric < self.best_metric["metric"]:
                    self.best_metric["metric"] = cur_metric
                    self.best_metric["epoch"] = epoch_id
                    save_load.save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.scaler,
                        self.best_metric,
                        self.output_dir,
                        "best_model",
                        self.equation,
                    )
                logger.info(
                    f"[Eval][Epoch {epoch_id}]"
                    f"[best metric: {self.best_metric['metric']}]"
                )
                logger.scaler("eval_metric", cur_metric, epoch_id, self.vdl_writer)

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
                    self.scaler,
                    {"metric": cur_metric, "epoch": epoch_id},
                    self.output_dir,
                    f"epoch_{epoch_id}",
                    self.equation,
                )

            # always save the latest model for convenient resume training
            save_load.save_checkpoint(
                self.model,
                self.optimizer,
                self.scaler,
                {"metric": cur_metric, "epoch": epoch_id},
                self.output_dir,
                "latest",
                self.equation,
            )

        # close VisualDL
        if self.vdl_writer is not None:
            self.vdl_writer.close()

    def eval(self, epoch_id=0):
        """Evaluation"""
        self.model.eval()

        # set eval func
        self.eval_func = ppsci.solver.eval.eval_func

        result = self.eval_func(self, epoch_id, self.log_freq)
        metric_msg = ", ".join(
            [self.eval_output_info[key].avg_info for key in self.eval_output_info]
        )
        logger.info(f"[Eval][Epoch {epoch_id}][Avg] {metric_msg}")
        self.eval_output_info.clear()

        self.model.train()
        return result

    def visualize(self, epoch_id=0):
        """Visualization"""
        self.model.eval()

        # init train func
        self.visu_func = ppsci.solver.visu.visualize_func

        self.visu_func(self, epoch_id)
        logger.info(f"[Visualize][Epoch {epoch_id}] Finished visualization.")

        self.model.train()

    def predict(self, input_dict):
        """Prediction"""
        pred_dict = self.model(input_dict)
        return pred_dict

    def export(self):
        """Export to inference model"""
        pretrained_path = self.cfg["Global"]["pretrained_model"]
        if pretrained_path is not None:
            save_load.load_pretrain(self.model, pretrained_path, self.equation)

        self.model.eval()

        input_spec = copy.deepcopy(self.cfg["Export"]["input_shape"])
        config.replace_shape_with_inputspec_(input_spec)
        static_model = paddle.jit.to_static(self.model, input_spec=input_spec)

        export_dir = self.cfg["Global"]["save_inference_dir"]
        save_path = os.path.join(export_dir, "inference")
        paddle.jit.save(static_model, save_path)
        logger.info(f"The inference model has been exported to {export_dir}.")
