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

import copy
import os
import random

import numpy as np
import paddle
import paddle.amp as amp
import paddle.distributed as dist
from packaging import version
from paddle.distributed import fleet
from visualdl import LogWriter

import ppsci
from ppsci.utils import config
from ppsci.utils import logger
from ppsci.utils import misc
from ppsci.utils import save_load


class Solver(object):
    """Base class for solver.

    Args:
        cfg (AttrDict): Configuration parsed from yaml.
        mode (str, optional): Running mode. Defaults to "train".
    """

    def __init__(self):
        pass

    def initialize_from_config(self, cfg, log_level, mode="train"):
        self.cfg = cfg
        self.mode = mode
        self.rank = dist.get_rank()

        # set random seed
        seed = self.cfg["Global"].get("seed", 42)
        if seed is not None and seed is not False:
            if not isinstance(seed, int):
                raise ValueError(f"Global.seed({seed}) should be a integer")
            paddle.seed(seed + self.rank)
            np.random.seed(seed + self.rank)
            random.seed(seed + self.rank)

        # init logger
        self.output_dir = self.cfg["Global"]["output_dir"]
        log_file = os.path.join(
            self.output_dir, self.cfg["Arch"]["name"], f"{mode}.log"
        )
        self.log_freq = self.cfg["Global"].get("log_freq", 20)
        logger.init_logger(log_file=log_file, log_level=log_level)
        config.print_config(self.cfg)

        # init VisualDL
        self.vdl_writer = None
        if self.rank == 0 and self.cfg["Global"]["use_visualdl"]:
            vdl_writer_path = os.path.join(self.output_dir, "visualdl")
            os.makedirs(vdl_writer_path, exist_ok=True)
            self.vdl_writer = LogWriter(logdir=vdl_writer_path)

        # set runtime device
        if self.cfg["Global"]["device"] not in [
            "cpu",
            "gpu",
            "xpu",
            "npu",
            "mlu",
            "ascend",
        ]:
            raise ValueError(
                f"Global.device({self.cfg['Global']['device']}) "
                f"must in ['cpu', 'gpu', 'xpu', 'npu', 'mlu', 'ascend']"
            )
        self.device = paddle.set_device(self.cfg["Global"]["device"])

        # log paddlepaddle's version
        paddle_version = (
            paddle.__version__
            if version.Version(paddle.__version__) != version.Version("0.0.0")
            else f"develop({paddle.version.commit[:7]})"
        )
        logger.info(f"Using paddlepaddle {paddle_version} on device {self.device}")

        # build geometry(ies) if specified
        if "Geometry" in self.cfg:
            self.geom = ppsci.geometry.build_geometry(self.cfg["Geometry"])
        else:
            self.geom = {}

        # build model
        self.model = ppsci.arch.build_model(self.cfg["Arch"])

        # build equation(s) if specified
        if "Equation" in self.cfg:
            self.equation = ppsci.equation.build_equation(self.cfg["Equation"])
        else:
            self.equation = {}

        # init AMP
        self.use_amp = "AMP" in self.cfg
        if self.use_amp:
            self.amp_level = self.cfg["AMP"].pop("level", "O1").upper()
            self.scaler = amp.GradScaler(True, **self.cfg["AMP"])
        else:
            self.amp_level = "O0"

        # init world_size
        self.world_size = dist.get_world_size()

    def train(self):
        """Training"""
        epochs = self.cfg["Global"]["epochs"]
        self.iters_per_epoch = self.cfg["Global"]["iters_per_epoch"]

        save_freq = self.cfg["Global"].get("save_freq", 1)

        eval_during_train = self.cfg["Global"].get("eval_during_train", True)
        eval_freq = self.cfg["Global"].get("eval_freq", 1)
        start_eval_epoch = self.cfg["Global"].get("start_eval_epoch", 1)

        # init gradient accumulation config
        self.update_freq = self.cfg["Global"].get("update_freq", 1)

        best_metric = {
            "metric": float("inf"),
            "epoch": 0,
        }

        # load checkpoint if specified
        if self.cfg["Global"]["checkpoints"] is not None:
            loaded_metric = save_load.load_checkpoint(
                self.cfg["Global"]["checkpoints"], self.model, self.optimizer
            )
            if isinstance(loaded_metric, dict):
                best_metric.update(loaded_metric)

        # init constraint(s)
        self.constraints = ppsci.constraint.build_constraint(
            self.cfg["Constraint"], self.equation, self.geom
        )

        # init optimizer and lr scheduler
        self.optimizer, self.lr_scheduler = ppsci.optimizer.build_optimizer(
            self.cfg["Optimizer"], [self.model], epochs, self.iters_per_epoch
        )

        self.train_output_info = {}
        self.train_time_info = {
            "batch_cost": misc.AverageMeter("batch_cost", ".5f", postfix="s"),
            "reader_cost": misc.AverageMeter("reader_cost", ".5f", postfix="s"),
        }

        # init train func
        self.train_mode = self.cfg["Global"].get("train_mode", None)
        if self.train_mode is None:
            self.train_epoch_func = ppsci.solver.train.train_epoch_func
        else:
            self.train_epoch_func = ppsci.solver.train.train_LBFGS_epoch_func

        # init distributed environment
        if self.world_size > 1:
            # TODO(sensen): support different kind of DistributedStrategy
            fleet.init(is_collective=True)
            self.model = fleet.distributed_model(self.model)
            self.optimizer = fleet.distributed_optimizer(self.optimizer)

        # train epochs
        self.global_step = 0
        for epoch_id in range(best_metric["epoch"] + 1, epochs + 1):
            self.train_epoch_func(self, epoch_id, self.log_freq)

            # log training summation at end of a epoch
            metric_msg = ", ".join(
                [self.train_output_info[key].avg_info for key in self.train_output_info]
            )
            logger.info(f"[Train][Epoch {epoch_id}/{epochs}][Avg] {metric_msg}")
            self.train_output_info.clear()

            # evaluate during training
            if (
                eval_during_train
                and epoch_id % eval_freq == 0
                and epoch_id >= start_eval_epoch
            ):
                cur_metric = self.eval(epoch_id)
                if cur_metric < best_metric["metric"]:
                    best_metric["metric"] = cur_metric
                    best_metric["epoch"] = epoch_id
                    save_load.save_checkpoint(
                        self.model,
                        self.optimizer,
                        best_metric,
                        self.output_dir,
                        self.cfg["Arch"]["name"],
                        "best_model",
                    )
            logger.info(
                f"[Eval][Epoch {epoch_id}]" f"[best metric: {best_metric['metric']}]"
            )
            logger.scaler("eval_metric", cur_metric, epoch_id, self.vdl_writer)

            # update learning rate by epoch
            if self.lr_scheduler.by_epoch:
                self.lr_scheduler.step()

            # save epoch model every `save_freq`
            if save_freq > 0 and epoch_id % save_freq == 0:
                save_load.save_checkpoint(
                    self.model,
                    self.optimizer,
                    {"metric": cur_metric, "epoch": epoch_id},
                    self.output_dir,
                    self.cfg["Arch"]["name"],
                    f"epoch_{epoch_id}",
                )

            # always save the latest model for convenient resume training
            save_load.save_checkpoint(
                self.model,
                self.optimizer,
                {"metric": cur_metric, "epoch": epoch_id},
                self.output_dir,
                self.cfg["Arch"]["name"],
                "latest",
            )

        # close VisualDL
        if self.vdl_writer is not None:
            self.vdl_writer.close()

    def eval(self, epoch_id=0):
        """Evaluation"""
        # load pretrained model if specified
        if self.cfg["Global"]["pretrained_model"] is not None:
            save_load.load_pretrain(self.model, self.cfg["Global"]["pretrained_model"])

        self.model.eval()

        # init train func
        self.eval_func = ppsci.solver.eval.eval_func

        # init validator(s) at the first time
        if not hasattr(self, "validator"):
            self.validator = ppsci.validate.build_validator(
                self.cfg["Validator"], self.geom, self.equation
            )

        self.eval_output_info = {}
        self.eval_time_info = {
            "batch_cost": misc.AverageMeter("batch_cost", ".5f", postfix="s"),
            "reader_cost": misc.AverageMeter("reader_cost", ".5f", postfix="s"),
        }

        result = self.eval_func(self, epoch_id, self.log_freq)
        metric_msg = ", ".join(
            [self.eval_output_info[key].avg_info for key in self.eval_output_info]
        )
        logger.info(f"[Eval][Epoch {epoch_id}][Avg] {metric_msg}")
        self.eval_output_info.clear()

        self.model.train()
        return result

    def predict(self, input_dict):
        """Prediction"""
        pred_dict = self.model(input_dict)
        return pred_dict

    def export(self):
        """Export to inference model"""
        pretrained_path = self.cfg["Global"]["pretrained_model"]
        if pretrained_path is not None:
            save_load.load_pretrain(self.model, pretrained_path)

        self.model.eval()

        input_spec = copy.deepcopy(self.cfg["Export"]["input_shape"])
        config.replace_shape_with_inputspec_(input_spec)
        static_model = paddle.jit.to_static(self.model, input_spec=input_spec)

        export_dir = self.cfg["Global"]["save_inference_dir"]
        save_path = os.path.join(export_dir, "inference")
        paddle.jit.save(static_model, save_path)
        logger.info(f"The inference model has been exported to {export_dir}.")
