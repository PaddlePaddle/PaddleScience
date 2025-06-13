# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import os
import random
import time
from argparse import Namespace
from collections import OrderedDict

import hydra
import numpy as np
import paddle
import paddle.amp as amp
import paddle.distributed as dist
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim
import yaml
from einops import rearrange
from omegaconf import DictConfig
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
from scipy.stats import linregress
from tqdm import tqdm
from utils import logging_utils
from visualdl import LogWriter

from ppsci.arch.data_efficient_nopt_model import YParams
from ppsci.arch.data_efficient_nopt_model import build_fno
from ppsci.arch.data_efficient_nopt_model import build_vmae
from ppsci.arch.data_efficient_nopt_model import fno_pretrain
from ppsci.arch.data_efficient_nopt_model import gaussian_blur
from ppsci.data.dataset.data_efficient_nopt_dataset import MixedDatasetLoader
from ppsci.data.dataset.data_efficient_nopt_dataset import PoisHelmDatasetLoader


def l2_err(pred, target, spatial_dim=(-1, -2, -3)):
    x = paddle.sum((pred - target) ** 2, axis=spatial_dim) / paddle.sum(
        target**2, axis=spatial_dim
    )
    x = paddle.sqrt(x)
    return paddle.mean(x)


def grad_norm(parameters):
    with paddle.no_grad():
        total_norm = 0
        for p in parameters:
            if p.grad is not None:
                total_norm += p.grad.data.pow(2).sum().item()
        return total_norm**0.5


def grad_clone(parameters):
    with paddle.no_grad():
        clones = []
        for p in parameters:
            if p.grad is not None:
                clones.append(p.grad.clone())
            else:
                clones.append(paddle.zeros_like(p))
        return clones


def param_norm(parameters):
    with paddle.no_grad():
        total_norm = 0
        for p in parameters:
            total_norm += p.pow(2).sum().item()
        return total_norm**0.5


def param_diff(params1, params2):
    with paddle.no_grad():
        total_norm = 0
        for p1, p2 in zip(params1, params2):
            total_norm += (p2 - p1).pow(2).sum().item()
        return total_norm**0.5


def add_weight_decay(model, weight_decay=1e-5, inner_lr=1e-3, skip_list=()):
    """From Ross Wightman at:
    https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3

    Goes through the parameter list and if the squeeze dim is 1 or 0 (usually means bias or scale)
    then don't apply weight decay.
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if param.stop_gradient:
            continue
        if len(param.squeeze().shape) <= 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {
            "params": no_decay,
            "weight_decay": 0.0,
        },
        {"params": decay, "weight_decay": weight_decay},
    ]


class Trainer:
    def __init__(self, params, global_rank, local_rank, device, sweep_id=None):
        self.device = device
        self.params = params
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.sweep_id = sweep_id
        self.log_to_screen = params.log_to_screen
        self.train_loss = nn.MSELoss()
        self.startEpoch = 0
        self.epoch = 0
        self.debug_grad = params.debug_grad
        self.mp_type = (
            "bfloat16"
            if paddle.device.cuda.device_count() >= 1
            and paddle.amp.is_bfloat16_supported()
            else "float16"
        )

        self.iters = 0
        self.initialize_data(self.params)
        print(f"Initializing model on rank {self.global_rank}")
        self.initialize_model(self.params)
        self.initialize_optimizer(self.params)
        if params.resuming:
            print("Loading checkpoint %s" % params.checkpoint_path)
            print("LOADING CHECKPOINTTTTTT")
            self.restore_checkpoint(params.checkpoint_path)
        if params.resuming is False and params.pretrained:
            print("Starting from pretrained model at %s" % params.pretrained_ckpt_path)
            self.restore_checkpoint(params.pretrained_ckpt_path)
            self.iters = 0
            self.startEpoch = 0
        self.initialize_scheduler(self.params)

    def single_print(self, *text):
        if self.global_rank == 0 and self.log_to_screen:
            print(" ".join([str(t) for t in text]))

    def initialize_data(self, params):
        if params.tie_batches:
            in_rank = 0
        else:
            in_rank = self.global_rank
        if self.log_to_screen:
            print(f"Initializing data on rank {self.global_rank}")

        if self.params.model_type == "fno":
            params.masking = ((params.nx, params.ny), params.mask_ratio)
            (
                self.train_data_loader,
                self.train_dataset,
                self.train_sampler,
            ) = PoisHelmDatasetLoader(
                params, params.train_path, dist.is_initialized(), train=True
            )
            (
                self.valid_data_loader,
                self.valid_dataset,
                self.valid_sampler,
            ) = PoisHelmDatasetLoader(
                params, params.val_path, dist.is_initialized(), train=False
            )
        elif self.params.model_type == "vmae":
            params.masking = (
                (
                    params.n_steps,
                    params.input_size // params.patch_size,
                    params.input_size // params.patch_size,
                ),
                params.mask_ratio,
            )

            (
                self.train_data_loader,
                self.train_dataset,
                self.train_sampler,
            ) = MixedDatasetLoader(
                params,
                params.train_data_paths,
                dist.is_initialized(),
                split="train",
                rank=in_rank,
                train_offset=self.params.embedding_offset,
            )
            self.valid_data_loader, self.valid_dataset, _ = MixedDatasetLoader(
                params,
                params.valid_data_paths,
                dist.is_initialized(),
                split="val",
                rank=in_rank,
            )
        if dist.is_initialized():
            self.train_sampler.set_epoch(0)

    def initialize_model(self, params):
        if self.params.model_type == "fno":
            self.model = fno_pretrain(params)
        elif self.params.model_type == "vmae":
            self.model = build_vmae(params)

        if dist.is_initialized():
            self.model = paddle.DataParallel(
                self.model,
                find_unused_parameters=True,
            )

        self.single_print(
            f"Model parameter count: {sum([p.numel() for p in self.model.parameters()])}"
        )

    def initialize_optimizer(self, params):
        parameters = add_weight_decay(self.model, self.params.weight_decay)
        if params.optimizer == "adam":
            self.optimizer = optim.AdamW(
                parameters=parameters, learning_rate=params.learning_rate
            )
        elif params.optimizer == "adan":
            raise NotImplementedError("Adan not implemented yet")
        elif params.optimizer == "sgd":
            self.optimizer = optim.SGD(
                parameters=self.model.parameters(),
                learning_rate=params.learning_rate,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Optimizer {params.optimizer} not supported")
        self.gscaler = amp.GradScaler(
            enable=(self.mp_type == paddle.float16 and params.enable_amp)
        )

    def initialize_scheduler(self, params):
        if params.scheduler_epochs > 0:
            sched_epochs = params.scheduler_epochs
        else:
            sched_epochs = params.max_epochs
        if params.scheduler == "cosine":
            if self.params.learning_rate < 0:
                self.scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
                    learning_rate=self.optimizer.get_lr(),
                    last_epoch=(self.startEpoch * params.epoch_size) - 1,
                    T_max=sched_epochs * params.epoch_size,
                    eta_min=params.learning_rate / 100,
                )
                self.optimizer.set_lr_scheduler(self.scheduler)
            else:
                k = params.warmup_steps
                if (self.startEpoch * params.epoch_size) < k:
                    warmup = paddle.optimizer.lr.LinearLR(
                        learning_rate=self.optimizer.get_lr(),
                        start_factor=0.01,
                        end_factor=1.0,
                        total_iters=k,
                    )
                    self.optimizer.set_lr_scheduler(warmup)
                    decay = paddle.optimizer.lr.CosineAnnealingDecay(
                        learning_rate=self.optimizer.get_lr(),
                        eta_min=params.learning_rate / 100,
                        T_max=sched_epochs,
                    )
                    self.optimizer.set_lr_scheduler(decay)

                    raise NotImplementedError("Scheduler not implemented yet")
                else:
                    self.scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
                        learning_rate=self.optimizer.get_lr(), T_max=sched_epochs
                    )
                    self.optimizer.set_lr_scheduler(self.scheduler)
        elif params.scheduler == "reducelr":
            self.scheduler = paddle.optimizer.lr.ReduceOnPlateau(
                learning_rate=self.optimizer.get_lr(),
                mode="min",
                patience=params.patience,
                verbose=True,
                min_lr=1e-3 * 1e-5,
                factor=0.2,
            )
            self.optimizer.set_lr_scheduler(self.scheduler)
        else:
            self.scheduler = None

    def save_checkpoint(self, checkpoint_path, model=None):
        """Save model and optimizer to checkpoint"""
        if not model:
            model = self.model

        paddle.save(
            {
                "iters": self.epoch * self.params.epoch_size,
                "epoch": self.epoch,
                "model_state": model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )

    def restore_checkpoint(self, checkpoint_path):
        """Load model/opt from path"""
        checkpoint = paddle.load(checkpoint_path)
        try:
            self.model.set_state_dict(checkpoint["model_state"])
        except:  # noqa
            new_state_dict = OrderedDict()
            for key, val in checkpoint["model_state"].items():
                name = key[7:]
                new_state_dict[name] = val
            self.model.set_state_dict(new_state_dict)

        if self.params.resuming:
            self.optimizer.set_state_dict(checkpoint["optimizer_state_dict"])
            self.startEpoch = checkpoint["epoch"]
            self.epoch = self.startEpoch
            self.iters = checkpoint["iters"]
        else:
            self.iters = 0
        checkpoint = None
        self.model = self.model

    def train_one_epoch(self):
        self.model.train()
        self.epoch += 1
        tr_time = 0
        data_time = 0
        data_start = time.time()
        self.model.train()
        logs = {
            "train_rmse": paddle.zeros([1]),
            "train_nrmse": paddle.zeros([1]),
            "train_l1": paddle.zeros([1]),
            "train_l2": paddle.zeros([1]),
            "train_loss": paddle.zeros([1]),
        }
        steps = 0
        self.single_print(
            "train_loader_size", len(self.train_data_loader), len(self.train_dataset)
        )
        for batch_idx, data in enumerate(self.train_data_loader):
            steps += 1
            try:
                if len(data) == 3:
                    inp, _, mask = map(lambda x: x, data)
                    if sum(self.params.blur) > 0:
                        inp_blur = []
                        for _inp in inp:
                            sigma = random.uniform(*self.params.blur)
                            _kernel = min(
                                int((sigma * 4 + 1) / 2) * 2 + 1,
                                (_inp.shape[2] // 2) * 2 - 1,
                            )
                            if _kernel >= 2:
                                _inp = gaussian_blur(
                                    _inp, kernel_size=[_kernel, _kernel], sigma=sigma
                                )
                            inp_blur.append(_inp)
                        inp_blur = paddle.stack(inp_blur, axis=0)
                    else:
                        inp_blur = inp.detach().clone()
                else:
                    inp, _ = map(lambda x: x, data)
                    mask = None
                    inp_blur = inp.detach().clone()
            except:  # noqa
                print("DATA FAILLL", inp.shape)
                raise "s"
            if len(inp.shape) == 5:
                inp = rearrange(inp, "b t c h w -> t b c h w")
                inp_blur = rearrange(inp_blur, "b t c h w -> t b c h w")

            logwriter = LogWriter(logdir="./runs/data_effient_nopt")
            data_time += time.time() - data_start
            dtime = time.time() - data_start

            self.model.require_backward_grad_sync = (
                1 + batch_idx
            ) % self.params.accum_grad == 0
            with amp.auto_cast(self.params.enable_amp, dtype=self.mp_type):
                model_start = time.time()

                output = self.model(inp_blur, mask)

                if len(inp.shape) == 5:
                    labels = rearrange(
                        inp.permute(1, 2, 0, 3, 4),
                        "b c t (h p1) (w p2) -> b (t h w) (p1 p2 c)",
                        h=inp.shape[3] // self.model.patch_size,
                        w=inp.shape[4] // self.model.patch_size,
                        p1=self.model.patch_size,
                        p2=self.model.patch_size,
                    )
                    if mask is not None:
                        mask = mask.flatten(1).to(paddle.bool)
                        if mask.sum() == 0:
                            labels = labels[~mask]
                        else:
                            labels = labels[mask]
                    labels = labels.reshape(
                        inp.shape[1], -1, inp.shape[2] * self.model.patch_size**2
                    )
                    spatial_dims = tuple(range(output.ndim))[2:]

                    residuals = output - labels
                    inp_norm = 1e-7 + labels.pow(2).mean(spatial_dims, keepdim=True)
                    raw_loss = (residuals).pow(2).mean(
                        spatial_dims, keepdim=True
                    ) / inp_norm
                    loss = raw_loss.mean() / self.params.accum_grad

                elif len(inp.shape) == 4:
                    spatial_dims = tuple(range(output.ndim))[1:]
                    if mask is not None:
                        labels = inp * (1 - mask)
                        output = output * (1 - mask)
                    else:
                        labels = inp

                    residuals = output - labels
                    raw_loss = (residuals) ** 2
                    loss = raw_loss.sum() / output.shape[0] / self.params.accum_grad

                forward_end = time.time()
                forward_time = forward_end - model_start
                with paddle.no_grad():
                    logs["train_l1"] += F.l1_loss(output, labels)
                    log_nrmse = raw_loss.sqrt().mean()
                    logs["train_nrmse"] += log_nrmse
                    logs["train_rmse"] += (
                        residuals.pow(2).mean(spatial_dims).sqrt().mean()
                    )
                    logs["train_l2"] += l2_err(output, labels, spatial_dims)
                    logs["train_loss"] += loss
                self.gscaler.scale(loss).backward()
                backward_end = time.time()
                backward_time = backward_end - forward_end

                if self.debug_grad and self.model.require_backward_grad_sync:
                    with paddle.no_grad():
                        self.gscaler.unscale_(self.optimizer)
                        grad_diff = grad_norm(self.model.parameters())
                        porig = [p.clone() for p in self.model.parameters()]

                optimizer_step = 0
                if self.model.require_backward_grad_sync:
                    self.gscaler.unscale_(self.optimizer)
                    paddle.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    self.gscaler.step(self.optimizer)
                    self.gscaler.update()
                    if self.debug_grad:
                        if self.global_rank == 0:
                            pdiff = param_diff(self.model.parameters(), porig)
                            self.single_print(
                                "grad_norm",
                                grad_diff,
                                "last_step_size",
                                pdiff,
                                "loss",
                                loss.item(),
                                "data_shape",
                                inp.shape,
                            )
                    self.optimizer.clear_gradients(set_to_zero=False)
                    if self.scheduler is not None:
                        self.scheduler.step()
                    optimizer_step = time.time() - backward_end
                tr_time += time.time() - model_start
                if (
                    self.log_to_screen
                    and batch_idx % self.params.log_interval == 0
                    and self.global_rank == 0
                ):
                    print(
                        f"Epoch {self.epoch} Batch {batch_idx} Train Loss {log_nrmse.item()}"
                    )
                if self.log_to_screen:
                    logwriter.add_scalar(
                        "train_avg_loss",
                        value=log_nrmse.item(),
                        step=self.iters + steps - 1,
                    )
                    print(
                        "Total Times. Global step: {}, Batch: {}, Rank: {}, Data Shape: {}, Data time: {}, Forward: {}, Backward: {}, Optimizer: {}".format(
                            self.iters + steps - 1,
                            batch_idx,
                            self.global_rank,
                            inp.shape,
                            dtime,
                            forward_time,
                            backward_time,
                            optimizer_step,
                        )
                    )
                data_start = time.time()
        logs = {k: v / steps for k, v in logs.items()}
        if dist.is_initialized():
            for key in sorted(logs.keys()):
                dist.all_reduce(logs[key].detach())
                logs[key] = float(logs[key] / dist.get_world_size())

        self.iters += steps
        if self.global_rank == 0:
            logs["iters"] = self.iters
            logs["parameter norm"] = param_norm(self.model.parameters())
        self.single_print("all reduces executed!")

        return tr_time, data_time, logs

    def single_dset_val(self, subset, logs, cutoff=40):
        if self.params.use_ddp:
            temp_loader = paddle.io.DataLoader(
                subset,
                batch_size=self.params.batch_size,
                num_workers=self.params.num_data_workers,
            )
        else:
            temp_loader = paddle.io.DataLoader(
                subset,
                batch_size=self.params.batch_size,
                num_workers=self.params.num_data_workers,
                shuffle=True,
                drop_last=True,
            )
        count = 0
        for _, data in enumerate(temp_loader):
            if count > cutoff:
                del temp_loader
                break
            count += 1

            inp = data[0]
            if len(inp.shape) == 5:
                inp = rearrange(inp, "b t c h w -> t b c h w")
            elif len(inp.shape) == 4:
                b, c, h, w = inp.shape
            output = self.model(inp, None)
            residuals = output - inp
            if self.params.model_type == "fno":
                spatial_dims = tuple(range(output.ndim))[1:]
            elif self.params.model_type == "vmae":
                spatial_dims = tuple(range(output.ndim))[2:]

            nmse = (
                residuals.pow(2).mean(spatial_dims, keepdim=True)
                / (1e-7 + inp.pow(2).mean(spatial_dims, keepdim=True))
            ).sqrt()

            logs["valid_nrmse"] = (
                logs.get("valid_nrmse", 0) * (count - 1) + nmse.mean()
            ) / count
            logs["valid_rmse"] = (
                logs.get("valid_mse", 0) * (count - 1)
                + residuals.pow(2).mean(spatial_dims).sqrt().mean()
            ) / count
            logs["valid_l1"] = (
                logs.get("valid_l1", 0) * (count - 1) + residuals.abs().mean()
            ) / count
            logs["valid_l2"] = (
                logs.get("valid_l2", 0) * (count - 1)
                + l2_err(output, inp, spatial_dims)
            ) / count

        else:
            del temp_loader
        return logs

    def validate_one_epoch(self, full=False):
        """
        Validates - for each batch just use a small subset to make it easier.

        Note: need to split datasets for meaningful metrics, but TBD.
        """
        self.model.eval()
        if full:
            cutoff = 999999999999
        else:
            cutoff = 40
        self.single_print("STARTING VALIDATION!!!")
        with paddle.no_grad():
            with amp.auto_cast(enable=False, dtype=self.mp_type):
                logs = {}
                if hasattr(self.valid_dataset, "sub_dsets"):
                    for subset_group in self.valid_dataset.sub_dsets:
                        for subset in subset_group.get_per_file_dsets():
                            logs = self.single_dset_val(subset, logs, cutoff)
                else:
                    logs = self.single_dset_val(self.valid_dataset, logs, cutoff)

            self.single_print("DONE VALIDATING - NOW SYNCING")

            if dist.is_initialized():
                for key in sorted(logs.keys()):
                    dist.all_reduce(logs[key].detach())
                    logs[key] = float(logs[key].item() / dist.get_world_size())
                    if "rmse" in key:
                        logs[key] = logs[key]
            self.single_print("DONE SYNCING - NOW LOGGING")
        return logs

    def train(self):
        self.single_print("Starting Training Loop...")

        for epoch in range(self.startEpoch, self.params.max_epochs):
            if dist.is_initialized():
                self.train_sampler.set_epoch(epoch)
            start = time.time()

            tr_time, data_time, train_logs = self.train_one_epoch()

            valid_start = time.time()
            train_logs["time/train_time"] = valid_start - start
            train_logs["time/train_data_time"] = data_time
            train_logs["time/train_compute_time"] = tr_time

            if epoch == self.params.max_epochs - 1:
                valid_logs = self.validate_one_epoch(True)
            else:
                valid_logs = self.validate_one_epoch()
            train_logs.update(valid_logs)
            post_start = time.time()
            gc.collect()
            paddle.device.cuda.empty_cache()

            if self.global_rank == 0:
                if self.params.save_checkpoint:
                    self.save_checkpoint(self.params.checkpoint_path)
                if epoch % self.params.checkpoint_save_interval == 0:
                    self.save_checkpoint(self.params.checkpoint_path + f"_epoch{epoch}")

                cur_time = time.time()
                self.single_print(
                    f"Time for train {valid_start-start}. For valid: {post_start-valid_start}. For postprocessing:{cur_time-post_start}"
                )
                self.single_print(
                    "Time taken for epoch {} is {} sec".format(
                        epoch + 1, time.time() - start
                    )
                )
                self.single_print("Train loss: {}.".format(train_logs["train_nrmse"]))


def train(cfg: DictConfig):
    params = YParams(os.path.abspath(cfg.yaml_config), cfg.config)
    params.use_ddp = cfg.use_ddp
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if cfg.use_ddp:
        dist.init_process_group("nccl")

    device = f"gpu:{local_rank}" if paddle.device.cuda.device_count() >= 1 else "cpu"
    paddle.set_device(device)

    params["batch_size"] = int(params.batch_size // world_size)
    params["startEpoch"] = 0
    if cfg.sweep_id:
        jid = os.environ["SLURM_JOBID"]
        expDir = os.path.join(
            params.exp_dir, cfg.sweep_id, cfg.config, str(cfg.run_name), jid
        )
    else:
        expDir = os.path.join(params.exp_dir, cfg.config, str(cfg.run_name))

    params["old_exp_dir"] = expDir
    params["experiment_dir"] = os.path.abspath(expDir)
    params["checkpoint_path"] = os.path.join(expDir, "training_checkpoints/ckpt.tar")
    params["best_checkpoint_path"] = os.path.join(
        expDir, "training_checkpoints/best_ckpt.tar"
    )
    params["old_checkpoint_path"] = os.path.join(
        params.old_exp_dir, "training_checkpoints/best_ckpt.tar"
    )

    if global_rank == 0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir)
            os.makedirs(os.path.join(expDir, "training_checkpoints/"))
    params["resuming"] = True if os.path.isfile(params.checkpoint_path) else False

    params["name"] = str(cfg.run_name)
    if global_rank == 0:
        logging_utils.log_to_file(
            logger_name=None, log_filename=os.path.join(expDir, "out.log")
        )
        logging_utils.log_versions()
        params.log()

    if global_rank == 0:
        logging_utils.log_to_file(
            logger_name=None, log_filename=os.path.join(expDir, "out.log")
        )
        logging_utils.log_versions()
        params.log()

    params["log_to_wandb"] = (global_rank == 0) and params["log_to_wandb"]
    params["log_to_screen"] = (global_rank == 0) and params["log_to_screen"]

    if global_rank == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(os.path.join(expDir, "hyperparams.yaml"), "w") as hpfile:
            yaml.dump(hparams, hpfile)
    trainer = Trainer(params, global_rank, local_rank, device, sweep_id=cfg.sweep_id)
    if cfg.sweep_id and trainer.global_rank == 0:
        print(cfg.sweep_id, trainer.params.entity, trainer.params.project)
    else:
        trainer.train()
    if params.log_to_screen:
        print("DONE ---- rank %d" % global_rank)


@paddle.no_grad()
def get_pred(cfg):
    with open(cfg.eval_config, "r") as stream:
        config = yaml.load(stream, yaml.FullLoader)
    if cfg.ckpt_path:
        save_dir = os.path.join("/".join(cfg.ckpt_path.split("/")[:-1]), "results_icl")
    else:
        basedir = os.path.join("exp", config["log"]["logdir"])
        save_dir = os.path.join(basedir, "results_icl")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, "fno-prediction-demo%d.pt" % (cfg.num_demos if cfg.num_demos else 0)
    )
    if paddle.device.cuda.device_count() >= 1:
        paddle.set_device("gpu")
    else:
        paddle.set_device("cpu")

    params = Namespace(**config["default"])
    if not hasattr(params, "n_demos"):
        params.n_demos = 0
    if "batch_size" in config:
        params.local_valid_batch_size = config["batch_size"]
    else:
        params.local_valid_batch_size = 1
    dataloader, dataset, sampler = PoisHelmDatasetLoader(
        params, params.test_path, dist.is_initialized(), train=False
    )
    if cfg.num_demos is not None and cfg.num_demos != 0:
        params.subsample = 1
        params.local_valid_batch_size = cfg.num_demos
        dataloader_icl, dataset_icl, _ = PoisHelmDatasetLoader(
            params, params.train_path, dist.is_initialized(), train=False
        )
        input_demos, target_demos = next(iter(dataloader_icl))
        input_demos = input_demos
        target_demos = target_demos

    model = build_fno(params)

    if cfg.ckpt_path:
        checkpoint = paddle.load(cfg.ckpt_path)
        try:
            model.set_state_dict(checkpoint["model_state"])
        except:  # noqa
            new_state_dict = OrderedDict()
            for key, val in checkpoint["model_state"].items():
                name = key
                if "module" in name:
                    name = name[7:]
                new_state_dict[name] = val
            state = model.state_dict()
            pretrained_dict = {
                k: v
                for k, v in new_state_dict.items()
                if k in state and state[k].size() == new_state_dict[k].size()
            }
            state.update(pretrained_dict)

            unload_keys = [k for k in new_state_dict.keys() if k not in pretrained_dict]
            if len(unload_keys) > 0:
                import warnings

                warnings.warn(
                    "Warning: unload keys during restoring checkpoint: %s"
                    % (str(unload_keys))
                )

    model.eval()
    truth_list = []
    pred_list = []
    losses = []
    losses_normalized = []
    pbar = tqdm(dataloader, total=len(dataloader))
    for inputs, targets in pbar:
        inputs, targets = inputs, targets
        if cfg.num_demos is None or cfg.num_demos == 0:
            u = model(inputs)
        else:
            model.target = targets
            u = model.forward_icl(inputs, input_demos, target_demos, use_tqdm=cfg.tqdm)

        data_loss = l2_err(u.detach(), targets.detach())
        losses.append(data_loss.item())
        data_loss_normalized = l2_err(
            u.detach() / paddle.abs(u).max(),
            targets.detach() / paddle.abs(targets).max(),
        )
        losses_normalized.append(data_loss_normalized.item())
        truth_list.append(targets.cpu())
        pred_list.append(u.cpu())

    slope, intercept, r, p, se = linregress(
        paddle.concat(pred_list, axis=0).view([-1]).numpy(),
        paddle.concat(truth_list, axis=0).view([-1]).numpy(),
    )
    print(
        "RMSE:",
        np.mean(losses),
        "RMSE (normalized)",
        np.mean(losses_normalized),
        "R2:",
        r,
        "Slope:",
        slope,
    )
    truth_arr = paddle.concat(truth_list, axis=0)
    pred_arr = paddle.concat(pred_list, axis=0)
    paddle.save(
        {
            "truth": truth_arr,
            "pred": pred_arr,
            "rmse": np.mean(losses),
            "rmse_normalized": np.mean(losses_normalized),
            "r2": r,
            "slope": slope,
        },
        save_path,
    )


def inference(cfg: DictConfig):
    get_pred(cfg)


@hydra.main(
    version_base=None, config_path="./config", config_name="data_efficient_nopt.yaml"
)
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "infer":
        inference(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'infer'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
