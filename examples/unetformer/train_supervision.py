import argparse
import os
import random
from pathlib import Path

import numpy as np
import paddle
from paddle_utils import PaddleFlag
from tools.cfg import py2cfg
from tools.metric import Evaluator


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    paddle.seed(seed=seed)
    PaddleFlag.cudnn_deterministic = True
    PaddleFlag.cudnn_benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


class Supervision_Train(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.loss = config.loss
        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)
        self.log_history = []
        self.prog_bar_metrics = []

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, config):
        model = cls(config)
        state_dict = paddle.load(checkpoint_path)
        model.set_state_dict(state_dict)
        print(f"Loaded model weights from {checkpoint_path}")
        return model

    def forward(self, x):
        seg_pre = self.net(x)
        return seg_pre

    def training_step(self, batch, batch_idx):
        img, mask = batch["img"], batch["gt_semantic_seg"]
        prediction = self.net(img)
        loss = self.loss(prediction, mask)
        if self.config.use_aux_loss:
            pre_mask = paddle.nn.functional.softmax(prediction[0], axis=1)
        else:
            pre_mask = paddle.nn.functional.softmax(prediction, axis=1)
        pre_mask = pre_mask.argmax(axis=1)
        for i in range(tuple(mask.shape)[0]):
            self.metrics_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].numpy())
        return {"loss": loss}

    def on_train_epoch_end(self):
        if "vaihingen" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif "potsdam" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif "whubuilding" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif "massbuilding" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif "cropland" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_train.F1())
        OA = np.nanmean(self.metrics_train.OA())
        iou_per_class = self.metrics_train.Intersection_over_Union()
        eval_value = {"mIoU": mIoU, "F1": F1, "OA": OA}
        print("train:", eval_value)
        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        self.metrics_train.reset()
        log_dict = {"train_mIoU": mIoU, "train_F1": F1, "train_OA": OA}
        print(f"Logging: {log_dict}")
        self.log_dict(log_dict, prog_bar=True)
        return log_dict

    def validation_step(self, batch, batch_idx):
        img, mask = batch["img"], batch["gt_semantic_seg"]
        prediction = self.forward(img)
        pre_mask = paddle.nn.functional.softmax(prediction, axis=1)
        pre_mask = pre_mask.argmax(axis=1)
        for i in range(tuple(mask.shape)[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())
        loss_val = self.loss(prediction, mask)
        return {"loss_val": loss_val}

    def on_validation_epoch_end(self):
        if "vaihingen" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif "potsdam" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif "whubuilding" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif "massbuilding" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif "cropland" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_val.F1())
        OA = np.nanmean(self.metrics_val.OA())
        iou_per_class = self.metrics_val.Intersection_over_Union()
        eval_value = {"mIoU": mIoU, "F1": F1, "OA": OA}
        print("val:", eval_value)
        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)
        self.metrics_val.reset()
        log_dict = {"val_mIoU": mIoU, "val_F1": F1, "val_OA": OA}
        self.log_dict(log_dict, prog_bar=True)
        return log_dict

    def configure_optimizers(self):
        return self.config.optimizer, self.config.lr_scheduler

    def train_dataloader(self):
        return self.config.train_loader

    def val_dataloader(self):
        return self.config.val_loader

    def log_dict(self, log_dict, prog_bar=False):
        self.log_history.append(log_dict)
        if prog_bar:
            self.prog_bar_metrics = log_dict
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])
            print(f"[Metrics] {metrics_str}")


class ModelCheckpoint:
    def __init__(self, save_top_k, monitor, save_last, mode, dirpath, filename):
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.save_last = save_last
        self.monitor_mode = mode
        self.dirpath = dirpath
        self.filename = filename
        self.best_metric = -float("inf") if mode == "max" else float("inf")
        self.best_path = ""
        self.current_epoch = 0

    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

    def on_validation_epoch_end(self, trainer, model, val_log):
        current_metric = val_log[self.monitor]
        save_best = False
        if self.monitor_mode == "max" and current_metric > self.best_metric:
            self.best_metric = current_metric
            save_best = True
        elif self.monitor_mode == "min" and current_metric < self.best_metric:
            self.best_metric = current_metric
            save_best = True
        if save_best:
            self._remove_old_checkpoints()

            self.best_path = os.path.join(
                self.dirpath, f"{self.filename}_epoch{self.current_epoch}_best.pdparams"
            )
            paddle.save(model.state_dict(), self.best_path)
            print(f"Saved best model to {self.best_path}")

        if self.save_last:
            last_path = os.path.join(self.dirpath, "last.pdparams")
            paddle.save(model.state_dict(), last_path)

    def _remove_old_checkpoints(self):
        if self.save_top_k <= 0:
            return

        all_files = [
            f
            for f in os.listdir(self.dirpath)
            if f.endswith(".pdparams") and self.filename in f
        ]

        all_files.sort(
            key=lambda x: os.path.getmtime(os.path.join(self.dirpath, x)), reverse=True
        )

        while len(all_files) >= self.save_top_k:
            file_to_remove = all_files.pop()
            os.remove(os.path.join(self.dirpath, file_to_remove))
            print(f"Removed old checkpoint: {file_to_remove}")


class CSVLogger:
    def __init__(self, save_dir, name):
        self.save_dir = os.path.join(save_dir, name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_file = os.path.join(self.save_dir, "metrics.csv")

        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write(
                    "epoch,train_loss,train_mIoU,train_F1,train_OA,val_loss,val_mIoU,val_F1,val_OA\n"
                )

    def log_metrics(self, epoch, train_log, val_log):
        with open(self.log_file, "a") as f:
            line = f"{epoch},{train_log.get('loss', '')},{train_log.get('train_mIoU', '')},{train_log.get('train_F1', '')},{train_log.get('train_OA', '')},"
            line += f"{val_log.get('loss_val', '')},{val_log.get('val_mIoU', '')},{val_log.get('val_F1', '')},{val_log.get('val_OA', '')}\n"
            f.write(line)


def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=config.save_top_k,
        monitor=config.monitor,
        save_last=config.save_last,
        mode=config.monitor_mode,
        dirpath=config.weights_path,
        filename=config.weights_name,
    )

    logger = CSVLogger("lightning_logs", name=config.log_name)

    model = Supervision_Train(config)

    if config.pretrained_ckpt_path:
        state_dict = paddle.load(config.pretrained_ckpt_path)
        model.set_state_dict(state_dict)

    paddle.set_device("gpu")

    optimizer, lr_scheduler = model.configure_optimizers()

    train_loader = model.train_dataloader()
    val_loader = model.val_dataloader()

    for epoch in range(config.max_epoch):
        print(f"Epoch {epoch+1}/{config.max_epoch}")
        model.train()
        train_losses = []
        for batch_idx, batch in enumerate(train_loader):
            output = model.training_step(batch, batch_idx)
            loss = output["loss"]
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            if batch_idx % 10 == 0:
                print(
                    f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
                )

        train_log = model.on_train_epoch_end()
        train_log["loss"] = np.mean(train_losses)
        if (epoch + 1) % config.check_val_every_n_epoch == 0:
            model.eval()
            val_losses = []
            for batch_idx, batch in enumerate(val_loader):
                output = model.validation_step(batch, batch_idx)
                val_losses.append(output["loss_val"].item())
            val_log = model.on_validation_epoch_end()
            val_log["loss_val"] = np.mean(val_losses)
            checkpoint_callback.on_validation_epoch_end(None, model, val_log)
            logger.log_metrics(epoch, train_log, val_log)
        if lr_scheduler:
            lr_scheduler.step()
        if config.resume_ckpt_path and epoch == 0:
            state = paddle.load(config.resume_ckpt_path)
            model.set_state_dict(state["model_state_dict"])
            optimizer.set_state_dict(state["optimizer_state_dict"])
            if lr_scheduler and "lr_scheduler_state_dict" in state:
                lr_scheduler.set_state_dict(state["lr_scheduler_state_dict"])
            print(f"Resumed training from checkpoint: {config.resume_ckpt_path}")


if __name__ == "__main__":
    main()
