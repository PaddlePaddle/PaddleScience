import argparse
import json
import os
import pprint
import time
from typing import Tuple

import numpy as np
import paddle
import paddle.nn as nn
from src import model_utils
from src import utils
from src.dataset import PASTIS_Dataset

# ---------- Args ----------
parser = argparse.ArgumentParser()

# Model & arch
parser.add_argument(
    "--model",
    default="utae",
    type=str,
    help="utae/unet3d/fpn/convlstm/convgru/uconvlstm/buconvlstm",
)
parser.add_argument("--encoder_widths", default="[64,64,64,128]", type=str)
parser.add_argument("--decoder_widths", default="[32,32,64,128]", type=str)
parser.add_argument("--out_conv", default="[32, 20]", type=str)
parser.add_argument("--str_conv_k", default=4, type=int)
parser.add_argument("--str_conv_s", default=2, type=int)
parser.add_argument("--str_conv_p", default=1, type=int)
parser.add_argument("--agg_mode", default="att_group", type=str)
parser.add_argument("--encoder_norm", default="group", type=str)
parser.add_argument("--n_head", default=16, type=int)
parser.add_argument("--d_model", default=256, type=int)
parser.add_argument("--d_k", default=4, type=int)
parser.add_argument("--padding_mode", default="reflect", type=str)

# Data / runtime
parser.add_argument("--dataset_folder", default="", type=str, help="PASTIS root")
parser.add_argument("--res_dir", default="./results", type=str, help="Output dir")
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--rdm_seed", default=1, type=int)
parser.add_argument("--device", default="gpu", type=str, help="gpu/cpu")
parser.add_argument("--display_step", default=50, type=int)
parser.add_argument("--cache", action="store_true", help="Keep dataset in RAM")
parser.add_argument("--mono_date", default=None, type=str)
parser.add_argument("--ref_date", default="2018-09-01", type=str)

# Training
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument(
    "--lr_decay", default="none", type=str, choices=["none", "step", "cosine"]
)
parser.add_argument("--lr_step_size", default=20, type=int)
parser.add_argument("--lr_gamma", default=0.5, type=float)
parser.add_argument("--grad_clip", default=0.0, type=float, help="0 to disable")
parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
parser.add_argument(
    "--early_stopping", default=0, type=int, help="Patience on val mIoU (0=disabled)"
)
parser.add_argument("--fold", default=None, type=int, help="1..5 for single fold")
parser.add_argument("--val_every", default=1, type=int)

# Labels
parser.add_argument("--num_classes", default=20, type=int)
parser.add_argument("--ignore_index", default=-1, type=int)
parser.add_argument("--pad_value", default=0, type=float)


# ---------- Helpers ----------
def set_seed(seed: int):
    np.random.seed(seed)
    paddle.seed(seed)


def prepare_output(config):
    os.makedirs(config.res_dir, exist_ok=True)
    for f in range(1, 6):
        os.makedirs(os.path.join(config.res_dir, f"Fold_{f}"), exist_ok=True)


def save_conf(config):
    path = os.path.join(config.res_dir, "conf.json")
    with open(path, "w") as f:
        json.dump(vars(config), f, indent=2)
    # also copy into each fold directory for convenience
    for f in range(1, 6):
        with open(os.path.join(config.res_dir, f"Fold_{f}", "conf.json"), "w") as g:
            json.dump(vars(config), g, indent=2)


def save_trainlog(trainlog: dict, config):
    with open(os.path.join(config.res_dir, "trainlog.json"), "w") as f:
        json.dump(trainlog, f, indent=2)


def save_metrics(metrics: dict, config):
    with open(os.path.join(config.res_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def get_iou_class():
    try:
        from src.learning.miou import IoU
    except Exception:
        from src.miou import IoU
    return IoU


def iterate(
    model, loader, criterion, device="gpu", optimizer=None, display_step=50
) -> Tuple[float, float, float]:
    IoU = get_iou_class()
    iou_meter = IoU(
        num_classes=config.num_classes,
        ignore_index=config.ignore_index,
        cm_device="cpu",
    )
    loss_sum = 0.0
    n_batches = 0

    model.train() if optimizer is not None else model.eval()
    scaler = None
    if optimizer is not None and config.amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

    for i, batch in enumerate(loader):
        (x, dates), y = batch
        if optimizer is not None:
            if config.amp:
                with paddle.amp.auto_cast():
                    out = model(x, batch_positions=dates)
                    B, C, H, W = out.shape
                    out_reshaped = out.transpose([0, 2, 3, 1]).reshape([-1, C])
                    y_reshaped = y.reshape([-1]).astype("int64")
                    loss = criterion(out_reshaped, y_reshaped)
                scaler.scale(loss).backward()
                scaler.minimize(optimizer, loss)
                optimizer.clear_grad()
            else:
                out = model(x, batch_positions=dates)
                B, C, H, W = out.shape
                out_reshaped = out.transpose([0, 2, 3, 1]).reshape([-1, C])
                y_reshaped = y.reshape([-1]).astype("int64")
                loss = criterion(out_reshaped, y_reshaped)
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
        else:
            with paddle.no_grad():
                out = model(x, batch_positions=dates)
                B, C, H, W = out.shape
                out_reshaped = out.transpose([0, 2, 3, 1]).reshape([-1, C])
                y_reshaped = y.reshape([-1]).astype("int64")
                loss = criterion(out_reshaped, y_reshaped)

        pred = nn.functional.softmax(out, axis=1).argmax(axis=1)
        iou_meter.add(pred, y)
        loss_sum += float(loss.numpy())
        n_batches += 1

        if (i + 1) % display_step == 0:
            miou, acc = iou_meter.get_miou_acc()
            mode = "train" if optimizer is not None else "val"
            print(
                f"{mode} step {i+1}/{len(loader)}  loss:{loss_sum/n_batches:.4f}  acc:{acc:.3f}  mIoU:{miou:.3f}"
            )

    miou, acc = iou_meter.get_miou_acc()
    return loss_sum / max(1, n_batches), float(acc), float(miou)


def main(config):
    set_seed(config.rdm_seed)
    if config.device == "gpu" and paddle.is_compiled_with_cuda():
        paddle.set_device("gpu")
    else:
        paddle.set_device("cpu")
        config.device = "cpu"

    prepare_output(config)
    save_conf(config)

    folds = [config.fold] if config.fold is not None else [1, 2, 3, 4, 5]
    trainlog = {}

    for fold in folds:
        print(f"===== Fold {fold} =====")
        # Data
        dt_train = PASTIS_Dataset(
            folder=config.dataset_folder,
            norm=True,
            target="semantic",
            folds=[f for f in range(1, 6) if f != fold],
            cache=config.cache,
        )
        dt_val = PASTIS_Dataset(
            folder=config.dataset_folder,
            norm=True,
            target="semantic",
            folds=[fold],
            cache=config.cache,
        )

        collate_fn = lambda x: utils.pad_collate(x, pad_value=config.pad_value)
        train_loader = paddle.io.DataLoader(
            dt_train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
        )
        val_loader = paddle.io.DataLoader(
            dt_val,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
        )

        # Model
        model = model_utils.get_model(config, mode="semantic")
        print(model)
        print("TOTAL TRAINABLE PARAMETERS :", model_utils.get_ntrainparams(model))

        # Optimizer, grad clip
        grad_clip = None
        if config.grad_clip and config.grad_clip > 0:
            grad_clip = paddle.nn.ClipGradByGlobalNorm(config.grad_clip)
        optimizer = paddle.optimizer.Adam(
            learning_rate=config.lr, parameters=model.parameters(), grad_clip=grad_clip
        )
        # LR scheduler
        if config.lr_decay == "step":
            sched = paddle.optimizer.lr.StepDecay(
                learning_rate=config.lr,
                step_size=config.lr_step_size,
                gamma=config.lr_gamma,
            )
            optimizer.set_lr_scheduler(sched)
        elif config.lr_decay == "cosine":
            sched = paddle.optimizer.lr.CosineAnnealingDecay(
                learning_rate=config.lr, T_max=config.epochs
            )
            optimizer.set_lr_scheduler(sched)
        else:
            sched = None

        criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index)

        # Train
        best_miou = -1.0
        epochs_no_improve = 0

        for epoch in range(1, config.epochs + 1):
            t0 = time.time()
            model.train()
            tr_loss, tr_acc, tr_miou = iterate(
                model,
                train_loader,
                criterion,
                device=config.device,
                optimizer=optimizer,
                display_step=config.display_step,
            )
            model.eval()
            va_loss, va_acc, va_miou = iterate(
                model,
                val_loader,
                criterion,
                device=config.device,
                optimizer=None,
                display_step=config.display_step,
            )
            dt = time.time() - t0

            # Scheduler step (if using)
            if sched is not None:
                sched.step()

            print(
                f"Epoch {epoch}/{config.epochs}  "
                f"train: loss {tr_loss:.4f} acc {tr_acc:.3f} miou {tr_miou:.3f} | "
                f"val: loss {va_loss:.4f} acc {va_acc:.3f} miou {va_miou:.3f}  "
                f"({dt/60:.1f} min)"
            )

            # Save best
            fold_dir = os.path.join(config.res_dir, f"Fold_{fold}")
            os.makedirs(fold_dir, exist_ok=True)
            ckpt_path = os.path.join(
                fold_dir, f"model_epoch_{epoch}_miou_{va_miou:.3f}.pdparams"
            )
            paddle.save(model.state_dict(), ckpt_path)

            if va_miou > best_miou:
                best_miou = va_miou
                # also update alias
                alias = os.path.join(fold_dir, "model.pdparams")
                paddle.save(model.state_dict(), alias)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Log
            trainlog[f"fold{fold}_epoch{epoch}"] = {
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "train_miou": tr_miou,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "val_miou": va_miou,
                "best_miou": best_miou,
                "lr": float(optimizer.get_lr())
                if hasattr(optimizer, "get_lr")
                else config.lr,
                "time_min": dt / 60.0,
            }
            save_trainlog(trainlog, config)

            # Early stopping
            if config.early_stopping and epochs_no_improve >= config.early_stopping:
                print(f"Early stopping triggered (patience={config.early_stopping}).")
                break

        print(f"[Fold {fold}] best mIoU = {best_miou:.3f}")

    print("Training done.")


if __name__ == "__main__":
    config = parser.parse_args()
    # device
    if config.device not in ["gpu", "cpu"]:
        config.device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
    # Print configuration
    pprint.pprint(vars(config))
    main(config)
