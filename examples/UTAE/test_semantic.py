"""
Main script for semantic experiments (Paddle Version)
Converted to PaddlePaddle
"""
import argparse
import json
import os
import pprint
import time

import numpy as np
import paddle
import paddle.nn as nn
from src import model_utils_paddle as model_utils
from src.dataset import PASTIS_Dataset
from src.learning.miou import IoU
from src.utils import pad_collate

parser = argparse.ArgumentParser()
# Model parameters
parser.add_argument(
    "--model",
    default="utae",
    type=str,
    help="Type of architecture to use. Can be one of: (utae/unet3d/fpn/convlstm/convgru/uconvlstm/buconvlstm)",
)
## U-TAE Hyperparameters
parser.add_argument("--encoder_widths", default="[64,64,64,128]", type=str)
parser.add_argument("--decoder_widths", default="[32,32,64,128]", type=str)
parser.add_argument("--out_conv", default="[32, 20]")
parser.add_argument("--str_conv_k", default=4, type=int)
parser.add_argument("--str_conv_s", default=2, type=int)
parser.add_argument("--str_conv_p", default=1, type=int)
parser.add_argument("--agg_mode", default="att_group", type=str)
parser.add_argument("--encoder_norm", default="group", type=str)
parser.add_argument("--n_head", default=16, type=int)
parser.add_argument("--d_model", default=256, type=int)
parser.add_argument("--d_k", default=4, type=int)

# Set-up parameters
parser.add_argument(
    "--dataset_folder",
    default="",
    type=str,
    help="Path to the folder where the results are saved.",
)
parser.add_argument(
    "--res_dir",
    default="./results",
    help="Path to the folder where the results should be stored",
)
parser.add_argument(
    "--num_workers", default=8, type=int, help="Number of data loading workers"
)
parser.add_argument("--rdm_seed", default=1, type=int, help="Random seed")
parser.add_argument(
    "--device",
    default="gpu",
    type=str,
    help="Name of device to use for tensor computations (gpu/cpu)",
)
parser.add_argument(
    "--display_step",
    default=50,
    type=int,
    help="Interval in batches between display of training metrics",
)
parser.add_argument(
    "--cache",
    dest="cache",
    action="store_true",
    help="If specified, the whole dataset is kept in RAM",
)
# Training parameters
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs per fold")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--mono_date", default=None, type=str)
parser.add_argument("--ref_date", default="2018-09-01", type=str)
parser.add_argument(
    "--fold",
    default=None,
    type=int,
    help="Do only one of the five fold (between 1 and 5)",
)
parser.add_argument("--num_classes", default=20, type=int)
parser.add_argument("--ignore_index", default=-1, type=int)
parser.add_argument("--pad_value", default=0, type=float)
parser.add_argument("--padding_mode", default="reflect", type=str)
parser.add_argument(
    "--val_every",
    default=1,
    type=int,
    help="Interval in epochs between two validation steps.",
)


def recursive_todevice(x, device):
    if isinstance(x, paddle.Tensor):
        return x.cuda() if device == "gpu" else x.cpu()
    else:
        return [recursive_todevice(c, device) for c in x]


def prepare_output(config):
    os.makedirs(config.res_dir, exist_ok=True)
    for fold in range(1, 6):
        os.makedirs(os.path.join(config.res_dir, "Fold_{}".format(fold)), exist_ok=True)


def checkpoint(log, config):
    with open(os.path.join(config.res_dir, "trainlog.json"), "w") as outfile:
        json.dump(log, outfile, indent=4)


def save_results(metrics, config):
    with open(os.path.join(config.res_dir, "test_metrics.json"), "w") as outfile:
        json.dump(metrics, outfile, indent=4)


def overall_performance(cm):
    """
    Computes overall accuracy and mean IoU from confusion matrix
    """
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)

    # Per-class IoU
    ious = []
    for i in range(cm.shape[0]):
        intersection = cm[i, i]
        union = np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i]
        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(0)

    mean_iou = np.mean(ious)
    return accuracy, mean_iou


def iterate(
    model, data_loader, criterion, config, optimizer=None, mode="train", device="gpu"
):
    loss_meter = 0
    iou_meter = IoU(
        num_classes=config.num_classes,
        ignore_index=config.ignore_index,
        cm_device="cpu" if device == "cpu" else "gpu",
    )

    t_start = time.time()
    for i, batch in enumerate(data_loader):
        if device == "gpu":
            batch = recursive_todevice(batch, "gpu")

        (x, dates), y = batch

        if mode != "train":
            with paddle.no_grad():
                out = model(x, batch_positions=dates)
        else:
            out = model(x, batch_positions=dates)

        # Paddle CrossEntropyLoss expects different format for 2D case
        # Need to reshape: out=[B*H*W, C], y=[B*H*W]
        B, C, H, W = out.shape
        out_reshaped = out.transpose([0, 2, 3, 1]).reshape([-1, C])  # [B*H*W, C]
        y_reshaped = y.reshape([-1])  # [B*H*W]

        loss = criterion(out_reshaped, y_reshaped.astype("int64"))

        if mode == "train":
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()

        pred = nn.functional.softmax(out, axis=1).argmax(axis=1)
        iou_meter.add(pred, y)
        loss_meter += loss.item()

        if (i + 1) % config.display_step == 0:
            miou, acc = iou_meter.get_miou_acc()
            print(
                f"{mode} - Step [{i+1}/{len(data_loader)}] "
                f"Loss: {loss_meter/(i+1):.4f} "
                f"Acc: {acc:.3f} "
                f"mIoU: {miou:.3f}"
            )

    miou, acc = iou_meter.get_miou_acc()
    t_end = time.time()

    return loss_meter / len(data_loader), acc, miou, t_end - t_start


def save_model(model, config, fold, epoch, miou):
    model_path = os.path.join(
        config.res_dir, f"Fold_{fold}", f"model_epoch_{epoch}_miou_{miou:.3f}.pdparams"
    )
    paddle.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def main(config):
    np.random.seed(config.rdm_seed)
    paddle.seed(config.rdm_seed)

    prepare_output(config)

    # Set device
    if config.device == "gpu" and paddle.is_compiled_with_cuda():
        paddle.device.set_device("gpu")
    else:
        paddle.device.set_device("cpu")
        config.device = "cpu"

    # Model parameters

    folds = [config.fold] if config.fold is not None else range(1, 6)

    overall_results = {}

    for fold in folds:
        print(f"Starting fold {fold}")

        # Datasets
        dt_train = PASTIS_Dataset(
            folder=config.dataset_folder,
            norm=True,
            target="semantic",
            folds=[
                f for f in range(1, 6) if f != fold
            ],  # Use all folds except current one for training
            cache=config.cache,
        )

        dt_val = PASTIS_Dataset(
            folder=config.dataset_folder,
            norm=True,
            target="semantic",
            folds=[fold],  # Use current fold for validation
            cache=config.cache,
        )

        print(f"Train samples: {len(dt_train)}, Val samples: {len(dt_val)}")

        # Data loaders
        collate_fn = lambda x: pad_collate(x, pad_value=config.pad_value)

        train_loader = paddle.io.DataLoader(
            dt_train,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
            collate_fn=collate_fn,
        )

        val_loader = paddle.io.DataLoader(
            dt_val,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # Model
        model = model_utils.get_model(config, mode="semantic")

        print(
            f"Model {config.model} - {model_utils.get_ntrainparams(model)} trainable parameters"
        )

        # Optimizer & criterion
        optimizer = paddle.optimizer.Adam(
            parameters=model.parameters(), learning_rate=config.lr
        )
        criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index)

        # Training
        trainlog = {}
        best_miou = 0

        for epoch in range(1, config.epochs + 1):
            print(f"Epoch {epoch}/{config.epochs}")

            # Training
            model.train()
            train_loss, train_acc, train_miou, train_time = iterate(
                model,
                train_loader,
                criterion,
                config,
                optimizer,
                "train",
                config.device,
            )

            # Validation
            if epoch % config.val_every == 0:
                model.eval()
                val_loss, val_acc, val_miou, val_time = iterate(
                    model,
                    val_loader,
                    criterion,
                    config,
                    mode="val",
                    device=config.device,
                )

                print(
                    f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.3f}, mIoU: {train_miou:.3f}"
                )
                print(
                    f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.3f}, mIoU: {val_miou:.3f}"
                )

                # Save best model
                if val_miou > best_miou:
                    best_miou = val_miou
                    save_model(model, config, fold, epoch, val_miou)

                # Log
                trainlog[epoch] = {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_miou": train_miou,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_miou": val_miou,
                }

            checkpoint(trainlog, config)

        overall_results[f"fold_{fold}"] = {
            "best_miou": best_miou,
            "final_train_miou": train_miou,
        }

    # Average results across folds
    if len(folds) > 1:
        mean_miou = np.mean(
            [overall_results[f"fold_{fold}"]["best_miou"] for fold in folds]
        )
        print(f"Average mIoU across {len(folds)} folds: {mean_miou:.3f}")
        overall_results["mean_miou"] = mean_miou

    save_results(overall_results, config)
    print("Training completed!")


if __name__ == "__main__":
    config = parser.parse_args()

    # Print configuration
    pprint.pprint(config.__dict__)

    main(config)
