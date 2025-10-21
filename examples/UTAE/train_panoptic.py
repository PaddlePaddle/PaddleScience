"""
Main script for panoptic experiments (Paddle Version)
Converted to PaddlePaddle
"""
import argparse
import json
import os
import pprint
import time

import numpy as np
import paddle
from src import model_utils as model_utils
from src.dataset import PASTIS_Dataset
from src.learning.weight_init import weight_init
from src.model_utils import get_ntrainparams
from src.panoptic.metrics import PanopticMeter
from src.panoptic.paps_loss import PaPsLoss
from src.utils import pad_collate

parser = argparse.ArgumentParser()
# PaPs Parameters
## Architecture Hyperparameters
parser.add_argument("--shape_size", default=16, type=int, help="Shape size for PaPs")
parser.add_argument(
    "--no_mask_conv",
    dest="mask_conv",
    action="store_false",
    help="With this flag no residual CNN is used after combination of global saliency and local shape.",
)
parser.add_argument(
    "--backbone",
    default="utae",
    type=str,
    help="Backbone encoder for PaPs (utae or uconvlstm)",
)

## Losses & metrics
parser.add_argument(
    "--l_center", default=1, type=float, help="Coefficient for centerness loss"
)
parser.add_argument("--l_size", default=1, type=float, help="Coefficient for size loss")
parser.add_argument(
    "--l_shape", default=0, type=float, help="Coefficient for shape loss"
)
parser.add_argument(
    "--l_class", default=1, type=float, help="Coefficient for class loss"
)
parser.add_argument(
    "--beta", default=4, type=float, help="Beta parameter for centerness loss"
)
parser.add_argument(
    "--no_autotune",
    dest="autotune",
    action="store_false",
    help="If this flag is used the confidence threshold for the pseudo-nms will NOT be tuned automatically on the validation set",
)
parser.add_argument(
    "--no_supmax",
    dest="supmax",
    action="store_false",
    help="If this flag is used, ALL local maxima are supervised (and not just the more confident center per ground truth object)",
)
parser.add_argument(
    "--warmup",
    default=5,
    type=int,
    help="Number of epochs to do with only the centerness loss as supervision.",
)
parser.add_argument(
    "--val_metrics_only",
    dest="val_metrics_only",
    action="store_true",
    help="If true, panoptic metrics are computed only on validation and test epochs.",
)
parser.add_argument(
    "--val_every",
    default=5,
    type=int,
    help="Interval in epochs between two validation steps.",
)
parser.add_argument(
    "--val_after",
    default=0,
    type=int,
    help="Do validation only after that many epochs.",
)

## Thresholds
parser.add_argument(
    "--min_remain",
    default=0.5,
    type=float,
    help="Minimum remain fraction for the pseudo-nms.",
)
parser.add_argument(
    "--mask_threshold",
    default=0.4,
    type=float,
    help="Binary threshold for instance masks",
)
parser.add_argument(
    "--min_confidence",
    default=0.2,
    type=float,
    help="Minimum confidence threshold for pseudo-nms",
)

# U-TAE Hyperparameters (if using U-TAE backbone)
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
    default="/home/aistudio/PASTIS",
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
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
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


def get_model(config):
    """Create PaPs model with specified backbone"""
    if config.backbone == "utae":
        from src.backbones.utae import UTAE
        from src.panoptic.paps import PaPs

        encoder = UTAE(
            input_dim=10,  # PASTIS has 10 spectral bands
            encoder_widths=eval(config.encoder_widths),
            decoder_widths=eval(config.decoder_widths),
            out_conv=eval(config.out_conv),
            str_conv_k=config.str_conv_k,
            str_conv_s=config.str_conv_s,
            str_conv_p=config.str_conv_p,
            agg_mode=config.agg_mode,
            encoder_norm=config.encoder_norm,
            n_head=config.n_head,
            d_model=config.d_model,
            d_k=config.d_k,
            encoder=True,  # Important: set to True for PaPs
            return_maps=True,  # Important: return feature maps
            pad_value=config.pad_value,
            padding_mode=config.padding_mode,
        )

        model = PaPs(
            encoder=encoder,
            num_classes=config.num_classes,
            shape_size=config.shape_size,
            mask_conv=config.mask_conv,
            min_confidence=config.min_confidence,
            min_remain=config.min_remain,
            mask_threshold=config.mask_threshold,
        )
    else:
        raise NotImplementedError(f"Backbone {config.backbone} not implemented yet")

    return model


def iterate(
    model,
    data_loader,
    criterion,
    panoptic_meter,
    config,
    optimizer=None,
    mode="train",
    device="gpu",
):
    loss_meter = 0
    batch_count = 0
    t_start = time.time()

    for i, batch in enumerate(data_loader):
        if device == "gpu":
            batch = recursive_todevice(batch, device)

        (x, dates), targets = batch
        targets = targets.astype("float32")

        # Forward pass
        if mode != "train":
            with paddle.no_grad():
                heatmap_only = mode == "train" and config.warmup > 0
                predictions = model(
                    x,
                    batch_positions=dates,
                    zones=targets[:, :, :, 2:3] if config.supmax else None,
                    heatmap_only=heatmap_only,
                    pseudo_nms=True,
                )  # Enable pseudo_nms for testing
        else:
            heatmap_only = i < config.warmup * len(data_loader) // config.epochs
            predictions = model(
                x,
                batch_positions=dates,
                zones=targets[:, :, :, 2:3] if config.supmax else None,
                heatmap_only=heatmap_only,
                pseudo_nms=True,
            )  # Enable pseudo_nms for testing

        # Compute loss
        loss = criterion(predictions, targets, heatmap_only=heatmap_only)

        if mode == "train":
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()

        # Update metrics
        loss_meter += loss.item()
        batch_count += 1

        # Add to panoptic meter (if not warmup)
        if not heatmap_only and not config.val_metrics_only:
            # Check if we have panoptic predictions (minimal debug)
            if predictions["pano_semantic"] is not None:
                panoptic_meter.add(predictions, targets)
            else:
                # If no panoptic predictions, skip adding to meter
                pass

        if (i + 1) % config.display_step == 0:
            SQ, RQ, PQ = panoptic_meter.value()
            print(
                f"{mode} - Step [{i+1}/{len(data_loader)}] Loss: {loss_meter/batch_count:.4f} "
                f"SQ: {SQ*100:.1f} RQ: {RQ*100:.1f} PQ: {PQ*100:.1f}"
            )

    t_end = time.time()
    total_time = t_end - t_start

    # Final metrics
    SQ, RQ, PQ = panoptic_meter.value()
    avg_loss = loss_meter / batch_count if batch_count > 0 else 0

    return avg_loss, SQ.item(), RQ.item(), PQ.item(), total_time


def main(config):
    paddle.seed(config.rdm_seed)
    np.random.seed(config.rdm_seed)

    prepare_output(config)

    # Save configuration for testing
    with open(os.path.join(config.res_dir, "conf.json"), "w") as f:
        json.dump(vars(config), f, indent=4)

    if config.fold is not None:
        folds = [config.fold]
    else:
        folds = [1, 2, 3, 4, 5]

    for fold in folds:
        print(f"Starting fold {fold}")

        # Dataset definition
        dt_args = dict(
            folder=config.dataset_folder,
            norm=True,
            reference_date=config.ref_date,
            mono_date=config.mono_date,
            target="instance",  # Important: use instance target for panoptic
            sats=["S2"],
        )

        # 5-fold split
        train_folds = [f for f in [1, 2, 3, 4, 5] if f != fold]
        val_fold = [fold]
        _test_fold = [fold]  # Same as validation for now

        dt_train = PASTIS_Dataset(**dt_args, folds=train_folds, cache=config.cache)
        dt_val = PASTIS_Dataset(**dt_args, folds=val_fold, cache=config.cache)

        print(f"Train samples: {len(dt_train)}, Val samples: {len(dt_val)}")

        collate_fn = lambda x: pad_collate(x, pad_value=config.pad_value)
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

        # Model definition
        model = get_model(config)
        model.apply(weight_init)
        config.N_params = get_ntrainparams(model)

        if config.device == "gpu":
            # Paddle automatically uses GPU when available, no need for explicit .cuda()
            pass

        print(f"Model {config.backbone} - {config.N_params} trainable parameters")

        # Loss and optimizer
        criterion = PaPsLoss(
            l_center=config.l_center,
            l_size=config.l_size,
            l_shape=config.l_shape,  # Re-enable shape loss
            l_class=config.l_class,
            beta=config.beta,
        )
        optimizer = paddle.optimizer.Adam(
            parameters=model.parameters(), learning_rate=config.lr
        )

        # Training
        trainlog = {}
        best_PQ = 0

        for epoch in range(1, config.epochs + 1):
            print(f"Epoch {epoch}/{config.epochs}")

            model.train()
            train_panoptic_meter = PanopticMeter(
                num_classes=config.num_classes,
                void_label=config.ignore_index if config.ignore_index != -1 else None,
            )
            train_loss, train_SQ, train_RQ, train_PQ, train_time = iterate(
                model,
                train_loader,
                criterion,
                train_panoptic_meter,
                config,
                optimizer,
                "train",
                config.device,
            )

            if epoch % config.val_every == 0 and epoch > config.val_after:
                model.eval()
                val_panoptic_meter = PanopticMeter(
                    num_classes=config.num_classes,
                    void_label=config.ignore_index
                    if config.ignore_index != -1
                    else None,
                )
                val_loss, val_SQ, val_RQ, val_PQ, val_time = iterate(
                    model,
                    val_loader,
                    criterion,
                    val_panoptic_meter,
                    config,
                    mode="val",
                    device=config.device,
                )

                print(
                    f"Train - Loss: {train_loss:.4f}, SQ: {train_SQ*100:.1f}, RQ: {train_RQ*100:.1f}, PQ: {train_PQ*100:.1f}"
                )
                print(
                    f"Val - Loss: {val_loss:.4f}, SQ: {val_SQ*100:.1f}, RQ: {val_RQ*100:.1f}, PQ: {val_PQ*100:.1f}"
                )

                trainlog[epoch] = {
                    "train_loss": train_loss,
                    "train_SQ": train_SQ,
                    "train_RQ": train_RQ,
                    "train_PQ": train_PQ,
                    "val_loss": val_loss,
                    "val_SQ": val_SQ,
                    "val_RQ": val_RQ,
                    "val_PQ": val_PQ,
                }

                checkpoint(trainlog, config)

                if val_PQ >= best_PQ:
                    best_PQ = val_PQ
                    paddle.save(
                        {
                            "epoch": epoch,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        os.path.join(config.res_dir, f"Fold_{fold}", "model.pdparams"),
                    )
            else:
                trainlog[epoch] = {
                    "train_loss": train_loss,
                    "train_SQ": train_SQ,
                    "train_RQ": train_RQ,
                    "train_PQ": train_PQ,
                }
                checkpoint(trainlog, config)

        print(f"Fold {fold} completed. Best PQ: {best_PQ:.4f}")


if __name__ == "__main__":
    config = parser.parse_args()
    pprint.pprint(vars(config))
    main(config)
