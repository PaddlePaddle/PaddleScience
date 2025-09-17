"""
Script for panoptic inference with pre-trained models (Paddle Version)
Converted to PaddlePaddle
"""
import argparse
import json
import os
import pprint

import numpy as np
import paddle
from src import model_utils as model_utils
from src.dataset import PASTIS_Dataset
from src.model_utils import get_ntrainparams
from src.panoptic.metrics import PanopticMeter
from src.panoptic.paps_loss import PaPsLoss
from src.utils import pad_collate

parser = argparse.ArgumentParser()
# Model parameters
parser.add_argument(
    "--weight_folder",
    type=str,
    default="",
    help="Path to the main folder containing the pre-trained weights",
)
parser.add_argument(
    "--dataset_folder",
    default="",
    type=str,
    help="Path to the folder where the results are saved.",
)
parser.add_argument(
    "--res_dir",
    default="./inference_paps",
    type=str,
    help="Path to directory where results are written.",
)
parser.add_argument(
    "--num_workers", default=4, type=int, help="Number of data loading workers"
)
parser.add_argument(
    "--fold",
    default=None,
    type=int,
    help="Do only one of the five fold (between 1 and 5)",
)
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
parser.add_argument("--batch_size", default=2, type=int, help="Batch size")


def recursive_todevice(x, device):
    if isinstance(x, paddle.Tensor):
        return x.cuda() if device == "gpu" else x.cpu()
    else:
        return [recursive_todevice(c, device) for c in x]


def prepare_output(config):
    os.makedirs(config.res_dir, exist_ok=True)
    for fold in range(1, 6):
        os.makedirs(os.path.join(config.res_dir, "Fold_{}".format(fold)), exist_ok=True)


def iterate(
    model,
    data_loader,
    criterion,
    panoptic_meter,
    config,
    optimizer=None,
    mode="test",
    device="gpu",
):
    """Inference iteration for panoptic segmentation"""
    loss_meter = 0

    for i, batch in enumerate(data_loader):
        if device == "gpu":
            batch = recursive_todevice(batch, device)

        (x, dates), targets = batch
        targets = targets.astype("float32")

        with paddle.no_grad():
            # Full panoptic prediction with pseudo-NMS
            predictions = model(
                x,
                batch_positions=dates,
                zones=targets[:, :, :, 2:3] if config.supmax else None,
                heatmap_only=False,
                pseudo_nms=True,
            )

            # Compute loss (optional for testing)
            loss = criterion(predictions, targets, heatmap_only=False)

        # Update metrics
        loss_meter += loss.item()

        # Add predictions to panoptic meter
        if predictions["pano_semantic"] is not None:
            panoptic_meter.add(predictions, targets)

        if (i + 1) % config.display_step == 0:
            SQ, RQ, PQ = panoptic_meter.value()
            print(
                f"{mode} - Step [{i+1}/{len(data_loader)}] Loss: {loss_meter/(i+1):.4f} "
                f"SQ: {SQ*100:.1f} RQ: {RQ*100:.1f} PQ: {PQ*100:.1f}"
            )

    # Final metrics
    SQ, RQ, PQ = panoptic_meter.value()

    metrics = {
        f"{mode}_loss": loss_meter / len(data_loader),
        f"{mode}_SQ": float(SQ),
        f"{mode}_RQ": float(RQ),
        f"{mode}_PQ": float(PQ),
    }

    return metrics, panoptic_meter.get_table()


def save_results(fold, metrics, tables, config):
    """Save test results"""
    fold_dir = os.path.join(config.res_dir, f"Fold_{fold}")

    # Save metrics as JSON
    with open(os.path.join(fold_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Save detailed tables as numpy binary file (same as PyTorch version)
    np.save(os.path.join(fold_dir, "test_tables"), tables)


def overall_performance(config):
    """Compute overall performance across all folds"""
    all_metrics = []
    all_tables = []

    for fold in range(1, 6):
        fold_dir = os.path.join(config.res_dir, f"Fold_{fold}")

        # Load metrics
        metrics_path = os.path.join(fold_dir, "test_metrics.json")
        if not os.path.exists(metrics_path):
            continue

        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        all_metrics.append(metrics)

        # Load tables (numpy format)
        tables_path = os.path.join(fold_dir, "test_tables.npy")
        if os.path.exists(tables_path):
            tables = np.load(tables_path)
            all_tables.append(tables)

    if not all_metrics:
        print("No test results found!")
        return

    # Compute averages
    avg_metrics = {}
    for key in all_metrics[0].keys():
        if isinstance(all_metrics[0][key], (int, float)):
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
            avg_metrics[key + "_std"] = np.std([m[key] for m in all_metrics])

    # Save overall results
    with open(os.path.join(config.res_dir, "overall_metrics.json"), "w") as f:
        json.dump(avg_metrics, f, indent=4)

    print("=== OVERALL PANOPTIC RESULTS ===")
    print(
        f"Average Loss: {avg_metrics['test_loss']:.4f} ± {avg_metrics['test_loss_std']:.4f}"
    )
    print(
        f"Average SQ: {avg_metrics['test_SQ']*100:.1f} ± {avg_metrics['test_SQ_std']*100:.1f}"
    )
    print(
        f"Average RQ: {avg_metrics['test_RQ']*100:.1f} ± {avg_metrics['test_RQ_std']*100:.1f}"
    )
    print(
        f"Average PQ: {avg_metrics['test_PQ']*100:.1f} ± {avg_metrics['test_PQ_std']*100:.1f}"
    )


def main(config):
    fold_sequence = [
        [[1, 2, 3], [4], [5]],
        [[2, 3, 4], [5], [1]],
        [[3, 4, 5], [1], [2]],
        [[4, 5, 1], [2], [3]],
        [[5, 1, 2], [3], [4]],
    ]

    paddle.seed(config.rdm_seed)
    np.random.seed(config.rdm_seed)
    prepare_output(config)

    # Set device
    if config.device == "gpu" and paddle.is_compiled_with_cuda():
        paddle.device.set_device("gpu")
    else:
        paddle.device.set_device("cpu")
        config.device = "cpu"

    # Create model
    model = model_utils.get_model(config, mode="panoptic")
    config.N_params = get_ntrainparams(model)
    print("TOTAL TRAINABLE PARAMETERS :", config.N_params)

    fold_sequence = (
        fold_sequence if config.fold is None else [fold_sequence[config.fold - 1]]
    )

    for fold, (train_folds, val_fold, test_fold) in enumerate(fold_sequence):
        if config.fold is not None:
            fold = config.fold - 1

        print(f"\n=== Testing Fold {fold + 1} ===")

        # Dataset definition
        dt_test = PASTIS_Dataset(
            folder=config.dataset_folder,
            norm=True,
            reference_date=config.ref_date,
            mono_date=config.mono_date,
            target="instance",  # Important: use instance target for panoptic
            sats=["S2"],
            folds=test_fold,
        )
        collate_fn = lambda x: pad_collate(x, pad_value=config.pad_value)
        test_loader = paddle.io.DataLoader(
            dt_test,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
        )

        print(f"Test samples: {len(dt_test)}")

        # Load weights
        weight_path = config.weight_folder

        if not os.path.exists(weight_path):
            print(f"Warning: Weight file not found at {weight_path}")
            continue

        checkpoint = paddle.load(weight_path)
        if "state_dict" in checkpoint:
            model.set_state_dict(checkpoint["state_dict"])
        else:
            model.set_state_dict(checkpoint)
        print(f"Loaded weights from {weight_path}")

        # Loss and metrics
        criterion = PaPsLoss(
            l_center=config.l_center,
            l_size=config.l_size,
            l_shape=config.l_shape,
            l_class=config.l_class,
            beta=config.beta,
        )

        panoptic_meter = PanopticMeter(
            num_classes=config.num_classes,
            void_label=config.ignore_index if config.ignore_index != -1 else None,
        )

        # Inference
        print("Testing . . .")
        model.eval()
        test_metrics, tables = iterate(
            model,
            data_loader=test_loader,
            criterion=criterion,
            panoptic_meter=panoptic_meter,
            config=config,
            optimizer=None,
            mode="test",
            device=config.device,
        )

        print(
            "Loss {:.4f}, SQ {:.1f}, RQ {:.1f}, PQ {:.1f}".format(
                test_metrics["test_loss"],
                test_metrics["test_SQ"] * 100,
                test_metrics["test_RQ"] * 100,
                test_metrics["test_PQ"] * 100,
            )
        )
        # print("test_metrics_SQ : ",test_metrics['test_SQ'])
        # print("test_metrics_RQ : ",test_metrics['test_RQ'])
        # print("test_metrics_PQ : ",test_metrics['test_PQ'])
        save_results(fold + 1, test_metrics, tables, config)

    if config.fold is None:
        overall_performance(config)


if __name__ == "__main__":
    test_config = parser.parse_args()

    # Try to load config from conf.json if it exists, otherwise use defaults
    conf_path = os.path.join(test_config.weight_folder, "conf.json")
    if os.path.exists(conf_path):
        with open(conf_path) as file:
            model_config = json.loads(file.read())
        config = {**model_config, **vars(test_config)}
    else:
        print("Warning: conf.json not found, using test script parameters only")
        # Set default model parameters for panoptic
        default_config = {
            "backbone": "utae",
            "encoder_widths": "[64,64,64,128]",
            "decoder_widths": "[32,32,64,128]",
            "out_conv": "[32, 20]",
            "str_conv_k": 4,
            "str_conv_s": 2,
            "str_conv_p": 1,
            "agg_mode": "att_group",
            "encoder_norm": "group",
            "n_head": 16,
            "d_model": 256,
            "d_k": 4,
            "num_classes": 20,
            "ignore_index": -1,
            "pad_value": 0,
            "padding_mode": "reflect",
            "ref_date": "2018-09-01",
            "mono_date": None,
            "rdm_seed": 1,
            "supmax": True,
            # PaPs specific parameters
            "shape_size": 16,
            "mask_conv": True,
            "min_confidence": 0.2,
            "min_remain": 0.5,
            "mask_threshold": 0.4,
            "l_center": 1,
            "l_size": 1,
            "l_shape": 1,
            "l_class": 1,
            "beta": 4,
        }
        config = {**default_config, **vars(test_config)}

    config = argparse.Namespace(**config)
    config.fold = test_config.fold

    pprint.pprint(vars(config))
    main(config)
