import argparse
import glob
import json
import os
import pprint
import re

import numpy as np
import paddle
import paddle.io as data
import paddle.nn as nn
from src import model_utils
from src import utils
from src.dataset import PASTIS_Dataset


def prepare_output(res_dir: str):
    os.makedirs(res_dir, exist_ok=True)
    for k in range(1, 6):
        os.makedirs(os.path.join(res_dir, f"Fold_{k}"), exist_ok=True)


def _auto_pick_ckpt(fold_dir: str) -> str:

    pref = os.path.join(fold_dir, "model.pdparams")
    if os.path.isfile(pref):
        return pref
    cands = glob.glob(os.path.join(fold_dir, "*.pdparams"))
    if not cands:
        raise ValueError(f"No .pdparams in {fold_dir}")

    def score(p):
        m = re.search(r"miou_([0-9.]+)\.pdparams$", p)
        return float(m.group(1)) if m else -1.0

    return max(cands, key=lambda p: (score(p), os.path.getmtime(p)))


def iterate_eval(model, data_loader, criterion, num_classes, ignore_index):
    # IoU 兼容两种路径
    try:
        from src.learning.miou import IoU
    except Exception:
        from src.miou import IoU

    iou_meter = IoU(num_classes=num_classes, ignore_index=ignore_index, cm_device="cpu")
    loss_sum, nb = 0.0, 0
    model.eval()
    with paddle.no_grad():
        for (x, dates), y in data_loader:
            out = model(x, batch_positions=dates)  # [B,C,H,W]
            B, C, H, W = out.shape
            logits = out.transpose([0, 2, 3, 1]).reshape([-1, C])
            target = y.reshape([-1]).astype("int64")
            loss = criterion(logits, target)
            loss_sum += float(loss.numpy())
            nb += 1
            pred = nn.functional.softmax(out, axis=1).argmax(axis=1)
            iou_meter.add(pred, y)
    miou, acc = iou_meter.get_miou_acc()
    return {
        "test_loss": loss_sum / max(1, nb),
        "test_accuracy": float(acc),
        "test_IoU": float(miou),
    }, iou_meter.confusion_matrix


def _to_eval_string(v, fallback: str):

    if v is None:
        return fallback
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(str(x) for x in v) + "]"
    return str(v)


def main(cfg):
    # 设备兜底
    if cfg.device == "gpu" and not paddle.is_compiled_with_cuda():
        print("⚠️ 当前环境未编译 CUDA，自动切到 CPU。")
        cfg.device = "cpu"
    paddle.set_device(cfg.device)
    np.random.seed(cfg.rdm_seed)
    paddle.seed(cfg.rdm_seed)
    prepare_output(cfg.res_dir)
    cfg.encoder_widths = _to_eval_string(
        getattr(cfg, "encoder_widths", None), "[64,64,64,128]"
    )
    cfg.decoder_widths = _to_eval_string(
        getattr(cfg, "decoder_widths", None), "[32,32,64,128]"
    )
    cfg.out_conv = _to_eval_string(getattr(cfg, "out_conv", None), "[32, 20]")

    # 构建模型
    model = model_utils.get_model(cfg, mode="semantic")
    print(model)
    print("TOTAL TRAINABLE PARAMETERS :", model_utils.get_ntrainparams(model))

    # 折序列（与训练一致）
    fold_sequence = [
        [[1, 2, 3], [4], [5]],
        [[2, 3, 4], [5], [1]],
        [[3, 4, 5], [1], [2]],
        [[4, 5, 1], [2], [3]],
        [[5, 1, 2], [3], [4]],
    ]

    # —— 如果指定 --weight_file，仅评一个 fold —— #
    if cfg.weight_file:
        if cfg.fold is not None:
            run_fold = cfg.fold
        else:
            m = re.search(r"[\\/](?:Fold_|fold_)(\d)[\\/]", cfg.weight_file)
            run_fold = int(m.group(1)) if m else 1
        seq = [fold_sequence[run_fold - 1]]
        print(f"Single-fold mode (from weight_file): fold={run_fold}")
    else:
        seq = fold_sequence if cfg.fold is None else [fold_sequence[cfg.fold - 1]]

    for idx, (_, _, test_fold) in enumerate(seq):
        fold_id = (
            (cfg.fold if cfg.fold is not None else (idx + 1))
            if not cfg.weight_file
            else run_fold
        )

        # 数据
        ds = PASTIS_Dataset(
            folder=cfg.dataset_folder,
            norm=True,
            reference_date=cfg.ref_date,
            mono_date=cfg.mono_date,
            target="semantic",
            sats=["S2"],
            folds=test_fold,
        )
        collate = lambda x: utils.pad_collate(x, pad_value=cfg.pad_value)
        loader = data.DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate,
            num_workers=cfg.num_workers,
        )
        print(
            f"#test samples: {len(ds)}, batch_size: {cfg.batch_size}, #batches: {len(loader)}"
        )

        # 权重
        if cfg.weight_file:
            wpath = cfg.weight_file
        else:
            if not cfg.weight_folder:
                raise ValueError("Provide --weight_file or --weight_folder")
            fold_dir = os.path.join(cfg.weight_folder, f"Fold_{fold_id}")
            if not os.path.isdir(fold_dir):
                raise ValueError(f"Fold dir not found: {fold_dir}")
            wpath = _auto_pick_ckpt(fold_dir)
        print(f"Loading weights: {wpath}")
        sd = paddle.load(wpath)
        state = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
        model.set_state_dict(state)

        # 损失
        w = paddle.ones([cfg.num_classes], dtype="float32")
        if 0 <= cfg.ignore_index < cfg.num_classes:
            w[cfg.ignore_index] = 0
        criterion = nn.CrossEntropyLoss(weight=w)

        # 推理
        print("Testing ...")
        metrics, cm = iterate_eval(
            model, loader, criterion, cfg.num_classes, cfg.ignore_index
        )
        print(
            f"[Fold {fold_id}] Loss {metrics['test_loss']:.4f}, "
            f"Acc {metrics['test_accuracy']:.2f}, IoU {metrics['test_IoU']:.4f}"
        )

        # 保存
        outd = os.path.join(cfg.res_dir, f"Fold_{fold_id}")
        os.makedirs(outd, exist_ok=True)
        with open(os.path.join(outd, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        np.save(os.path.join(outd, "confusion_matrix.npy"), cm)
        print(f"Saved metrics and confusion matrix to {outd}")

        # --weight_file 触发的单折模式：跑完即结束
        if cfg.weight_file:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 运行 / 数据
    parser.add_argument("--dataset_folder", type=str, default="", help="PASTIS 根目录")
    parser.add_argument("--res_dir", type=str, default="./inference_utae")
    parser.add_argument("--fold", type=int, default=None, help="1..5；指定时只评该折")
    parser.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--num_workers", type=int, default=0)

    # 权重
    parser.add_argument(
        "--weight_folder", type=str, default="", help="results 根目录，自动为每折挑 ckpt"
    )
    parser.add_argument(
        "--weight_file", type=str, default="", help="单个 .pdparams 路径（只评一个 fold）"
    )

    # 模型结构（与训练一致；可按需覆盖）
    parser.add_argument("--model", type=str, default="utae")
    parser.add_argument("--encoder_widths", type=str, default="[64,64,64,128]")
    parser.add_argument("--decoder_widths", type=str, default="[32,32,64,128]")
    parser.add_argument("--out_conv", type=str, default="[32, 20]")
    parser.add_argument("--str_conv_k", type=int, default=4)
    parser.add_argument("--str_conv_s", type=int, default=2)
    parser.add_argument("--str_conv_p", type=int, default=1)
    parser.add_argument("--agg_mode", type=str, default="att_group")
    parser.add_argument("--encoder_norm", type=str, default="group")
    parser.add_argument("--n_head", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_k", type=int, default=4)
    parser.add_argument("--padding_mode", type=str, default="reflect")

    # 标签 / 批量
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--ignore_index", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--ref_date", type=str, default="2018-09-01")
    parser.add_argument("--pad_value", type=float, default=0.0)
    parser.add_argument("--rdm_seed", type=int, default=1)
    parser.add_argument("--mono_date", type=str, default=None)

    cfg = parser.parse_args()
    pprint.pprint(vars(cfg))
    main(cfg)
