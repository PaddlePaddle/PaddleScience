import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import cv2
import numpy as np
import paddle
from paddle_utils import add_tensor_methods
from tqdm import tqdm
from train_supervision import Evaluator
from train_supervision import Supervision_Train
from train_supervision import py2cfg
from train_supervision import random

add_tensor_methods()


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    paddle.seed(seed=seed)


def label2rgb(mask):
    h, w = tuple(mask.shape)[0], tuple(mask.shape)[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[(np.newaxis), :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 204, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
    return mask_rgb


def img_writer(inp):
    mask, mask_id, rgb = inp
    if rgb:
        mask_name_tif = mask_id + ".png"
        mask_tif = label2rgb(mask)
        cv2.imwrite(mask_name_tif, mask_tif)
    else:
        mask_png = mask.astype(np.uint8)
        mask_name_png = mask_id + ".png"
        cv2.imwrite(mask_name_png, mask_png)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to config")
    arg("-o", "--output_path", type=Path, required=True, help="Output path for masks")
    arg("--rgb", help="Output RGB images", action="store_true")
    return parser.parse_args()


def main():
    seed_everything(42)
    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + ".pdparams"),
        config=config,
    )
    model.to(device="gpu")
    model.eval()
    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()
    test_dataset = config.test_dataset
    test_loader = paddle.io.DataLoader(
        dataset=test_dataset,
        batch_size=2,
        num_workers=4,
        drop_last=False,
        shuffle=False,
    )

    results = []
    with paddle.no_grad():
        for batch in tqdm(test_loader):
            images = batch["img"]
            images = images.astype("float32")
            raw_predictions = model(images)

            raw_predictions = paddle.nn.functional.softmax(raw_predictions, axis=1)
            predictions = raw_predictions.argmax(axis=1)

            image_ids = batch["img_id"]
            masks_true = batch["gt_semantic_seg"]

            for i in range(len(image_ids)):
                mask = predictions[i].numpy()
                evaluator.add_batch(pre_image=mask, gt_image=masks_true[i].numpy())
                mask_name = image_ids[i]
                results.append((mask, str(args.output_path / mask_name), args.rgb))

    iou_per_class = evaluator.Intersection_over_Union()
    f1_per_class = evaluator.F1()
    OA = evaluator.OA()

    for class_name, class_iou, class_f1 in zip(
        config.classes, iou_per_class, f1_per_class
    ):
        print(f"F1_{class_name}: {class_f1:.4f}, IOU_{class_name}: {class_iou:.4f}")

    print(
        f"F1: {np.nanmean(f1_per_class[:-1]):.4f}, "
        f"mIOU: {np.nanmean(iou_per_class[:-1]):.4f}, "
        f"OA: {OA:.4f}"
    )

    t0 = time.time()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(img_writer, results)
    t1 = time.time()
    print(f"Images writing time: {t1 - t0:.2f} seconds")


if __name__ == "__main__":
    main()
