import argparse
import glob
import multiprocessing as mp
import multiprocessing.pool as mpp
import os
import random
import time

import albumentations as albu
import cv2
import numpy as np
import paddle
from paddle_utils import PaddleFlag
from paddle_utils import add_tensor_methods
from PIL import Image

SEED = 42
add_tensor_methods()


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    paddle.seed(seed=seed)
    paddle.seed(seed=seed)
    PaddleFlag.cudnn_deterministic = True
    PaddleFlag.cudnn_benchmark = True


ImSurf = np.array([255, 255, 255])
Building = np.array([255, 0, 0])
LowVeg = np.array([255, 255, 0])
Tree = np.array([0, 255, 0])
Car = np.array([0, 255, 255])
Clutter = np.array([0, 0, 255])
Boundary = np.array([0, 0, 0])
num_classes = 6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", default="data/vaihingen/train_images")
    parser.add_argument("--mask-dir", default="data/vaihingen/train_masks")
    parser.add_argument("--output-img-dir", default="data/vaihingen/train/images_1024")
    parser.add_argument("--output-mask-dir", default="data/vaihingen/train/masks_1024")
    parser.add_argument("--eroded", action="store_true")
    parser.add_argument("--gt", action="store_true")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--val-scale", type=float, default=1.0)
    parser.add_argument("--split-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    return parser.parse_args()


def get_img_mask_padded(image, mask, patch_size, mode):
    img, mask = np.array(image), np.array(mask)
    oh, ow = tuple(img.shape)[0], tuple(img.shape)[1]
    rh, rw = oh % patch_size, ow % patch_size
    width_pad = 0 if rw == 0 else patch_size - rw
    height_pad = 0 if rh == 0 else patch_size - rh
    h, w = oh + height_pad, ow + width_pad
    pad_img = albu.PadIfNeeded(
        min_height=h,
        min_width=w,
        position="bottom_right",
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
    )(image=img)
    pad_mask = albu.PadIfNeeded(
        min_height=h,
        min_width=w,
        position="bottom_right",
        border_mode=cv2.BORDER_CONSTANT,
        value=6,
    )(image=mask)
    img_pad, mask_pad = pad_img["image"], pad_mask["image"]
    img_pad = cv2.cvtColor(np.array(img_pad), cv2.COLOR_RGB2BGR)
    mask_pad = cv2.cvtColor(np.array(mask_pad), cv2.COLOR_RGB2BGR)
    return img_pad, mask_pad


def pv2rgb(mask):
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


def car_color_replace(mask):
    mask = cv2.cvtColor(np.array(mask.copy()), cv2.COLOR_RGB2BGR)
    mask[np.all(mask == [0, 255, 255], axis=-1)] = [0, 204, 255]
    return mask


def rgb_to_2D_label(_label):
    _label = _label.transpose(2, 0, 1)
    label_seg = np.zeros(tuple(_label.shape)[1:], dtype=np.uint8)
    label_seg[np.all(_label.transpose([1, 2, 0]) == ImSurf, axis=-1)] = 0
    label_seg[np.all(_label.transpose([1, 2, 0]) == Building, axis=-1)] = 1
    label_seg[np.all(_label.transpose([1, 2, 0]) == LowVeg, axis=-1)] = 2
    label_seg[np.all(_label.transpose([1, 2, 0]) == Tree, axis=-1)] = 3
    label_seg[np.all(_label.transpose([1, 2, 0]) == Car, axis=-1)] = 4
    label_seg[np.all(_label.transpose([1, 2, 0]) == Clutter, axis=-1)] = 5
    label_seg[np.all(_label.transpose([1, 2, 0]) == Boundary, axis=-1)] = 6
    return label_seg


def image_augment(image, mask, patch_size, mode="train", val_scale=1.0):
    image_list = []
    mask_list = []
    image_width, image_height = image.size[1], image.size[0]
    mask_width, mask_height = mask.size[1], mask.size[0]
    assert image_height == mask_height and image_width == mask_width
    if mode == "train":
        h_vlip = paddle.vision.transforms.RandomHorizontalFlip(prob=1.0)
        v_vlip = paddle.vision.transforms.RandomVerticalFlip(prob=1.0)
        image_h_vlip, mask_h_vlip = h_vlip(image.copy()), h_vlip(mask.copy())
        image_v_vlip, mask_v_vlip = v_vlip(image.copy()), v_vlip(mask.copy())
        image_list_train = [image, image_h_vlip, image_v_vlip]
        mask_list_train = [mask, mask_h_vlip, mask_v_vlip]
        for i in range(len(image_list_train)):
            image_tmp, mask_tmp = get_img_mask_padded(
                image_list_train[i], mask_list_train[i], patch_size, mode
            )
            mask_tmp = rgb_to_2D_label(mask_tmp.copy())
            image_list.append(image_tmp)
            mask_list.append(mask_tmp)
    else:
        rescale = paddle.vision.transforms.Resize(
            size=(int(image_width * val_scale), int(image_height * val_scale))
        )
        image, mask = rescale(image.copy()), rescale(mask.copy())
        image, mask = get_img_mask_padded(image.copy(), mask.copy(), patch_size, mode)
        mask = rgb_to_2D_label(mask.copy())
        image_list.append(image)
        mask_list.append(mask)
    return image_list, mask_list


def randomsizedcrop(image, mask):
    h, w = tuple(image.shape)[0], tuple(image.shape)[1]
    crop = albu.RandomSizedCrop(
        min_max_height=(int(3 * h // 8), int(h // 2)), width=h, height=w
    )(image=image.copy(), mask=mask.copy())
    img_crop, mask_crop = crop["image"], crop["mask"]
    return img_crop, mask_crop


def car_aug(image, mask):
    assert tuple(image.shape)[:2] == tuple(mask.shape)
    v_flip = albu.VerticalFlip(p=1.0)(image=image.copy(), mask=mask.copy())
    h_flip = albu.HorizontalFlip(p=1.0)(image=image.copy(), mask=mask.copy())
    rotate_90 = albu.RandomRotate90(p=1.0)(image=image.copy(), mask=mask.copy())
    image_vflip, mask_vflip = v_flip["image"], v_flip["mask"]
    image_hflip, mask_hflip = h_flip["image"], h_flip["mask"]
    image_rotate, mask_rotate = rotate_90["image"], rotate_90["mask"]
    image_list = [image, image_vflip, image_hflip, image_rotate]
    mask_list = [mask, mask_vflip, mask_hflip, mask_rotate]
    return image_list, mask_list


def vaihingen_format(inp):
    (
        img_path,
        mask_path,
        imgs_output_dir,
        masks_output_dir,
        eroded,
        gt,
        mode,
        val_scale,
        split_size,
        stride,
    ) = inp
    img_filename = os.path.splitext(os.path.basename(img_path))[0]
    mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
    if eroded:
        mask_path = mask_path[:-4] + "_noBoundary.tif"
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("RGB")
    if gt:
        mask_ = car_color_replace(mask)
        out_origin_mask_path = os.path.join(
            masks_output_dir + "/origin/", "{}.tif".format(mask_filename)
        )
        cv2.imwrite(out_origin_mask_path, mask_)
    image_list, mask_list = image_augment(
        image=img.copy(),
        mask=mask.copy(),
        patch_size=split_size,
        mode=mode,
        val_scale=val_scale,
    )
    assert img_filename == mask_filename and len(image_list) == len(mask_list)
    for m in range(len(image_list)):
        k = 0
        img = image_list[m]
        mask = mask_list[m]
        assert (
            tuple(img.shape)[0] == tuple(mask.shape)[0]
            and tuple(img.shape)[1] == tuple(mask.shape)[1]
        )
        if gt:
            mask = pv2rgb(mask)
        for y in range(0, tuple(img.shape)[0], stride):
            for x in range(0, tuple(img.shape)[1], stride):
                img_tile = img[y : y + split_size, x : x + split_size]
                mask_tile = mask[y : y + split_size, x : x + split_size]
                if (
                    tuple(img_tile.shape)[0] == split_size
                    and tuple(img_tile.shape)[1] == split_size
                    and tuple(mask_tile.shape)[0] == split_size
                    and tuple(mask_tile.shape)[1] == split_size
                ):
                    image_crop, mask_crop = randomsizedcrop(img_tile, mask_tile)
                    bins = np.array(range(num_classes + 1))
                    class_pixel_counts, _ = np.histogram(mask_crop, bins=bins)
                    cf = class_pixel_counts / (
                        tuple(mask_crop.shape)[0] * tuple(mask_crop.shape)[1]
                    )
                    if cf[4] > 0.1 and mode == "train":
                        car_imgs, car_masks = car_aug(image_crop, mask_crop)
                        for i in range(len(car_imgs)):
                            out_img_path = os.path.join(
                                imgs_output_dir,
                                "{}_{}_{}_{}.tif".format(img_filename, m, k, i),
                            )
                            cv2.imwrite(out_img_path, car_imgs[i])
                            out_mask_path = os.path.join(
                                masks_output_dir,
                                "{}_{}_{}_{}.png".format(mask_filename, m, k, i),
                            )
                            cv2.imwrite(out_mask_path, car_masks[i])
                    else:
                        out_img_path = os.path.join(
                            imgs_output_dir, "{}_{}_{}.tif".format(img_filename, m, k)
                        )
                        cv2.imwrite(out_img_path, img_tile)
                        out_mask_path = os.path.join(
                            masks_output_dir, "{}_{}_{}.png".format(mask_filename, m, k)
                        )
                        cv2.imwrite(out_mask_path, mask_tile)
                k += 1


if __name__ == "__main__":
    seed_everything(SEED)
    args = parse_args()
    imgs_dir = args.img_dir
    masks_dir = args.mask_dir
    imgs_output_dir = args.output_img_dir
    masks_output_dir = args.output_mask_dir
    gt = args.gt
    eroded = args.eroded
    mode = args.mode
    val_scale = args.val_scale
    split_size = args.split_size
    stride = args.stride
    img_paths = glob.glob(os.path.join(imgs_dir, "*.tif"))
    mask_paths_raw = glob.glob(os.path.join(masks_dir, "*.tif"))
    if eroded:
        mask_paths = [(p[:-15] + ".tif") for p in mask_paths_raw]
    else:
        mask_paths = mask_paths_raw
    paddle.sort(x=img_paths), paddle.argsort(x=img_paths)
    paddle.sort(x=mask_paths), paddle.argsort(x=mask_paths)
    if not os.path.exists(imgs_output_dir):
        os.makedirs(imgs_output_dir)
    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)
        if gt:
            os.makedirs(masks_output_dir + "/origin")
    inp = [
        (
            img_path,
            mask_path,
            imgs_output_dir,
            masks_output_dir,
            eroded,
            gt,
            mode,
            val_scale,
            split_size,
            stride,
        )
        for img_path, mask_path in zip(img_paths, mask_paths)
    ]
    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(vaihingen_format, inp)
    t1 = time.time()
    split_time = t1 - t0
    print("images spliting spends: {} s".format(split_time))
