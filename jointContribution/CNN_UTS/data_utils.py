# data_utils.py
import os
import random

import paddle
import pandas as pd
from PIL import Image


def device2str(type=None, index=None, *, device=None):
    type = device if device else type
    if isinstance(type, int):
        type = f"gpu:{type}"
    elif isinstance(type, str):
        if "cuda" in type:
            type = type.replace("cuda", "gpu")
        if "cpu" in type:
            type = "cpu"
        elif index is not None:
            type = f"{type}:{index}"
    elif isinstance(type, paddle.CPUPlace) or (type is None):
        type = "cpu"
    elif isinstance(type, paddle.CUDAPlace):
        type = f"gpu:{type.get_device_id()}"
    return type


class CustomDataset(paddle.io.Dataset):
    def __init__(self, data, device="cpu"):
        self.data = data
        self.device = device
        self.preload_to_device()

    def preload_to_device(self):
        self.data = [
            (
                image.to(self.device),
                group,
                paddle.to_tensor(data=features).astype(dtype="float32").to(self.device),
            )
            for image, group, features in self.data
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, group, features = self.data[index]
        return image, group, features


image_transforms = paddle.vision.transforms.Compose(
    transforms=[
        paddle.vision.transforms.CenterCrop(size=224),
        paddle.vision.transforms.ToTensor(),
        paddle.vision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def make_dataset(data_folder, N=1, verbose=False, device="cpu"):
    random.seed(16)
    this_data = []
    all_subfolders = [
        f
        for f in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, f)) and len(f.split("_")) >= 3
    ]

    def safe_folder_sort_key(x):
        parts = x.split("_")
        try:
            return float(parts[-3])
        except Exception:
            return float("inf")

    subfolders = sorted(all_subfolders, key=safe_folder_sort_key)
    grouped_subfolders = [[] for _ in range(5)]
    for i, subfolder in enumerate(subfolders):
        index = i // (len(subfolders) // 5)
        if index >= 5:
            index = 4
        grouped_subfolders[index].append(subfolder)
    if verbose:
        print("分组结果：", grouped_subfolders)
    chunk_keys = {}
    for i, gs in enumerate(grouped_subfolders):
        for sf in gs:
            chunk_keys[sf] = i
    sample_keys = {k: i for i, k in enumerate(subfolders)}
    for _ in range(len(subfolders) // 5 + 1):
        for k, group in enumerate(grouped_subfolders):
            if not group:
                continue
            selected_subfolder = random.choice(group)
            group.remove(selected_subfolder)
            folder_path = os.path.join(data_folder, selected_subfolder)
            if not os.path.isdir(folder_path):
                print(f"Warning: {folder_path} is not a valid directory")
                continue
            csv_data = None
            try:
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".csv"):
                        csv_path = os.path.join(folder_path, file_name)
                        try:
                            csv_data = pd.read_csv(csv_path)
                            break
                        except Exception as e:
                            print(f"Error reading CSV file {csv_path}: {str(e)}")
                            continue
            except Exception as e:
                print(f"Error accessing directory {folder_path}: {str(e)}")
                continue
            num = 0
            try:
                image_names = [
                    image_name
                    for image_name in os.listdir(folder_path)
                    if image_name.endswith(".jpg")
                ]
                image_names.sort()
            except Exception as e:
                print(f"Error reading images from {folder_path}: {str(e)}")
                continue
            for i, image_name in enumerate(image_names):
                if i % N != 0:
                    continue
                num += 1
                image_path = os.path.join(folder_path, image_name)
                image_data = Image.open(image_path).convert("RGB")
                image_data = image_transforms(image_data)
                if csv_data is not None:
                    image_features = (
                        csv_data.loc[csv_data["Image Name"] == image_name, "UTS (MPa)"]
                        .values[0]
                        .astype(float)
                    )
                else:
                    image_features = None
                this_data.append(
                    (
                        image_data,
                        (
                            chunk_keys[selected_subfolder],
                            sample_keys[selected_subfolder],
                        ),
                        image_features,
                    )
                )
            if verbose:
                print(f"文件夹 {selected_subfolder} 采样图片数: {num}")
    return CustomDataset(this_data, device=device)
