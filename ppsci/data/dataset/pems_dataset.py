import os
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from paddle.io import Dataset
from paddle.vision.transforms import Compose


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def add_window_horizon(data, in_step=12, out_step=12):
    length = len(data)
    end_index = length - out_step - in_step
    X = []
    Y = []
    for i in range(end_index + 1):
        X.append(data[i : i + in_step])
        Y.append(data[i + in_step : i + in_step + out_step])
    return X, Y


def get_edge_index(file_path, bi=True, reduce="mean"):
    TYPE_DICT = {0: np.int64, 1: np.int64, 2: np.float32}
    df = pd.read_csv(
        os.path.join(file_path, "dist.csv"),
        skiprows=1,
        header=None,
        sep=",",
        dtype=TYPE_DICT,
    )

    edge_index = df.loc[:, [0, 1]].values.T
    edge_attr = df.loc[:, 2].values

    if bi:
        re_edge_index = np.concatenate((edge_index[1:, :], edge_index[:1, :]), axis=0)
        edge_index = np.concatenate((edge_index, re_edge_index), axis=-1)
        edge_attr = np.concatenate((edge_attr, edge_attr), axis=0)

    num = np.max(edge_index) + 1
    adj = np.zeros((num, num), dtype=np.float32)

    if reduce == "sum":
        adj[edge_index[0], edge_index[1]] = 1.0
    elif reduce == "mean":
        adj[edge_index[0], edge_index[1]] = 1.0
        adj = adj / adj.sum(axis=-1)
    else:
        raise ValueError

    return edge_index, edge_attr, adj


class PEMSDataset(Dataset):
    """Dataset class for PEMSD4 and PEMSD8 dataset.

    Args:
        file_path (str): Dataset root path.
        split (str): Dataset split label.
        input_keys (Tuple[str, ...]): A tuple of input keys.
        label_keys (Tuple[str, ...]): A tuple of label keys.
        weight_dict (Optional[Dict[str, float]]): Define the weight of each constraint variable. Defaults to None.
        transforms (Optional[Compose]): Compose object contains sample wise transform(s). Defaults to None.
        norm_input (bool): Whether to normalize the input. Defaults to True.
        norm_label (bool): Whether to normalize the output. Defaults to False.
        input_len (int): The input timesteps. Defaults to 12.
        label_len (int): The output timesteps. Defaults to 12.

    Examples:
        >>> import ppsci
        >>> dataset = ppsci.data.dataset.PEMSDataset(
        ...     "./Data/PEMSD4",
        ...     "train",
        ...     ("input",),
        ...     ("label",),
        ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        file_path: str,
        split: str,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        weight_dict: Optional[Dict[str, float]] = None,
        transforms: Optional[Compose] = None,
        norm_input: bool = True,
        norm_label: bool = False,
        input_len: int = 12,
        label_len: int = 12,
    ):
        super().__init__()

        self.input_keys = input_keys
        self.label_keys = label_keys
        self.weight_dict = weight_dict

        self.transforms = transforms
        self.norm_input = norm_input
        self.norm_label = norm_label

        data = np.load(os.path.join(file_path, f"{split}.npy")).astype(np.float32)

        self.mean = np.load(os.path.join(file_path, "mean.npy")).astype(np.float32)
        self.std = np.load(os.path.join(file_path, "std.npy")).astype(np.float32)
        self.scaler = StandardScaler(self.mean, self.std)

        X, Y = add_window_horizon(data, input_len, label_len)
        if norm_input:
            X = self.scaler.transform(X)
        if norm_label:
            Y = self.scaler.transform(Y)

        self._len = X.shape[0]

        self.input = {input_keys[0]: X}
        self.label = {label_keys[0]: Y}

        if weight_dict is not None:
            self.weight_dict = {key: np.array(1.0) for key in self.label_keys}
            self.weight_dict.update(weight_dict)
        else:
            self.weight = {}

    def __getitem__(self, idx):
        input_item = {key: value[idx] for key, value in self.input.items()}
        label_item = {key: value[idx] for key, value in self.label.items()}
        weight_item = {key: value[idx] for key, value in self.weight.items()}

        if self.transforms is not None:
            input_item, label_item, weight_item = self.transforms(
                input_item, label_item, weight_item
            )

        return (input_item, label_item, weight_item)

    def __len__(self):
        return self._len
