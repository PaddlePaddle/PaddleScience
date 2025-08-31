import os
import os.path as osp
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
from paddle import io


class Normalizer:
    def __init__(self, x: np.ndarray, y1: np.ndarray, y2: np.ndarray):
        self.x_flag = False
        self.y1_flag = False
        self.y2_flag = False

        old_x_shape = x.shape
        old_y1_shape = y1.shape
        old_y2_shape = y2.shape

        x = x.reshape(-1, x.shape[-1])
        y1 = y1.reshape(-1, y1.shape[-1])
        y2 = y2.reshape(-1, y2.shape[-1])

        self.x_mean = np.mean(x, axis=0, keepdims=True).astype("float32")
        self.x_std = (np.std(x, axis=0, keepdims=True) + 1e-8).astype("float32")
        self.y1_mean = np.mean(y1, axis=0, keepdims=True).astype("float32")
        self.y1_std = (np.std(y1, axis=0, keepdims=True) + 1e-8).astype("float32")
        self.y2_mean = np.mean(y2, axis=0, keepdims=True).astype("float32")
        self.y2_std = (np.std(y2, axis=0, keepdims=True) + 1e-8).astype("float32")

        x = x.reshape(old_x_shape)
        y1 = y1.reshape(old_y1_shape)
        y2 = y2.reshape(old_y2_shape)

    def is_apply_x(self) -> bool:
        return self.x_flag

    def is_apply_y1(self) -> bool:
        return self.y1_flag

    def is_apply_y2(self) -> bool:
        return self.y2_flag

    def apply_x(self, x: np.ndarray, inverse: bool = False) -> np.ndarray:
        old_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        if not inverse:
            x = (x - self.x_mean) / self.x_std
            self.x_flag = True
        else:
            x = x * self.x_std + self.x_mean
        return x.reshape(old_shape).astype("float32")

    def apply_y1(self, y1: np.ndarray, inverse: bool = False) -> np.ndarray:
        old_shape = y1.shape
        y1 = y1.reshape(-1, y1.shape[-1])
        if not inverse:
            y1 = (y1 - self.y1_mean) / self.y1_std
            self.y1_flag = True
        else:
            y1 = y1 * self.y1_std + self.y1_mean
        return y1.reshape(old_shape).astype("float32")

    def apply_y2(self, y2: np.ndarray, inverse: bool = False) -> np.ndarray:
        old_shape = y2.shape
        y2 = y2.reshape(-1, y2.shape[-1])
        if not inverse:
            y2 = (y2 - self.y2_mean) / self.y2_std
            self.y2_flag = True
        else:
            y2 = y2 * self.y2_std + self.y2_mean
        return y2.reshape(old_shape).astype("float32")


class LatentNODataset(io.Dataset):
    """LatentNO Dataset for PaddleScience automatic training.

    Args:
        data_name (str): Data name identifier
        data_mode (str): "train" or "val"
        data_normalize (bool): Whether to normalize data
        data_concat (bool): Whether to concatenate x and y1
        input_keys (Tuple[str, ...]): Input keys, such as ("x", "y1")
        label_keys (Tuple[str, ...]): Label keys, such as ("y2",)
        weight_dict (Optional[Dict[str, float]]): Weight dictionary for loss terms
        transform_fn (Optional[Callable]): Optional transform function
    """

    # Whether support batch indexing for speeding up fetching process.
    batch_index: bool = True

    def __init__(
        self,
        data_name: str,
        data_mode: str,
        data_normalize: bool,
        data_concat: bool,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        weight_dict: Optional[Dict[str, float]] = None,
        transform_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.data_name = data_name
        self.data_mode = data_mode
        self.data_normalize = data_normalize
        self.data_concat = data_concat
        self.input_keys = list(input_keys)
        self.label_keys = list(label_keys)
        self.weight_dict = weight_dict or {}
        self.transform_fn = transform_fn

        # Load data
        data_file = osp.join("datas", f"{data_name}_{data_mode}.npy")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        dataset = np.load(data_file, allow_pickle=True).tolist()

        x = dataset["x"].astype("float32")
        y1 = dataset["y1"].astype("float32")
        y2 = dataset["y2"].astype("float32")

        x = x.reshape((x.shape[0], -1, x.shape[-1]))
        y1 = y1.reshape((y1.shape[0], -1, y1.shape[-1]))
        y2 = y2.reshape((y2.shape[0], -1, y2.shape[-1]))
        # Concatenate x and y1 if required
        if data_concat:
            y1 = np.concatenate((x, y1), axis=-1)

        # Initialize normalizer
        self.normalizer = Normalizer(x, y1, y2)

        # Apply normalization if required
        if data_normalize:
            x = self.normalizer.apply_x(x)
            y1 = self.normalizer.apply_y1(y1)
            y2 = self.normalizer.apply_y2(y2)

        # Store data in input and label dictionaries following PaddleScience convention
        self.input = {
            self.input_keys[0]: x,  # "x"
            self.input_keys[1]: y1,  # "y1"
        }

        self.label = {
            self.label_keys[0]: y2,  # "y2"
        }

        self._length = x.shape[0]

    def __getitem__(self, index: int):
        # Get input data
        input_item = {key: value[index] for key, value in self.input.items()}

        # Get label data
        label_item = {key: value[index] for key, value in self.label.items()}

        weight_item = {}
        if self.weight_dict:
            for key in self.label_keys:
                if key in self.weight_dict:
                    weight_item[key] = self.weight_dict[key]

        if self.transform_fn:
            input_item, label_item = self.transform_fn(input_item, label_item)

        return input_item, label_item, weight_item

    def __len__(self):
        return self._length


class LatentNODataset_time(io.Dataset):
    """LatentNO Dataset for PaddleScience automatic training.

    Args:
        data_name (str): Data name identifier
        data_mode (str): "train" or "val"
        data_normalize (bool): Whether to normalize data
        data_concat (bool): Whether to concatenate x and y1
        input_keys (Tuple[str, ...]): Input keys, such as ("x", "y1")
        label_keys (Tuple[str, ...]): Label keys, such as ("y2",)
        weight_dict (Optional[Dict[str, float]]): Weight dictionary for loss terms
        transform_fn (Optional[Callable]): Optional transform function
    """

    # Whether support batch indexing for speeding up fetching process.
    batch_index: bool = True

    def __init__(
        self,
        data_name: str,
        data_mode: str,
        data_normalize: bool,
        data_concat: bool,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        weight_dict: Optional[Dict[str, float]] = None,
        transform_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.data_name = data_name
        self.data_mode = data_mode
        self.data_normalize = data_normalize
        self.data_concat = data_concat
        self.input_keys = list(input_keys)
        self.label_keys = list(label_keys)
        self.weight_dict = weight_dict or {}
        self.transform_fn = transform_fn

        # Load data
        data_file = osp.join("datas", f"{data_name}_{data_mode}.npy")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        dataset = np.load(data_file, allow_pickle=True).tolist()

        x = dataset["x"].astype("float32")
        y1 = dataset["y1"].astype("float32")
        y2 = dataset["y2"].astype("float32")

        x = x.reshape((x.shape[0], -1, x.shape[-1]))
        y1 = y1.reshape((y1.shape[0], -1, y1.shape[-1]))
        y2 = y2.reshape((y2.shape[0], -1, y2.shape[-1]))
        # Concatenate x and y1 if required
        if data_concat:
            y1 = np.concatenate((x, y1), axis=-1)

        # Initialize normalizer
        self.normalizer = Normalizer(x, y1, y2)

        # Apply normalization if required
        if data_normalize:
            x = self.normalizer.apply_x(x)
            y1 = self.normalizer.apply_y1(y1)
            y2 = self.normalizer.apply_y2(y2)

        # Store data in input and label dictionaries following PaddleScience convention
        self.input = {
            self.input_keys[0]: x,  # "x"
            self.input_keys[1]: y1,  # "y1"
            self.input_keys[2]: y2,  # "y2"
        }

        self.label = {
            self.label_keys[0]: y2,  # "y2"
        }

        self._length = x.shape[0]

    def __getitem__(self, index: int):
        # Get input data
        input_item = {key: value[index] for key, value in self.input.items()}

        # Get label data
        label_item = {key: value[index] for key, value in self.label.items()}

        # Prepare weight dict
        weight_item = {}
        if self.weight_dict:
            for key in self.label_keys:
                if key in self.weight_dict:
                    weight_item[key] = self.weight_dict[key]

        # Apply transform if provided
        if self.transform_fn:
            input_item, label_item = self.transform_fn(input_item, label_item)

        return input_item, label_item, weight_item

    def __len__(self):
        return self._length
