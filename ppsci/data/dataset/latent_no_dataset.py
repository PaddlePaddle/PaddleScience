import os
import os.path as osp
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import paddle


class Normalizer:
    def __init__(self, x, y1, y2):
        self.x_flag = False
        self.y1_flag = False
        self.y2_flag = False
        old_x_shape = x.shape
        old_y1_shape = y1.shape
        old_y2_shape = y2.shape
        x = paddle.reshape(x, (-1, x.shape[-1]))
        y1 = paddle.reshape(y1, (-1, y1.shape[-1]))
        y2 = paddle.reshape(y2, (-1, y2.shape[-1]))
        self.x_mean = paddle.mean(x, axis=0)
        self.x_std = paddle.std(x, axis=0) + 1e-8
        self.y1_mean = paddle.mean(y1, axis=0)
        self.y1_std = paddle.std(y1, axis=0) + 1e-8
        self.y2_mean = paddle.mean(y2, axis=0)
        self.y2_std = paddle.std(y2, axis=0) + 1e-8
        x = paddle.reshape(x, old_x_shape)
        y1 = paddle.reshape(y1, old_y1_shape)
        y2 = paddle.reshape(y2, old_y2_shape)

    def is_apply_x(self):
        return self.x_flag

    def is_apply_y1(self):
        return self.y1_flag

    def is_apply_y2(self):
        return self.y2_flag

    def apply_x(self, x, device, inverse=False):
        self.x_mean = self.x_mean.to(device)
        self.x_std = self.x_std.to(device)

        old_x_shape = x.shape
        x = paddle.reshape(x, (-1, x.shape[-1]))
        if not inverse:
            x = (x - self.x_mean) / self.x_std
            self.x_flag = True
        else:
            x = x * self.x_std + self.x_mean
        x = paddle.reshape(x, old_x_shape)
        return x

    def apply_y1(self, y1, device, inverse=False):
        self.y1_mean = self.y1_mean.to(device)
        self.y1_std = self.y1_std.to(device)

        old_y1_shape = y1.shape
        y1 = paddle.reshape(y1, (-1, y1.shape[-1]))
        if not inverse:
            y1 = (y1 - self.y1_mean) / self.y1_std
            self.y1_flag = True
        else:
            y1 = y1 * self.y1_std + self.y1_mean
        y1 = paddle.reshape(y1, old_y1_shape)
        return y1

    def apply_y2(self, y2, device, inverse=False):
        self.y2_mean = self.y2_mean.to(device)
        self.y2_std = self.y2_std.to(device)

        old_y2_shape = y2.shape
        y2 = paddle.reshape(y2, (-1, y2.shape[-1]))
        if not inverse:
            y2 = (y2 - self.y2_mean) / self.y2_std
            self.y2_flag = True
        else:
            y2 = y2 * self.y2_std + self.y2_mean
        y2 = paddle.reshape(y2, old_y2_shape)
        return y2


class LatentNODataset(paddle.io.Dataset):
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

        data_file = osp.join("datas", f"{data_name}_{data_mode}.npy")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        dataset = np.load(data_file, allow_pickle=True).tolist()

        x = np.array(dataset["x"], dtype=np.float32)
        y1 = np.array(dataset["y1"], dtype=np.float32)
        y2 = np.array(dataset["y2"], dtype=np.float32)

        x = np.reshape(x, (x.shape[0], -1, x.shape[-1]))
        y1 = np.reshape(y1, (y1.shape[0], -1, y1.shape[-1]))
        y2 = np.reshape(y2, (y2.shape[0], -1, y2.shape[-1]))

        if data_concat:
            y1 = np.concatenate((x, y1), axis=-1)

        x_tensor = paddle.to_tensor(x)
        y1_tensor = paddle.to_tensor(y1)
        y2_tensor = paddle.to_tensor(y2)

        self.normalizer = Normalizer(x_tensor, y1_tensor, y2_tensor)

        if data_normalize:
            x = self.normalizer.apply_x(x_tensor, "cpu").numpy()
            y1 = self.normalizer.apply_y1(y1_tensor, "cpu").numpy()
            y2 = self.normalizer.apply_y2(y2_tensor, "cpu").numpy()
        else:
            x = x
            y1 = y1
            y2 = y2

        self.input_data = {
            self.input_keys[0]: x,
            self.input_keys[1]: y1,
        }

        self.label_data = {
            self.label_keys[0]: y2,
        }

        self._length = x.shape[0]

    def __getitem__(self, index: int):
        input_item = {
            key: paddle.to_tensor(value[index], dtype="float32")
            for key, value in self.input_data.items()
        }

        label_item = {
            key: paddle.to_tensor(value[index], dtype="float32")
            for key, value in self.label_data.items()
        }

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


class LatentNODataset_time(paddle.io.Dataset):
    """LatentNO Dataset for PaddleScience automatic training with time series.

    Args:
        data_name (str): Data name identifier
        data_mode (str): "train" or "val"
        data_normalize (bool): Whether to normalize data
        data_concat (bool): Whether to concatenate x and y1
        input_keys (Tuple[str, ...]): Input keys, such as ("x", "y1", "y2")
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

        data_file = osp.join("datas", f"{data_name}_{data_mode}.npy")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        dataset = np.load(data_file, allow_pickle=True).tolist()

        x = np.array(dataset["x"], dtype=np.float32)
        y1 = np.array(dataset["y1"], dtype=np.float32)
        y2 = np.array(dataset["y2"], dtype=np.float32)

        x = np.reshape(x, (x.shape[0], -1, x.shape[-1]))
        y1 = np.reshape(y1, (y1.shape[0], -1, y1.shape[-1]))
        y2 = np.reshape(y2, (y2.shape[0], -1, y2.shape[-1]))

        if data_concat:
            y1 = np.concatenate((x, y1), axis=-1)

        x_tensor = paddle.to_tensor(x)
        y1_tensor = paddle.to_tensor(y1)
        y2_tensor = paddle.to_tensor(y2)

        self.normalizer = Normalizer(x_tensor, y1_tensor, y2_tensor)

        if data_normalize:
            x = self.normalizer.apply_x(x_tensor, "cpu").numpy()
            y1 = self.normalizer.apply_y1(y1_tensor, "cpu").numpy()
            y2 = self.normalizer.apply_y2(y2_tensor, "cpu").numpy()
        else:
            x = x
            y1 = y1
            y2 = y2

        self.input_data = {
            self.input_keys[0]: x,
            self.input_keys[1]: y1,
            self.input_keys[2]: y2,
        }

        self.label_data = {
            self.label_keys[0]: y2,
        }

        self._length = x.shape[0]

    def __getitem__(self, index: int):
        input_item = {
            key: paddle.to_tensor(value[index], dtype="float32")
            for key, value in self.input_data.items()
        }

        label_item = {
            key: paddle.to_tensor(value[index], dtype="float32")
            for key, value in self.label_data.items()
        }

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
