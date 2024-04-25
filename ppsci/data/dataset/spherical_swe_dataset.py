from pathlib import Path
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import paddle
from paddle import io


class SphericalSWEDataset(io.Dataset):
    """Loads a Spherical Shallow Water equations dataset

    Training contains 200 samples in resolution 32x64.
    Testing contains 50 samples at resolution 32x64 and
    50 samples at resolution 64x128.

    Args:
        test_resolutions (List[str,...]): The resolutions to test dataset. Default is ["34x64","64x128"].
        training (str): Wether to use training or test dataset. Default is 'train'.
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        data_dir: str,
        weight_dict: Optional[Dict[str, float]] = None,
        test_resolutions: Tuple[str, ...] = ["34x64", "64x128"],
        train_resolution: str = "34x64",
        training: str = "train",
    ):
        super().__init__()
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.data_dir = data_dir
        self.weight_dict = {} if weight_dict is None else weight_dict
        if weight_dict is not None:
            self.weight_dict = {key: 1.0 for key in self.label_keys}
            self.weight_dict.update(weight_dict)

        self.test_resolutions = test_resolutions
        self.train_resolution = train_resolution
        self.training = training

        # train path
        path_train = (
            Path(self.data_dir)
            .joinpath(f"train_SWE_{self.train_resolution}.npy")
            .as_posix()
        )
        self.x_train, self.y_train = self.read_data(path_train)
        # test path
        path_test_1 = (
            Path(self.data_dir)
            .joinpath(f"test_SWE_{self.test_resolutions[0]}.npy")
            .as_posix()
        )
        self.x_test_1, self.y_test_1 = self.read_data(path_test_1)
        path_test_2 = (
            Path(self.data_dir)
            .joinpath(f"test_SWE_{self.test_resolutions[1]}.npy")
            .as_posix()
        )
        self.x_test_2, self.y_test_2 = self.read_data(path_test_2)

    def read_data(self, path):
        # load with numpy
        data = np.load(path, allow_pickle=True).item()
        x = data["x"].astype("float32")
        y = data["y"].astype("float32")
        del data
        return x, y

    def __len__(self):
        if self.training == "train":
            return self.x_train.shape[0]
        elif self.training == "test_32x64":
            return self.x_test_1.shape[0]
        else:
            return self.x_test_2.shape[0]

    def __getitem__(self, index):
        if self.training == "train":
            x = self.x_train[index]
            y = self.y_train[index]

        elif self.training == "test_32x64":
            x = self.x_test_1[index]
            y = self.y_test_1[index]
        else:
            x = self.x_test_2[index]
            y = self.y_test_2[index]

        input_item = {self.input_keys[0]: x}
        label_item = {self.label_keys[0]: y}
        weight_item = self.weight_dict

        return input_item, label_item, weight_item


if __name__ == "__main__":
    dataset = SphericalSWEDataset(
        input_keys=("x",),
        label_keys=("y",),
        data_dir="./datasets/SWE/",
        weight_dict=None,
        test_resolutions=["32x64", "64x128"],
        train_resolution="32x64",
        training="test_64x128",
    )

    train_loader = paddle.io.DataLoader(
        dataset, batch_size=10, shuffle=False, num_workers=0, drop_last=False
    )

    for batch_index, (data, target, _) in enumerate(train_loader):
        # 这里data是一个形状为[batch_size, channels, height, width]的tensor
        # target是一个包含batch_size个元素的一维tensor，每个元素是对应data中图像的标签

        # 在这里进行训练或验证等操作
        print(f"Batch {batch_index}:")
        print(data["x"].shape)
        # print(data['x'])
        print(target["y"].shape)
        # print(data['x'])
