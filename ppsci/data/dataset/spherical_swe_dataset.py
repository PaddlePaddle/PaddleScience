from pathlib import Path
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
from paddle import io


class SphericalSWEDataset(io.Dataset):
    """Loads a Spherical Shallow Water equations dataset

    Training contains 200 samples in resolution 32x64.
    Testing contains 50 samples at resolution 32x64 and 50 samples at resolution 64x128.

    Args:
        input_keys (Tuple[str, ...]): Input keys, such as ("input",).
        label_keys (Tuple[str, ...]): Output keys, such as ("output",).
        data_dir (str): The directory to load data from.
        weight_dict (Optional[Dict[str, float]], optional): Define the weight of each constraint variable.
            Defaults to None.
        test_resolutions (Tuple[str, ...], optional): The resolutions to test dataset. Defaults to ["34x64", "64x128"].
        train_resolution (str, optional): The resolutions to train dataset. Defaults to "34x64".
        data_split (str, optional): Specify the dataset split, either 'train' , 'test_32x64',or 'test_64x128'.
            Defaults to "train".

    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        data_dir: str,
        weight_dict: Optional[Dict[str, float]] = None,
        test_resolutions: Tuple[str, ...] = ["34x64", "64x128"],
        train_resolution: str = "34x64",
        data_split: str = "train",
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
        self.data_split = data_split

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
        if self.data_split == "train":
            return self.x_train.shape[0]
        elif self.data_split == "test_32x64":
            return self.x_test_1.shape[0]
        else:
            return self.x_test_2.shape[0]

    def __getitem__(self, index):
        if self.data_split == "train":
            x = self.x_train[index]
            y = self.y_train[index]

        elif self.data_split == "test_32x64":
            x = self.x_test_1[index]
            y = self.y_test_1[index]
        else:
            x = self.x_test_2[index]
            y = self.y_test_2[index]

        input_item = {self.input_keys[0]: x}
        label_item = {self.label_keys[0]: y}
        weight_item = self.weight_dict

        return input_item, label_item, weight_item
