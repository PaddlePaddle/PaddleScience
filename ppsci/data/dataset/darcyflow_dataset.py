from pathlib import Path
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import paddle
from paddle import io


# normalization, pointwise gaussian
class UnitGaussianNormalizer:
    def __init__(self, x, eps=1e-7, reduce_dim=[0], verbose=True):
        super().__init__()
        n_samples, *shape = x.shape
        self.sample_shape = shape
        self.verbose = verbose
        self.reduce_dim = reduce_dim

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = paddle.mean(x, reduce_dim, keepdim=True).squeeze(0)
        self.std = paddle.std(x, reduce_dim, keepdim=True).squeeze(0)
        self.eps = eps

        if verbose:
            print(
                f"UnitGaussianNormalizer init on {n_samples}, reducing over {reduce_dim}, samples of shape {shape}."
            )
            print(f"   Mean and std of shape {self.mean.shape}, eps={eps}")

    def encode(self, x):

        x -= self.mean
        x /= self.std + self.eps
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps  # T*batch*n
                mean = self.mean[:, sample_idx]

        # x is in shape of batch*n or T*batch*n
        x *= std
        x += mean

        return x


def get_grid_positional_encoding(
    input_tensor, grid_boundaries=[[0, 1], [0, 1]], channel_dim=1
):
    """Appends grid positional encoding to an input tensor, concatenating as additional dimensions along the channels.

    Args:
        input_tensor (paddle.Tensor): The input tensor.
        grid_boundaries (list, optional): The boundaries of the grid. Defaults to [[0, 1], [0, 1]].
        channel_dim (int, optional): The location of unsqueeze. Defaults to 1.
    """

    shape = list(input_tensor.shape)
    if len(shape) == 2:
        height, width = shape
    else:
        _, height, width = shape

    xt = paddle.linspace(grid_boundaries[0][0], grid_boundaries[0][1], height + 1)[:-1]
    yt = paddle.linspace(grid_boundaries[1][0], grid_boundaries[1][1], width + 1)[:-1]

    grid_x, grid_y = paddle.meshgrid(xt, yt, indexing="ij")

    if len(shape) == 2:
        grid_x = grid_x.unsqueeze(channel_dim)
        grid_y = grid_y.unsqueeze(channel_dim)
    else:
        grid_x = grid_x.unsqueeze(0).unsqueeze(channel_dim)
        grid_y = grid_y.unsqueeze(0).unsqueeze(channel_dim)

    return grid_x, grid_y


def regular_grid(spatial_dims, grid_boundaries=[[0, 1], [0, 1]]):
    """
    Appends grid positional encoding to an input tensor, concatenating as additional dimensions along the channels
    """
    height, width = spatial_dims

    xt = paddle.linspace(grid_boundaries[0][0], grid_boundaries[0][1], height + 1)[:-1]
    yt = paddle.linspace(grid_boundaries[1][0], grid_boundaries[1][1], width + 1)[:-1]

    grid_x, grid_y = paddle.meshgrid(xt, yt, indexing="ij")

    grid_x = grid_x.tile((1, 1))
    grid_y = grid_y.tile((1, 1))

    return grid_x, grid_y


class PositionalEmbedding2D:
    def __init__(self, grid_boundaries=[[0, 1], [0, 1]]):
        self.grid_boundaries = grid_boundaries
        self._grid = None
        self._res = None

    def grid(self, spatial_dims, dtype):
        """Grid generates 2D grid needed for pos encoding
        and caches the grid associated with MRU resolution

        Args:
            spatial_dims (tuple[int,...]): Sizes of spatial resolution.
            dtype (str): Dtype to encode data.

        Returns:
            paddle.Tensor: Output grids to concatenate
        """
        # handle case of multiple train resolutions
        if self._grid is None or self._res != spatial_dims:
            grid_x, grid_y = regular_grid(
                spatial_dims, grid_boundaries=self.grid_boundaries
            )

            grid_x = grid_x.astype(dtype).unsqueeze(0).unsqueeze(0)
            grid_y = grid_y.astype(dtype).unsqueeze(0).unsqueeze(0)
            self._grid = grid_x, grid_y
            self._res = spatial_dims

        return self._grid

    def __call__(self, data):
        if data.ndim == 3:
            data = data.unsqueeze(0)
        x, y = self.grid(data.shape[-2:], data.dtype)
        out = paddle.concat(
            (data, x.expand([1, -1, -1, -1]), y.expand([1, -1, -1, -1])), axis=1
        )
        return out.squeeze(0)


class DarcyFlowDataset(io.Dataset):
    """Loads a small Darcy-Flow dataset

    Training contains 1000 samples in resolution 16x16.
    Testing contains 100 samples at resolution 16x16 and
    50 samples at resolution 32x32.

    Args:
        input_keys (Tuple[str, ...]): Input keys, such as ("input",).
        label_keys (Tuple[str, ...]): Output keys, such as ("output",).
        data_dir (str): The directory to load data from.
        weight_dict (Optional[Dict[str, float]], optional): Define the weight of each constraint variable. Defaults to None.
        test_resolutions (List[int,...]): The resolutions to test dataset. Default is [16, 32].
        grid_boundaries (List[int,...]): The boundaries of the grid. Default is [[0,1],[0,1]].
        positional_encoding (bool): Whether to use positional encoding. Default is True
        encode_input (bool): Whether to encode the input. Default is False
        encode_output (bool): Whether to encode the output. Default is True
        encoding (str): The type of encoding. Default is 'channel-wise'.
        channel_dim (int): The location of unsqueeze. Default is 1.
            where to put the channel dimension. Defaults size is batch, channel, height, width
        data_split (str): Wether to use training or test dataset. Default is 'train'.
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        data_dir: str,
        weight_dict: Optional[Dict[str, float]] = None,
        test_resolutions: Tuple[int, ...] = [32],
        train_resolution: int = 32,
        grid_boundaries: Tuple[Tuple[int, ...], ...] = [[0, 1], [0, 1]],
        positional_encoding: bool = True,
        encode_input: bool = False,
        encode_output: bool = True,
        encoding: str = "channel-wise",
        channel_dim: int = 1,
        data_split: str = "train",
    ):
        super().__init__()
        for res in test_resolutions:
            if res not in [16, 32]:
                raise ValueError(
                    f"Only 32 and 64 are supported for test resolution, but got {test_resolutions}"
                )

        self.input_keys = input_keys
        self.label_keys = label_keys
        self.data_dir = data_dir
        self.weight_dict = {} if weight_dict is None else weight_dict
        if weight_dict is not None:
            self.weight_dict = {key: 1.0 for key in self.label_keys}
            self.weight_dict.update(weight_dict)

        self.test_resolutions = test_resolutions
        self.train_resolution = train_resolution
        self.grid_boundaries = grid_boundaries
        self.positional_encoding = positional_encoding
        self.encode_input = encode_input
        self.encode_output = encode_output
        self.encoding = encoding
        self.channel_dim = channel_dim
        self.data_split = data_split

        # train path
        path_train = (
            Path(self.data_dir)
            .joinpath(f"darcy_train_{self.train_resolution}.npy")
            .as_posix()
        )
        self.x_train, self.y_train = self.read_data(path_train)
        # test path
        path_test_1 = (
            Path(self.data_dir)
            .joinpath(f"darcy_test_{self.test_resolutions[0]}.npy")
            .as_posix()
        )
        self.x_test_1, self.y_test_1 = self.read_data(path_test_1)
        path_test_2 = (
            Path(self.data_dir)
            .joinpath(f"darcy_test_{self.test_resolutions[1]}.npy")
            .as_posix()
        )
        self.x_test_2, self.y_test_2 = self.read_data(path_test_2)

        # input encoder
        if self.encode_input:
            self.input_encoder = self.encode_data(self.x_train)
            self.x_train = self.input_encoder.encode(self.x_train)
            self.x_test_1 = self.input_encoder.encode(self.x_test_1)
            self.x_test_2 = self.input_encoder.encode(self.x_test_2)
        else:
            self.input_encoder = None
        # output encoder
        if self.encode_output:
            self.output_encoder = self.encode_data(self.y_train)
            self.y_train = self.output_encoder.encode(self.y_train)
        else:
            self.output_encoder = None

        if positional_encoding:
            self.transform_x = PositionalEmbedding2D(grid_boundaries)

    def read_data(self, path):
        # load with numpy
        data = np.load(path, allow_pickle=True).item()
        x = (
            paddle.to_tensor(data["x"])
            .unsqueeze(self.channel_dim)
            .astype("float32")
            .clone()
        )
        y = paddle.to_tensor(data["y"]).unsqueeze(self.channel_dim).clone()
        del data
        return x, y

    def encode_data(self, data):
        if self.encoding == "channel-wise":
            reduce_dims = list(range(data.ndim))
        elif self.encoding == "pixel-wise":
            reduce_dims = [0]
        input_encoder = UnitGaussianNormalizer(data, reduce_dim=reduce_dims)
        return input_encoder

    def __len__(self):
        if self.data_split == "train":
            return self.x_train.shape[0]
        elif self.data_split == "test_16x16":
            return self.x_test_1.shape[0]
        else:
            return self.x_test_2.shape[0]

    def __getitem__(self, index):
        if self.data_split == "train":
            x = self.x_train[index]
            y = self.y_train[index]

        elif self.data_split == "test_16x16":
            x = self.x_test_1[index]
            y = self.y_test_1[index]
        else:
            x = self.x_test_2[index]
            y = self.y_test_2[index]

        if self.transform_x is not None:
            x = self.transform_x(x)

        input_item = {self.input_keys[0]: x}
        label_item = {self.label_keys[0]: y}
        weight_item = self.weight_dict

        return input_item, label_item, weight_item
