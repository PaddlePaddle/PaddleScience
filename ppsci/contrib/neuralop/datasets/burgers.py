from pathlib import Path

import numpy as np
import paddle

from .tensor_dataset import TensorDataset


def load_burgers_1d(
    data_path, n_train, n_test, batch_train=32, batch_test=100, time=1, grid=[0, 1]
):

    data_path = Path(data_path).joinpath("burgers.pdtensor").as_posix()
    data = paddle.load(data_path)

    x_train = data[0:n_train, :, 0]
    x_test = data[n_train : (n_train + n_test), :, 0]

    y_train = data[0:n_train, :, time]
    y_test = data[n_train : (n_train + n_test), :, time]

    s = x_train.size(-1)

    if grid is not None:
        grid = paddle.linspace(grid[0], grid[1], s + 1)[0:-1].view(1, -1)

        grid_train = grid.tile([n_train, 1])
        grid_test = grid.tile([n_test, 1])

        x_train = paddle.concat((x_train.unsqueeze(1), grid_train.unsqueeze(1)), 1)
        x_test = paddle.concat((x_test.unsqueeze(1), grid_test.unsqueeze(1)), 1)

    train_loader = paddle.io.DataLoader(
        paddle.io.TensorDataset([x_train, y_train]),
        batch_size=batch_train,
        shuffle=False,
    )
    test_loader = paddle.io.DataLoader(
        paddle.io.TensorDataset([x_test, y_test]),
        batch_size=batch_test,
        shuffle=False,
    )

    return train_loader, test_loader


def load_burgers_1dtime(
    data_path,
    n_train,
    n_test,
    batch_size=32,
    batch_size_test=100,
    temporal_length=101,
    spatial_length=128,
    temporal_subsample=1,
    spatial_subsample=1,
    pad=0,
):
    """
    Load burgers.mat data. Given the initial condition (t=0),
    predict timesteps 1 to temporal_length.
    """
    with np.load(data_path) as data:
        x_data = data["input"]
        y_data = data["output"]
        visc = data["visc"]

    x_data = paddle.to_tensor(x_data.astype(np.float32))
    x_data = x_data[:, :spatial_length:spatial_subsample]
    y_data = paddle.to_tensor(y_data.astype(np.float32))
    y_data = y_data[
        :, :temporal_length:temporal_subsample, :spatial_length:spatial_subsample
    ]
    visc = paddle.to_tensor(visc.astype(np.float32)).item()

    x_train = x_data[:n_train]
    y_train = y_data[:n_train]
    x_test = x_data[n_train : n_train + n_test]
    y_test = y_data[n_train : n_train + n_test]

    domain_lengths = [spatial_length / 128, (temporal_length - 1) / 100]
    domain_starts = [0.0, 0.0]

    spatial_length = spatial_length // spatial_subsample
    temporal_length = temporal_length // temporal_subsample

    if pad:
        x_train = paddle.nn.Pad1D(pad)(x_train)
        x_test = paddle.nn.Pad1D(pad)(x_test)
        spatial_length += 2 * pad
        temporal_length += 2 * pad
        incrs = [spatial_subsample / 128, temporal_subsample / 100]
        domain_lengths = [d + incr * pad for d, incr in zip(domain_lengths, incrs)]
        domain_starts = [-incr * pad for incr in incrs]

    # TODO: use include_endpoint arg here
    grid_x = paddle.to_tensor(
        np.linspace(domain_starts[0], domain_lengths[0], spatial_length + 1)[:-1],
        dtype=paddle.float,
    )
    grid_t = paddle.to_tensor(
        np.linspace(domain_starts[1], domain_lengths[1], temporal_length),
        dtype=paddle.float,
    )

    grid_x = grid_x.reshape([1, 1, spatial_length])
    grid_t = grid_t.reshape([1, temporal_length, 1])

    x_train = x_train.reshape([n_train, 1, spatial_length]).tile(
        [1, temporal_length, 1]
    )
    x_test = x_test.reshape([n_test, 1, spatial_length]).tile([1, temporal_length, 1])

    # TODO: add option to not have positional encoding
    x_train = paddle.stack(
        [
            x_train,
            grid_t.tile([n_train, 1, spatial_length]),
            grid_x.tile([n_train, temporal_length, 1]),
        ],
        axis=3,
    )
    x_test = paddle.stack(
        [
            x_test,
            grid_t.tile([n_test, 1, spatial_length]),
            grid_x.tile([n_test, temporal_length, 1]),
        ],
        axis=3,
    )

    x_train = x_train.transpose([0, 3, 1, 2])
    x_test = x_test.transpose([0, 3, 1, 2])
    y_train = y_train.unsqueeze(1)
    y_test = y_test.unsqueeze(1)

    train_db = TensorDataset(x_train, y_train)
    train_loader = paddle.io.DataLoader(train_db, batch_size=batch_size, shuffle=False)

    test_db = TensorDataset(x_test, y_test)
    test_loader = paddle.io.DataLoader(
        test_db, batch_size=batch_size_test, shuffle=False
    )

    output_encoder = None
    test_loaders = {"test": test_loader}

    return train_loader, test_loaders, output_encoder
