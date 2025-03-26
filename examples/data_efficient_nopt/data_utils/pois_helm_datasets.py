import logging

import h5py
import numpy as np
import paddle
from paddle.io import DataLoader
from paddle.io import Dataset
from paddle.io import DistributedBatchSampler

from .masking_generator import MaskingGenerator


def get_data_loader(params, location, distributed, train=True):
    # masking (float): %INVISIBLE pixels
    transform = paddle.to_tensor
    print(
        f"Current batch size for {'train' if train else 'val'} loader is {int(params.batch_size)}"
    )

    dataset = PoisHelmDataset(params, location, transform, train)

    sampler = DistributedBatchSampler(dataset, shuffle=train) if distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=int(params.batch_size),
        num_workers=params.num_data_workers,
        shuffle=False,  # (sampler is None),
        drop_last=True,
    )
    print(f"There are {len(dataset)} samples used")
    return dataloader, dataset, sampler


class PoisHelmDataset(Dataset):
    def __init__(self, params, location, transform, train):
        self.transform = transform
        self.params = params
        self.location = location
        self.train = train
        self.masking = params.mask_ratio if hasattr(params, "mask_ratio") else False
        if hasattr(self.params, "subsample") and (self.train):
            self.subsample = self.params.subsample
        else:
            self.subsample = 1  # subsample only if training
        self.scales = None
        self._get_files_stats()
        if isinstance(self.masking, float):  # and self.masking > 0:
            self.mask_generator = MaskingGenerator(
                (self.img_shape_x, self.img_shape_y), self.masking
            )
        file = self._open_file(self.location)
        self.data = file["fields"]
        if self.train:
            if hasattr(self.params, "train_rand_idx_path"):
                self.train_rand_idx = np.load(self.params.train_rand_idx_path)
                logging.info("Randomizing train dataset using given random index path")
            else:
                self.train_rand_idx = range(
                    self.data.shape[0]
                )  # np.random.permutation(self.data.shape[0])
            self.train_rand_idx = self.train_rand_idx[self.pt_idxs[0] : self.pt_idxs[1]]
            self.data = self.data[()][self.train_rand_idx, ...]
            logging.info(
                "Getting only data idx for training set for length: {}".format(
                    len(self.train_rand_idx)
                )
            )
        if "tensor" in list(file.keys()):
            self.tensor = file["tensor"]
            if self.train:
                self.tensor = self.tensor[()][self.train_rand_idx, ...]
        else:
            self.tensor = None

    def _get_files_stats(self):
        self.file = self.location
        with h5py.File(self.file, "r") as _f:
            logging.info("Getting file stats from {}".format(self.file))
            if len(_f["fields"].shape) == 4:
                self.n_demos = None
                self.n_samples = _f["fields"].shape[0]
                self.img_shape_x = _f["fields"].shape[2]
                self.img_shape_y = _f["fields"].shape[3]
                self.in_channels = _f["fields"].shape[1] - 1
            elif len(_f["fields"].shape) == 5:
                # if hasattr(self.params, "n_demos") and self.params.n_demos > 0:
                self.n_demos = _f["fields"].shape[2]
                assert self.n_demos >= self.params.n_demos
                self.n_samples = _f["fields"].shape[0]
                self.img_shape_x = _f["fields"].shape[3]
                self.img_shape_y = _f["fields"].shape[4]
                self.in_channels = _f["fields"].shape[1] - 1
            if "tensor" in list(_f.keys()):
                self.tensor_shape = _f["tensor"].shape[1]
            else:
                self.tensor_shape = 0
        if self.train:
            if hasattr(self.params, "pt_split"):  # pretrain-train split
                self.pt_split = self.params.pt_split
            else:
                self.pt_split = [0.9, 0.1]
            logging.info(
                "Split training set into {} for pretrain, {} for train. ".format(
                    self.pt_split[0], self.pt_split[1]
                )
            )
            if hasattr(self.params, "pt"):  # pretrain or train
                self.pt = self.params.pt
            else:
                self.pt = "train"
            if int(sum(self.pt_split)) == 1:
                self.n_samples *= self.pt_split[
                    -1 if self.pt == "train" else 0
                ]  # if split is float summed to 1, separate based on the two portions
            else:
                assert int(sum(self.pt_split)) <= self.n_samples
                self.n_samples = self.pt_split[
                    -1 if self.pt == "train" else 0
                ]  # if split is int, separate based on the two numbers
            self.n_samples = int(self.n_samples)
            self.pt_idxs = (
                [-self.n_samples, None] if self.pt == "train" else [0, self.n_samples]
            )
        self.n_samples /= self.subsample
        self.n_samples = int(self.n_samples)
        logging.info(
            "Found data at path {}. Number of examples: {}. Image Shape: {} x {}".format(
                self.location, self.n_samples, self.img_shape_x, self.img_shape_y
            )
        )
        if hasattr(self.params, "scales_path"):
            self.scales = np.load(self.params.scales_path)
            self.scales = np.array([s if s != 0 else 1 for s in self.scales])
            self.scales = self.scales.astype("float32")
            measure_x = self.scales[-2] / self.img_shape_x
            measure_y = self.scales[-1] / self.img_shape_y
            self.measure = measure_x * measure_y
            logging.info(
                "Scales for PDE are (source, tensor, sol, domain): {}".format(
                    self.scales
                )
            )
            logging.info(
                "Measure of the set is lx/nx * ly/ny =  {}/{} * {}/{}".format(
                    self.scales[-2], self.img_shape_x, self.scales[-1], self.img_shape_y
                )
            )

    def __len__(self):
        return self.n_samples

    def _open_file(self, path):
        return h5py.File(path, "r")

    def _getitem_single(self, local_idx):
        if self.params.n_demos == 0:
            if self.n_demos is None:
                X = self.data[local_idx, 0 : self.in_channels]
            else:
                X = self.data[local_idx, 0 : self.in_channels, 0]
        else:
            if self.train:
                # numpy choice => Tensor => paddle.take_along_axis
                demo_indices = np.random.choice(
                    range(self.n_demos), self.params.n_demos, replace=False
                )  # TODO: do we allow replace? (i.e. duplicated demos)
                X = np.take(
                    self.data[local_idx, 0 : self.in_channels],
                    np.insert(demo_indices, 0, 0),
                    1,
                )
            else:
                X = self.data[
                    local_idx, 0 : self.in_channels, : self.params.n_demos + 1
                ]  # +1 for query
        if self.tensor is not None:  # append coefficient tensor to channels
            tensor = []
            for tidx in range(self.tensor_shape):
                coef = np.full(
                    (1, self.img_shape_x, self.img_shape_y),
                    self.tensor[local_idx, tidx],
                )
                tensor.append(coef)
            X = np.concatenate([X] + tensor, axis=0).astype("float32")

        if self.scales is not None:
            f_norm = np.linalg.norm(X[0]) * self.measure
            f_scaling = f_norm / self.scales[0]
            X = (
                X / f_scaling
            )  # ensures that 10f and 10k for example, have the same input
            # scale the tensors
            # we don't have tensors (coefficients) in input now
            X[self.in_channels :] = (
                X[self.in_channels :]
                / self.scales[
                    self.in_channels : (self.in_channels + self.tensor_shape),
                    None,
                    None,
                ]
            )

        X = self.transform(X)

        if self.params.n_demos == 0:
            if self.n_demos is None:
                y = self.data[local_idx, self.in_channels :]
            else:
                y = self.data[local_idx, self.in_channels :, 0]
        else:
            if self.train:
                y = np.take(
                    self.data[local_idx, self.in_channels :],
                    np.insert(demo_indices, 0, 0),
                    1,
                )
            else:
                y = self.data[
                    local_idx, self.in_channels :, : self.params.n_demos + 1
                ]  # +1 for query
        y = self.transform(y)

        if isinstance(self.masking, float):
            mask = self.mask_generator().reshape(1, self.img_shape_x, self.img_shape_y)
            return X, y, mask
        else:
            return X, y

    def __getitem__(self, idx):
        local_idx = int(idx * self.subsample)
        if self.params.n_demos > 0 and self.n_demos is None:
            # manually select demos from all samples; coefficients are different
            candidate_idx = list(range(self.n_samples))
            candidate_idx.remove(idx)
            idx_range = (
                (
                    np.random.choice(
                        candidate_idx, size=self.params.n_demos, replace=False
                    )
                    * self.subsample
                )
                .astype(int)
                .tolist()
            )
            # concatenate X and Y into channels
            idx_range.append(local_idx)
            X, Y, y = [], [], []
            _X, y = self._getitem_single(idx_range[-1])
            X.append(_X)
            for idx in idx_range[:-1]:
                _X, _y = self._getitem_single(idx)
                X.append(_X)
                Y.append(_y)
            X += Y
            X = paddle.concat(X, axis=0)
            return X, y
        else:
            mask = None
            _data = self._getitem_single(local_idx)
            if len(_data) == 2:
                X, y = _data
            else:
                X, y, mask = _data
            if self.params.n_demos > 0:
                # get demos with the same coefficients
                X = paddle.concat(
                    [
                        X.view([-1, self.img_shape_x, self.img_shape_y]),
                        y[:, 1:].view([-1, self.img_shape_x, self.img_shape_y]),
                    ],
                    axis=0,
                )
                y = y[:, 0]
            if mask is None:
                return X, y
            else:
                return X, y, mask
