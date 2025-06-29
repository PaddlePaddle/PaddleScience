# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# refs: https://github.com/delta-lab-ai/data_efficient_nopt

"""
Remember to parameterize the file paths eventually
"""
import glob
import logging
import os
from typing import Iterator
from typing import TypeVar

import h5py
import numpy as np
import paddle
from paddle.io import DataLoader
from paddle.io import Dataset
from paddle.io import DistributedBatchSampler
from paddle.io import RandomSampler
from paddle.io import Sampler

__all__ = [
    "MultisetSampler",
]

T_co = TypeVar("T_co", covariant=True)
broken_paths = []


class BaseHDF5DirectoryDataset(Dataset):
    """
    Base class for data loaders. Returns data in T x B x C x H x W format.

    Note - doesn't currently normalize because the data is on wildly different
    scales but probably should.

    Split is provided so I can be lazy and not separate out HDF5 files.

    Takes in path to directory of HDF5 files to construct dset.

    Args:
        path (str): Path to directory of HDF5 files
        include_string (str): Only include files with this string in name
        n_steps (int): Number of steps to include in each sample
        dt (int): Time step between samples
        split (str): train/val/test split
        train_val_test (tuple): Percent of data to use for train/val/test
        subname (str): Name to use for dataset
        split_level (str): 'sample' or 'file' - whether to split by samples within a file
                        (useful for data segmented by parameters) or file (mostly INS right now)
    """

    def __init__(
        self,
        path,
        include_string="",
        n_steps=1,
        dt=1,
        split="train",
        train_val_test=None,
        subname=None,
        extra_specific=False,
        rollout=1,
    ):
        super().__init__()
        self.path = path
        self.split = split
        self.extra_specific = extra_specific
        if subname is None:
            self.subname = path.split("/")[-1]
        else:
            self.subname = subname
        self.dt = dt
        self.rollout = rollout
        self.n_steps = n_steps
        self.include_string = include_string
        self.train_val_test = train_val_test
        self.partition = {"train": 0, "val": 1, "test": 2}[split]
        (
            self.time_index,
            self.sample_index,
            self.field_names,
            self.type,
            self.split_level,
        ) = self._specifics()
        self._get_directory_stats(path)
        if self.extra_specific:
            self.title = self.more_specific_title(self.type, path, include_string)
        else:
            self.title = self.type

    def get_name(self, full_name=False):
        if full_name:
            return self.subname + "_" + self.type
        else:
            return self.type

    def more_specific_title(self, type, path, include_string):
        """
        Override this to add more info to the dataset name
        """
        return type

    @staticmethod
    def _specifics():
        raise NotImplementedError

    def get_per_file_dsets(self):
        if self.split_level == "file" or len(self.files_paths) == 1:
            return [self]
        else:
            sub_dsets = []
            for file in self.files_paths:
                subd = self.__class__(
                    self.path,
                    file,
                    n_steps=self.n_steps,
                    dt=self.dt,
                    split=self.split,
                    train_val_test=self.train_val_test,
                    subname=self.subname,
                    extra_specific=True,
                )
                sub_dsets.append(subd)
            return sub_dsets

    def _get_specific_stats(self, f):
        raise NotImplementedError

    def _get_specific_bcs(self, f):
        raise NotImplementedError

    def _reconstruct_sample(self, file, sample_idx, time_idx, n_steps):
        raise NotImplementedError

    def _get_directory_stats(self, path):
        self.files_paths = glob.glob(path + "/*.h5") + glob.glob(path + "/*.hdf5")
        self.files_paths.sort()
        self.n_files = len(self.files_paths)
        self.file_steps = []
        self.file_nsteps = []
        self.file_samples = []
        self.split_offsets = []
        self.offsets = [0]
        file_paths = []
        for file in self.files_paths:
            if len(self.include_string) > 0 and self.include_string not in file:
                continue
            elif file in broken_paths:
                continue
            else:
                file_paths.append(file)
                try:
                    with h5py.File(file, "r") as _f:
                        samples, steps = self._get_specific_stats(_f)
                        if steps - self.n_steps - (self.dt - 1) < 1:
                            print(
                                "WARNING: File {} has {} steps, but n_steps is {}. Setting file steps = max allowable.".format(
                                    file, steps, self.n_steps
                                )
                            )
                            file_nsteps = steps - self.dt
                        else:
                            file_nsteps = self.n_steps
                        self.file_nsteps.append(file_nsteps)
                        self.file_steps.append(steps - file_nsteps - (self.dt - 1))
                        if self.split_level == "sample":
                            partition = self.partition
                            sample_per_part = np.ceil(
                                np.absolute(np.array(self.train_val_test) * samples)
                            ).astype(int)
                            sample_per_part[2] = max(
                                samples - sample_per_part[0] - sample_per_part[1], 0
                            )
                            if self.train_val_test[0] >= 0:
                                self.split_offsets.append(
                                    self.file_steps[-1]
                                    * sum(sample_per_part[:partition])
                                )
                            else:
                                if partition == 0:
                                    self.split_offsets.append(
                                        self.file_steps[-1] * (1 - sum(sample_per_part))
                                    )
                                else:
                                    self.split_offsets.append(
                                        self.file_steps[-1]
                                        * (1 - sum(sample_per_part[partition:]))
                                    )
                            split_samples = sample_per_part[partition]
                        else:
                            split_samples = samples
                        self.file_samples.append(split_samples)
                        self.offsets.append(
                            self.offsets[-1]
                            + (steps - file_nsteps - (self.dt - 1)) * split_samples
                        )
                except:  # noqa
                    print(
                        "WARNING: Failed to open file {}. Continuing without it.".format(
                            file
                        )
                    )
                    raise RuntimeError("Failed to open file {}".format(file))
        self.files_paths = file_paths
        self.offsets[0] = -1
        self.files = [None for _ in self.files_paths]
        self.len = self.offsets[-1]
        if self.split_level == "file":
            if self.train_val_test is None:
                print(
                    "WARNING: No train/val/test split specified. Using all data for training."
                )
                self.split_offset = 0
                self.len = self.offsets[-1]
            else:
                print("Using train/val/test split: {}".format(self.train_val_test))
                total_samples = sum(self.file_samples)
                if (
                    self.train_val_test[1] * total_samples < 1
                    or self.train_val_test[2] * total_samples < 1
                ):
                    ideal_split_offsets = [
                        self.train_val_test[i] * total_samples for i in range(3)
                    ]
                    ideal_split_offsets = [
                        int(value) if value >= 1 else value
                        for value in ideal_split_offsets
                    ]
                else:
                    ideal_split_offsets = [
                        int(self.train_val_test[i] * total_samples) for i in range(3)
                    ]
                if ideal_split_offsets[0] > 0:
                    end_ind = 0
                elif ideal_split_offsets[0] == 0:
                    ideal_split_offsets[0] = abs(self.train_val_test[0] * total_samples)
                    assert ideal_split_offsets[0] < 1 and ideal_split_offsets[0] > 0
                    end_ind = total_samples - round(sum(ideal_split_offsets[1:])) - 1
                else:
                    ideal_split_offsets[0] = -ideal_split_offsets[0]
                    end_ind = (
                        total_samples
                        - round(sum(ideal_split_offsets[1:]))
                        - ideal_split_offsets[0]
                    )
                for i in range(self.partition + 1):
                    run_sum = 0
                    start_ind = end_ind
                    for samples, steps in zip(self.file_samples, self.file_steps):
                        run_sum += samples
                        if run_sum <= ideal_split_offsets[i]:
                            end_ind += round(samples * (steps))
                            if run_sum == ideal_split_offsets[i]:
                                break
                        else:
                            end_ind += round(
                                np.abs((run_sum - samples) - ideal_split_offsets[i])
                                * (steps)
                            )
                            break
                start_ind, end_ind = int(start_ind), int(end_ind)
                self.split_offset = start_ind
                self.len = end_ind - start_ind

    def _open_file(self, file_ind):
        _file = h5py.File(self.files_paths[file_ind], "r")
        self.files[file_ind] = _file

    def __getitem__(self, index):
        if self.split_level == "file":
            index = index + self.split_offset
        file_idx = int(np.searchsorted(self.offsets, index, side="right") - 1)
        nsteps = self.file_nsteps[file_idx] + self.rollout - 1
        local_idx = index - max(self.offsets[file_idx], 0)
        if self.split_level == "sample":
            sample_idx = (local_idx + self.split_offsets[file_idx]) // self.file_steps[
                file_idx
            ]
        else:
            sample_idx = local_idx // self.file_steps[file_idx]
        time_idx = local_idx % self.file_steps[file_idx]

        if self.files[file_idx] is None:
            self._open_file(file_idx)

        time_idx = (
            time_idx - self.dt if time_idx >= self.file_steps[file_idx] else time_idx
        )
        time_idx += nsteps
        trajectory = self._reconstruct_sample(
            self.files[file_idx], sample_idx, time_idx, nsteps
        )
        try:
            trajectory = self._reconstruct_sample(
                self.files[file_idx], sample_idx, time_idx, nsteps
            )
        except:  # noqa:
            raise RuntimeError(
                f"Failed to reconstruct sample for file {self.files_paths[file_idx]} sample {sample_idx} time {time_idx}"
            )
        return trajectory[:-1], trajectory[-1]

    def __len__(self):
        return self.len


class SWEDataset(BaseHDF5DirectoryDataset):
    @staticmethod
    def _specifics():
        time_index = 0
        sample_index = None
        field_names = ["h"]
        type = "swe"
        split_level = "sample"
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f):
        samples = list(f.keys())
        steps = f[samples[0]]["data"].shape[0]
        return len(samples), steps

    def _get_specific_bcs(self, f):
        return [0, 0]

    def _reconstruct_sample(self, file, sample_idx, time_idx, n_steps):
        samples = list(file.keys())
        return file[samples[sample_idx]]["data"][
            time_idx - n_steps * self.dt : time_idx + self.dt
        ].transpose(0, 3, 1, 2)


class DiffRe2DDataset(BaseHDF5DirectoryDataset):
    @staticmethod
    def _specifics():
        time_index = 0
        sample_index = None
        field_names = ["activator", "inhibitor"]
        type = "diffre2d"
        split_level = "sample"
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f):
        samples = list(f.keys())
        steps = f[samples[0]]["data"].shape[0]
        return len(samples), steps

    def _get_specific_bcs(self, f):
        return [0, 0]

    def _reconstruct_sample(self, file, sample_idx, time_idx, n_steps):
        samples = list(file.keys())
        return file[samples[sample_idx]]["data"][
            time_idx - n_steps * self.dt : time_idx + self.dt
        ].transpose(0, 3, 1, 2)


class IncompNSDataset(BaseHDF5DirectoryDataset):
    """
    Order Vx, Vy, "particles"
    """

    @staticmethod
    def _specifics():
        time_index = 1
        sample_index = 0
        field_names = ["Vx", "Vy", "particles"]
        type = "incompNS"
        split_level = "file"
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f):
        samples = f["velocity"].shape[0]
        steps = f["velocity"].shape[1]
        return samples, steps

    def _reconstruct_sample(self, file, sample_idx, time_idx, n_steps):
        velocity = file["velocity"][
            sample_idx, time_idx - n_steps * self.dt : time_idx + self.dt
        ]
        particles = file["particles"][
            sample_idx, time_idx - n_steps * self.dt : time_idx + self.dt
        ]
        comb = np.concatenate([velocity, particles], -1)
        return comb.transpose((0, 3, 1, 2))

    def _get_specific_bcs(self, f):
        return [0, 0]


class PDEArenaINS(BaseHDF5DirectoryDataset):
    """
    Order Vx, Vy, density, pressure
    """

    @staticmethod
    def _specifics():
        time_index = 1
        sample_index = 0
        field_names = ["Vx", "Vy", "u"]
        type = "pa_ins"
        split_level = "sample"
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f):
        samples = f["Vx"].shape[0]
        steps = f["Vx"].shape[1]
        return samples, steps

    def more_specific_title(self, type, path, include_string):
        """
        Override this to add more info to the dataset name
        """
        split_path = self.include_string.split("/")[-1].split("_")
        buoy = split_path[-3]
        nu = split_path[-2]
        return f"{type}_buoy{buoy}_nu{nu}"

    def _reconstruct_sample(self, file, sample_idx, time_idx, n_steps):
        vx = file["Vx"][sample_idx, time_idx - n_steps * self.dt : time_idx + self.dt]
        vy = file["Vy"][sample_idx, time_idx - n_steps * self.dt : time_idx + self.dt]
        density = file["u"][
            sample_idx, time_idx - n_steps * self.dt : time_idx + self.dt
        ]
        comb = np.stack([vx, vy, density], 1)
        return comb

    def _get_specific_bcs(self, f):
        return [0, 0]


class CompNSDataset(BaseHDF5DirectoryDataset):
    """
    Order Vx, Vy, density, pressure
    """

    @staticmethod
    def _specifics():
        time_index = 1
        sample_index = 0
        field_names = ["Vx", "Vy", "density", "pressure"]
        type = "compNS"
        split_level = "sample"
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f):
        samples = f["Vx"].shape[0]
        steps = f["Vx"].shape[1]
        return samples, steps

    def more_specific_title(self, type, path, include_string):
        """
        Override this to add more info to the dataset name
        """
        cns_path = self.include_string.split("/")[-1].split("_")
        ic = cns_path[2]
        m = cns_path[3]
        res = cns_path[-2]

        return f"{type}_{ic}_{m}_res{res}"

    def _reconstruct_sample(self, file, sample_idx, time_idx, n_steps):
        vx = file["Vx"][sample_idx, time_idx - n_steps * self.dt : time_idx + self.dt]
        vy = file["Vy"][sample_idx, time_idx - n_steps * self.dt : time_idx + self.dt]
        density = file["density"][
            sample_idx, time_idx - n_steps * self.dt : time_idx + self.dt
        ]
        p = file["pressure"][
            sample_idx, time_idx - n_steps * self.dt : time_idx + self.dt
        ]

        comb = np.stack([vx, vy, density, p], 1)
        return comb

    def _get_specific_bcs(self, f):
        return [1, 1]


class BurgersDataset(BaseHDF5DirectoryDataset):
    """
    Order Vx, Vy, density, pressure
    """

    @staticmethod
    def _specifics():
        time_index = 1
        sample_index = 0
        field_names = ["Vx"]
        type = "burgers"
        split_level = "sample"
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f):
        samples = f["tensor"].shape[0]
        steps = f["tensor"].shape[1]
        return samples, steps

    def _reconstruct_sample(self, file, sample_idx, time_idx, n_steps):
        vx = file["tensor"][
            sample_idx, time_idx - n_steps * self.dt : time_idx + self.dt
        ]
        vx = vx[:, None, :, None]
        return vx

    def _get_specific_bcs(self, f):
        return [1, 1]


class DiffSorb1DDataset(BaseHDF5DirectoryDataset):
    @staticmethod
    def _specifics():
        time_index = 0
        sample_index = None
        field_names = ["u"]
        type = "diffsorb"
        split_level = "sample"
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f):
        samples = list(f.keys())
        steps = f[samples[0]]["data"].shape[0]
        return len(samples), steps

    def _get_specific_bcs(self, f):
        return [0, 0]

    def _reconstruct_sample(self, file, sample_idx, time_idx, n_steps):
        samples = list(file.keys())
        return file[samples[sample_idx]]["data"][
            time_idx - n_steps * self.dt : time_idx + self.dt
        ].transpose(0, 2, 1)[:, :, :, None]


class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        assert mask_ratio < 1 and mask_ratio >= 0
        self.mask_ratio = mask_ratio
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        if self.mask_ratio > 0:
            mask_per_frame = np.hstack(
                [
                    np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
                    np.ones(self.num_masks_per_frame),
                ]
            )
        elif self.mask_ratio == 0:
            mask_per_frame = np.hstack(
                [
                    np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
                ]
            )
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames, 1)).flatten()
        return mask


class MaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        assert mask_ratio < 1 and mask_ratio >= 0
        self.height, self.width = input_size
        self.mask_ratio = mask_ratio
        self.frames = 1
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        if self.mask_ratio > 0:
            mask_per_frame = np.hstack(
                [
                    np.zeros(self.num_masks_per_frame),
                    np.ones(self.num_patches_per_frame - self.num_masks_per_frame),
                ]
            )
        else:
            mask_per_frame = np.hstack(
                [
                    np.ones(self.num_patches_per_frame - self.num_masks_per_frame),
                ]
            )
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames, 1)).flatten()
        return mask.astype(np.float16)


class MultisetSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset."""

    def __init__(
        self,
        dataset: Dataset,
        base_sampler: Sampler,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        max_samples=10,
        rank=0,
        distributed=True,
    ) -> None:
        self.batch_size = batch_size
        self.sub_dsets = dataset.sub_dsets
        if distributed:
            self.sub_samplers = [
                base_sampler(dataset, drop_last=drop_last) for dataset in self.sub_dsets
            ]
        else:
            self.sub_samplers = [base_sampler(dataset) for dataset in self.sub_dsets]
        self.dataset = dataset
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.max_samples = max_samples
        self.rank = rank

    def __iter__(self) -> Iterator[T_co]:
        samplers = [iter(sampler) for sampler in self.sub_samplers]
        sampler_choices = list(range(len(samplers)))
        count = 0
        while len(sampler_choices) > 0:
            count += 1
            index_sampled = paddle.randint(0, len(sampler_choices), shape=(1,)).item()
            dset_sampled = sampler_choices[index_sampled]
            offset = max(0, self.dataset.offsets[dset_sampled])
            try:
                queue = []
                for i in range(self.batch_size):
                    queue.append(next(samplers[dset_sampled]) + offset)
                if len(queue) == self.batch_size:
                    for d in queue:
                        yield d
            except Exception as err:
                print("ERRRR", err)
                sampler_choices.pop(index_sampled)
                print(
                    f"Note: dset {dset_sampled} fully used. Dsets remaining: {len(sampler_choices)}"
                )
                continue
            if count >= self.max_samples:
                break

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        for sampler in self.sub_samplers:
            sampler.set_epoch(epoch)
        self.epoch = epoch


def PoisHelmDatasetLoader(params, location, distributed, train=True):
    transform = paddle.to_tensor
    # dataset[0] = [[4, 64, 64], [1, 64, 64]]
    dataset = PoisHelmDataset(params, location, transform, train)
    sampler = DistributedBatchSampler(dataset, shuffle=train) if distributed else None
    dataloader = DataLoader(dataset, batch_size=params.batch_size, num_workers=params.num_data_workers)
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
            self.subsample = 1
        self.scales = None
        self._get_files_stats()
        if isinstance(self.masking, float):
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
                self.train_rand_idx = range(self.data.shape[0])
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
            if hasattr(self.params, "pt_split"):
                self.pt_split = self.params.pt_split
            else:
                self.pt_split = [0.9, 0.1]
            logging.info(
                "Split training set into {} for pretrain, {} for train. ".format(
                    self.pt_split[0], self.pt_split[1]
                )
            )
            if hasattr(self.params, "pt"):
                self.pt = self.params.pt
            else:
                self.pt = "train"
            if int(sum(self.pt_split)) == 1:
                self.n_samples *= self.pt_split[-1 if self.pt == "train" else 0]
            else:
                assert int(sum(self.pt_split)) <= self.n_samples
                self.n_samples = self.pt_split[-1 if self.pt == "train" else 0]
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
                demo_indices = np.random.choice(
                    range(self.n_demos), self.params.n_demos, replace=False
                )
                X = np.take(
                    self.data[local_idx, 0 : self.in_channels],
                    np.insert(demo_indices, 0, 0),
                    1,
                )
            else:
                X = self.data[
                    local_idx, 0 : self.in_channels, : self.params.n_demos + 1
                ]
        if self.tensor is not None:
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
            X = X / f_scaling
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
                y = self.data[local_idx, self.in_channels :, : self.params.n_demos + 1]
        y = self.transform(y)

        if isinstance(self.masking, float):
            mask = self.mask_generator().reshape(1, self.img_shape_x, self.img_shape_y)
            return X, y, mask
        else:
            return X, y

    def __getitem__(self, idx):
        local_idx = int(idx * self.subsample)
        if self.params.n_demos > 0 and self.n_demos is None:
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


def MixedDatasetLoader(
    params, paths, distributed, split="train", rank=0, train_offset=0
):
    train_val_test = params.train_val_test
    if split == "pretrain":
        train_val_test = [
            params.train_val_test[0] * params.pretrain_train[0],
            train_val_test[1],
            train_val_test[2],
        ]
        split = "train"
    elif split == "train":
        train_val_test = [
            -params.train_val_test[0]
            * params.pretrain_train[1]
            * params.train_subsample,
            train_val_test[1],
            train_val_test[2],
        ]
    dataset = MixedDataset(
        paths,
        n_steps=params.n_steps,
        train_val_test=train_val_test,
        split=split,
        tie_fields=params.tie_fields,
        use_all_fields=params.use_all_fields,
        enforce_max_steps=params.enforce_max_steps,
        train_offset=train_offset,
        masking=params.masking if hasattr(params, "masking") else None,
        blur=params.blur if hasattr(params, "blur") else None,
        rollout=getattr(params, "rollout", 1),
    )
    if distributed:
        base_sampler = DistributedBatchSampler
    else:
        base_sampler = RandomSampler
    sampler = MultisetSampler(
        dataset,
        base_sampler,
        params.batch_size,
        distributed=distributed,
        max_samples=params.epoch_size,
        rank=rank,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(params.batch_size),
        num_workers=params.num_data_workers,
        shuffle=False,
        drop_last=True,
    )
    return dataloader, dataset, sampler


DSET_NAME_TO_OBJECT = {
    "incompNS": IncompNSDataset,
    "diffre2d": DiffRe2DDataset,
}


class MixedDataset(Dataset):
    def __init__(
        self,
        path_list=[],
        n_steps=1,
        dt=1,
        train_val_test=(0.8, 0.1, 0.1),
        split="train",
        tie_fields=True,
        use_all_fields=True,
        extended_names=False,
        enforce_max_steps=False,
        train_offset=0,
        masking=None,
        blur=None,
        rollout=1,
    ):
        super().__init__()
        self.train_offset = train_offset
        self.path_list, self.type_list, self.include_string = zip(*path_list)
        self.tie_fields = tie_fields
        self.extended_names = extended_names
        self.split = split
        self.sub_dsets = []
        self.offsets = [0]
        self.train_val_test = train_val_test
        self.use_all_fields = use_all_fields
        self.rollout = rollout

        for dset, path, include_string in zip(
            self.type_list, self.path_list, self.include_string
        ):
            subdset = DSET_NAME_TO_OBJECT[dset](
                path,
                include_string,
                n_steps=n_steps,
                dt=dt,
                train_val_test=train_val_test,
                split=split,
                rollout=self.rollout,
            )
            try:
                len(subdset)
            except ValueError:
                raise ValueError(
                    f"Dataset {path} is empty. Check that n_steps < trajectory_length in file."
                )
            self.sub_dsets.append(subdset)
            self.offsets.append(self.offsets[-1] + len(self.sub_dsets[-1]))
        self.offsets[0] = -1

        self.subset_dict = self._build_subset_dict()

        self.masking = masking
        if (
            self.masking
            and type(self.masking) in [tuple, list]
            and len(self.masking) == 2
        ):
            self.mask_generator = TubeMaskingGenerator(self.masking[0], self.masking[1])
        self.blur = blur

    def get_state_names(self):
        name_list = []
        if self.use_all_fields:
            for name, dset in DSET_NAME_TO_OBJECT.items():
                field_names = dset._specifics()[2]
                name_list += field_names
            return name_list
        else:
            visited = set()
            for dset in self.sub_dsets:
                name = dset.get_name()
                if name not in visited:
                    visited.add(name)
                    name_list.append(dset.field_names)
        return [f for fl in name_list for f in fl]

    def _build_subset_dict(self):
        if self.tie_fields:
            subset_dict = {
                "swe": [3],
                "incompNS": [0, 1, 2],
                "compNS": [0, 1, 2, 3],
                "diffre2d": [4, 5],
            }
        elif self.use_all_fields:
            cur_max = 0
            subset_dict = {}
            for name, dset in DSET_NAME_TO_OBJECT.items():
                field_names = dset._specifics()[2]
                subset_dict[name] = list(range(cur_max, cur_max + len(field_names)))
                cur_max += len(field_names)
        else:
            subset_dict = {}
            cur_max = self.train_offset
            for dset in self.sub_dsets:
                name = dset.get_name(self.extended_names)
                if name not in subset_dict:
                    subset_dict[name] = list(
                        range(cur_max, cur_max + len(dset.field_names))
                    )
                    cur_max += len(dset.field_names)
        return subset_dict

    def __getitem__(self, index):
        file_idx = np.searchsorted(self.offsets, index, side="right") - 1
        local_idx = index - max(self.offsets[file_idx], 0)

        x, y = self.sub_dsets[file_idx][local_idx]
        try:
            x, y = self.sub_dsets[file_idx][local_idx]
        except:  # noqa
            print(
                "FAILED AT ", file_idx, local_idx, index, int(os.environ.get("RANK", 0))
            )

        if (
            self.masking
            and type(self.masking) in [tuple, list]
            and len(self.masking) == 2
        ):
            mask = self.mask_generator()
            return x, y, mask
        else:
            return x, y

    def __len__(self):
        return sum([len(dset) for dset in self.sub_dsets])
