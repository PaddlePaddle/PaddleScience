"""
Remember to parameterize the file paths eventually
"""
import glob

import h5py
import numpy as np
from paddle.io import Dataset

broken_paths = [""]


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
        self.extra_specific = extra_specific  # Whether to use parameters in name
        if subname is None:
            self.subname = path.split("/")[-1]
        else:
            self.subname = subname
        self.dt = dt  # TODO:
        self.rollout = rollout
        self.n_steps = n_steps
        self.include_string = include_string
        # self.time_index, self.sample_index = self._set_specifics()
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
        # Sets self.field_names, self.dataset_type
        raise NotImplementedError  # Per dset

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
        raise NotImplementedError  # Per dset

    def _get_specific_bcs(self, f):
        raise NotImplementedError  # Per dset

    def _reconstruct_sample(self, file, sample_idx, time_idx, n_steps):
        raise NotImplementedError  # Per dset - should be (x=(-history:local_idx+dt) so that get_item can split into x, y

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
            # Total hack to avoid complications from folder with two sizes.
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
                            # Compute which are in the given partition
                            partition = self.partition
                            sample_per_part = np.ceil(
                                np.absolute(np.array(self.train_val_test) * samples)
                            ).astype(int)
                            # Make sure rounding works
                            sample_per_part[2] = max(
                                samples - sample_per_part[0] - sample_per_part[1], 0
                            )
                            # I forget where the file steps formula came from, but offset by steps per sample
                            # * samples of previous partitions
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
        # print(self.file_steps, self.file_samples)
        self.files_paths = file_paths
        self.offsets[0] = -1  # Just to make sure it doesn't put us in file -1
        self.files = [None for _ in self.files_paths]
        self.len = self.offsets[-1]
        if self.split_level == "file":
            # Figure out our split offset - by sample
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
                    ]  # TODO: how many "files" to use
                    ideal_split_offsets = [
                        int(value) if value >= 1 else value
                        for value in ideal_split_offsets
                    ]
                else:
                    ideal_split_offsets = [
                        int(self.train_val_test[i] * total_samples) for i in range(3)
                    ]  # TODO: how many "files" to use
                # Doing this the naive way because I only need to do it once
                # Iterate through files until we get enough samples for set
                if ideal_split_offsets[0] > 0:
                    end_ind = 0
                elif ideal_split_offsets[0] == 0:
                    # use fraction to indicate how many "sample" to use
                    ideal_split_offsets[0] = abs(self.train_val_test[0] * total_samples)
                    assert ideal_split_offsets[0] < 1 and ideal_split_offsets[0] > 0
                    end_ind = total_samples - round(sum(ideal_split_offsets[1:])) - 1
                else:
                    # negative means reverse indexing in this split (train)
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
            # else:

    def _open_file(self, file_ind):
        _file = h5py.File(self.files_paths[file_ind], "r")
        self.files[file_ind] = _file

    def __getitem__(self, index):
        if self.split_level == "file":
            index = index + self.split_offset
        file_idx = int(
            np.searchsorted(self.offsets, index, side="right") - 1
        )  # which file we are on
        # print('sample from:', self.files_paths[file_idx])
        nsteps = (
            self.file_nsteps[file_idx] + self.rollout - 1
        )  # Number of steps per sample in given file
        local_idx = index - max(self.offsets[file_idx], 0)  # First offset is -1
        if self.split_level == "sample":
            sample_idx = (local_idx + self.split_offsets[file_idx]) // self.file_steps[
                file_idx
            ]
        else:
            sample_idx = local_idx // self.file_steps[file_idx]
        time_idx = local_idx % self.file_steps[file_idx]

        # open image file
        if self.files[file_idx] is None:
            self._open_file(file_idx)

        # if we are on the last image in a file shift backward. Double counting until I bother fixing this.
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
            # bcs = self._get_specific_bcs(self.files[file_idx])
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
        return [0, 0]  # Non-periodic

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
        return [0, 0]  # Non-periodic

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
        steps = f["velocity"].shape[1]  # Per dset
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
        return [0, 0]  # Non-periodic


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
        steps = f["Vx"].shape[1]  # Per dset
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
        return comb  # .transpose((0, 3, 1, 2))

    def _get_specific_bcs(self, f):
        return [0, 0]  # Not Periodic


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
        steps = f["Vx"].shape[1]  # Per dset
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
        return comb  # .transpose((0, 3, 1, 2))

    def _get_specific_bcs(self, f):
        return [1, 1]  # Periodic


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
        steps = f["tensor"].shape[1]  # Per dset
        return samples, steps

    def _reconstruct_sample(self, file, sample_idx, time_idx, n_steps):
        vx = file["tensor"][
            sample_idx, time_idx - n_steps * self.dt : time_idx + self.dt
        ]
        # print(vx.shape)
        vx = vx[:, None, :, None]
        return vx  # .transpose((0, 3, 1, 2))

    def _get_specific_bcs(self, f):
        return [1, 1]  # Periodic


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
        return [0, 0]  # Non-periodic

    def _reconstruct_sample(self, file, sample_idx, time_idx, n_steps):
        samples = list(file.keys())
        return file[samples[sample_idx]]["data"][
            time_idx - n_steps * self.dt : time_idx + self.dt
        ].transpose(0, 2, 1)[:, :, :, None]
