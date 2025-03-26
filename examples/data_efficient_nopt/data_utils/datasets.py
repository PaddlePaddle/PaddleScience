"""
Remember to parameterize the file paths eventually
"""
import os

import numpy as np
from paddle.io import DataLoader
from paddle.io import Dataset
from paddle.io import DistributedBatchSampler
from paddle.io import RandomSampler

from .hdf5_datasets import DiffRe2DDataset
from .hdf5_datasets import IncompNSDataset
from .masking_generator import TubeMaskingGenerator
from .mixed_dset_sampler import MultisetSampler

broken_paths = []
# IF YOU ADD A NEW DSET MAKE SURE TO UPDATE THIS MAPPING SO MIXED DSET KNOWS HOW TO USE IT
DSET_NAME_TO_OBJECT = {
    "incompNS": IncompNSDataset,
    "diffre2d": DiffRe2DDataset,
}


def get_data_loader(params, paths, distributed, split="train", rank=0, train_offset=0):
    # paths, types, include_string = zip(*paths)
    train_val_test = params.train_val_test
    if split == "pretrain":
        train_val_test = [
            params.train_val_test[0] * params.pretrain_train[0],
            train_val_test[1],
            train_val_test[2],
        ]
        split = "train"  # then restore to train split
    elif split == "train":
        # negative means reverse indexing
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
    # dataset = IncompNSDataset(paths[0], n_steps=params.n_steps, train_val_test=params.train_val_test, split=split)
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
    )  # , seed=seed)
    # sampler = DistributedBatchSampler(dataset) if distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=int(params.batch_size),
        num_workers=params.num_data_workers,
        shuffle=False,  # (sampler is None),
        drop_last=True,
    )
    return dataloader, dataset, sampler


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
        # Global dicts used by Mixed DSET.
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
            # Check to make sure our dataset actually exists with these settings
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

        self.masking = masking  # None or ((#frames, height, width), mask_ratio)
        if (
            self.masking
            and type(self.masking) in [tuple, list]
            and len(self.masking) == 2
        ):  # and self.masking[1] > 0.:
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
                name = dset.get_name()  # Could use extended names here
                if name not in visited:
                    visited.add(name)
                    name_list.append(dset.field_names)
        return [f for fl in name_list for f in fl]  # Flatten the names

    def _build_subset_dict(self):
        # Maps fields to subsets of variables
        if self.tie_fields:  # Hardcoded, but seems less effective anyway
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
        file_idx = (
            np.searchsorted(self.offsets, index, side="right") - 1
        )  # which dataset are we are on
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
        ):  # and self.masking[1] > 0.:
            mask = self.mask_generator()
            # return x, file_idx, paddle.to_tensor(self.subset_dict[self.sub_dsets[file_idx].get_name()]), bcs, y, mask, x_blur
            return x, y, mask
        else:
            return x, y

    def __len__(self):
        return sum([len(dset) for dset in self.sub_dsets])
