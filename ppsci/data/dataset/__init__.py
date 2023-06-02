# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

from ppsci.data.dataset.array_dataset import IterableNamedArrayDataset
from ppsci.data.dataset.array_dataset import NamedArrayDataset
from ppsci.data.dataset.csv_dataset import CSVDataset
from ppsci.data.dataset.csv_dataset import IterableCSVDataset
from ppsci.data.dataset.era5_dataset import ERA5Dataset
from ppsci.data.dataset.era5_dataset import ERA5SampledDataset
from ppsci.data.dataset.mat_dataset import IterableMatDataset
from ppsci.data.dataset.mat_dataset import MatDataset
from ppsci.data.dataset.trphysx_dataset import CylinderDataset
from ppsci.data.dataset.trphysx_dataset import LorenzDataset
from ppsci.data.dataset.trphysx_dataset import RosslerDataset
from ppsci.data.dataset.vtu_dataset import VtuDataset
from ppsci.data.process import transform
from ppsci.utils import logger

__all__ = [
    "IterableNamedArrayDataset",
    "NamedArrayDataset",
    "CSVDataset",
    "IterableCSVDataset",
    "ERA5Dataset",
    "ERA5SampledDataset",
    "IterableMatDataset",
    "MatDataset",
    "CylinderDataset",
    "LorenzDataset",
    "RosslerDataset",
    "VtuDataset",
    "build_dataset",
]


def build_dataset(cfg):
    """Build dataset

    Args:
        cfg (List[AttrDict]): dataset config list.

    Returns:
        Dict[str, io.Dataset]: dataset.
    """
    cfg = copy.deepcopy(cfg)

    dataset_cls = cfg.pop("name")
    if "transforms" in cfg:
        cfg["transforms"] = transform.build_transforms(cfg.pop("transforms"))

    dataset = eval(dataset_cls)(**cfg)

    logger.debug(str(dataset))

    return dataset
