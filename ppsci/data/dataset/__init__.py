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
from typing import TYPE_CHECKING

from ppsci.data.dataset.airfoil_dataset import MeshAirfoilDataset
from ppsci.data.dataset.array_dataset import ChipHeatDataset
from ppsci.data.dataset.array_dataset import ContinuousNamedArrayDataset
from ppsci.data.dataset.array_dataset import IterableNamedArrayDataset
from ppsci.data.dataset.array_dataset import NamedArrayDataset
from ppsci.data.dataset.atmospheric_dataset import GridMeshAtmosphericDataset
from ppsci.data.dataset.cgcnn_dataset import CIFData as CGCNNDataset
from ppsci.data.dataset.csv_dataset import CSVDataset
from ppsci.data.dataset.csv_dataset import IterableCSVDataset
from ppsci.data.dataset.cylinder_dataset import MeshCylinderDataset
from ppsci.data.dataset.darcyflow_dataset import DarcyFlowDataset
from ppsci.data.dataset.dgmr_dataset import DGMRDataset
from ppsci.data.dataset.drivaernet_dataset import DrivAerNetDataset
from ppsci.data.dataset.drivaernetplusplus_dataset import DrivAerNetPlusPlusDataset
from ppsci.data.dataset.enso_dataset import ENSODataset
from ppsci.data.dataset.era5_dataset import ERA5Dataset
from ppsci.data.dataset.era5_dataset import ERA5SampledDataset
from ppsci.data.dataset.ext_moe_enso_dataset import ExtMoEENSODataset
from ppsci.data.dataset.fwi_dataset import FWIDataset
from ppsci.data.dataset.ifm_moe_dataset import IFMMoeDataset
from ppsci.data.dataset.mat_dataset import IterableMatDataset
from ppsci.data.dataset.mat_dataset import MatDataset
from ppsci.data.dataset.moflow_dataset import MOlFLOWDataset
from ppsci.data.dataset.mrms_dataset import MRMSDataset
from ppsci.data.dataset.mrms_dataset import MRMSSampledDataset
from ppsci.data.dataset.npz_dataset import IterableNPZDataset
from ppsci.data.dataset.npz_dataset import NPZDataset
from ppsci.data.dataset.pems_dataset import PEMSDataset
from ppsci.data.dataset.radar_dataset import RadarDataset
from ppsci.data.dataset.sevir_dataset import SEVIRDataset
from ppsci.data.dataset.spherical_swe_dataset import SphericalSWEDataset
from ppsci.data.dataset.trphysx_dataset import CylinderDataset
from ppsci.data.dataset.trphysx_dataset import LorenzDataset
from ppsci.data.dataset.trphysx_dataset import RosslerDataset
from ppsci.data.dataset.vtu_dataset import VtuDataset
from ppsci.data.process import transform
from ppsci.utils import logger

if TYPE_CHECKING:
    from paddle import io

__all__ = [
    "IterableNamedArrayDataset",
    "NamedArrayDataset",
    "ContinuousNamedArrayDataset",
    "ChipHeatDataset",
    "CSVDataset",
    "IterableCSVDataset",
    "ERA5Dataset",
    "ERA5SampledDataset",
    "GridMeshAtmosphericDataset",
    "IterableMatDataset",
    "MatDataset",
    "MRMSDataset",
    "MRMSSampledDataset",
    "IterableNPZDataset",
    "NPZDataset",
    "PEMSDataset",
    "CylinderDataset",
    "LorenzDataset",
    "RadarDataset",
    "RosslerDataset",
    "VtuDataset",
    "DGMRDataset",
    "MeshAirfoilDataset",
    "MeshCylinderDataset",
    "DarcyFlowDataset",
    "SphericalSWEDataset",
    "ENSODataset",
    "ExtMoEENSODataset",
    "SEVIRDataset",
    "MOlFLOWDataset",
    "build_dataset",
    "CGCNNDataset",
    "FWIDataset",
    "DrivAerNetDataset",
    "DrivAerNetPlusPlusDataset",
    "IFMMoeDataset",
]


def build_dataset(cfg) -> "io.Dataset":
    """Build dataset

    Args:
        cfg (List[DictConfig]): Dataset config list.

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
