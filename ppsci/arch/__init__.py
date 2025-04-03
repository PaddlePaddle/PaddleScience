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

from __future__ import annotations

import copy

from ppsci.arch.afno import AFNONet  # isort:skip
from ppsci.arch.afno import PrecipNet  # isort:skip
from ppsci.arch.amgnet import AMGNet  # isort:skip
from ppsci.arch.base import Arch  # isort:skip
from ppsci.arch.cfdgcn import CFDGCN  # isort:skip
from ppsci.arch.chip_deeponets import ChipDeepONets  # isort:skip
from ppsci.arch.crystalgraphconvnet import CrystalGraphConvNet  # isort:skip
from ppsci.arch.cuboid_transformer import CuboidTransformer  # isort:skip
from ppsci.arch.cvit import CVit  # isort:skip
from ppsci.arch.cvit import CVit1D  # isort:skip
from ppsci.arch.deeponet import DeepONet  # isort:skip
from ppsci.arch.dgmr import DGMR  # isort:skip
from ppsci.arch.embedding_koopman import CylinderEmbedding  # isort:skip
from ppsci.arch.embedding_koopman import LorenzEmbedding  # isort:skip
from ppsci.arch.embedding_koopman import RosslerEmbedding  # isort:skip
from ppsci.arch.epnn import Epnn  # isort:skip
from ppsci.arch.extformer_moe_cuboid import ExtFormerMoECuboid  # isort:skip
from ppsci.arch.gan import Discriminator  # isort:skip
from ppsci.arch.gan import Generator  # isort:skip
from ppsci.arch.geofno import FNO1d  # isort:skip
from ppsci.arch.graphcast import GraphCastNet  # isort:skip
from ppsci.arch.he_deeponets import HEDeepONets  # isort:skip
from ppsci.arch.lno import LNO  # isort:skip
from ppsci.arch.mlp import MLP  # isort:skip
from ppsci.arch.mlp import ModifiedMLP  # isort:skip
from ppsci.arch.mlp import PirateNet  # isort:skip
from ppsci.arch.model_list import ModelList  # isort:skip
from ppsci.arch.nowcastnet import NowcastNet  # isort:skip
from ppsci.arch.phycrnet import PhyCRNet  # isort:skip
from ppsci.arch.phylstm import DeepPhyLSTM  # isort:skip
from ppsci.arch.physx_transformer import PhysformerGPT2  # isort:skip
from ppsci.arch.sfnonet import SFNONet  # isort:skip
from ppsci.arch.spinn import SPINN  # isort:skip
from ppsci.arch.tfnonet import TFNO1dNet, TFNO2dNet, TFNO3dNet  # isort:skip
from ppsci.arch.transformer import Transformer  # isort:skip
from ppsci.arch.unetex import UNetEx  # isort:skip
from ppsci.arch.unonet import UNONet  # isort:skip
from ppsci.arch.uscnn import USCNN  # isort:skip
from ppsci.arch.vae import AutoEncoder  # isort:skip
from ppsci.arch.velocitygan import VelocityDiscriminator  # isort:skip
from ppsci.arch.velocitygan import VelocityGenerator  # isort:skip
from ppsci.arch.moflow_net import MoFlowNet, MoFlowProp  # isort:skip
from ppsci.utils import logger  # isort:skip
from ppsci.arch.regdgcnn import RegDGCNN  # isort:skip
from ppsci.arch.regpointnet import RegPointNet  # isort:skip
from ppsci.arch.ifm_mlp import IFMMLP  # isort:skip

__all__ = [
    "MoFlowNet",
    "MoFlowProp",
    "AFNONet",
    "AMGNet",
    "Arch",
    "AutoEncoder",
    "build_model",
    "CFDGCN",
    "ChipDeepONets",
    "CrystalGraphConvNet",
    "CuboidTransformer",
    "CVit",
    "CVit1D",
    "CylinderEmbedding",
    "DeepONet",
    "DeepPhyLSTM",
    "DGMR",
    "Discriminator",
    "Epnn",
    "ExtFormerMoECuboid",
    "FNO1d",
    "Generator",
    "GraphCastNet",
    "HEDeepONets",
    "LorenzEmbedding",
    "LNO",
    "MLP",
    "ModelList",
    "ModifiedMLP",
    "NowcastNet",
    "PhyCRNet",
    "PhysformerGPT2",
    "PirateNet",
    "PrecipNet",
    "RosslerEmbedding",
    "SFNONet",
    "SPINN",
    "TFNO1dNet",
    "TFNO2dNet",
    "TFNO3dNet",
    "Transformer",
    "UNetEx",
    "UNONet",
    "USCNN",
    "VelocityDiscriminator",
    "VelocityGenerator",
    "RegDGCNN",
    "RegPointNet",
    "IFMMLP",
]


def build_model(cfg):
    """Build model

    Args:
        cfg (DictConfig): Arch config.

    Returns:
        nn.Layer: Model.
    """
    cfg = copy.deepcopy(cfg)
    arch_cls = cfg.pop("name")
    arch = eval(arch_cls)(**cfg)

    logger.debug(str(arch))

    return arch
