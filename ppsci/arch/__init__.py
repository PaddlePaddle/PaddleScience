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

from ppsci.arch.base import Arch  # isort:skip
from ppsci.arch.amgnet import AMGNet  # isort:skip
from ppsci.arch.mlp import MLP  # isort:skip
from ppsci.arch.mlp import FullyConnectedLayer  # isort:skip
from ppsci.arch.deeponet import DeepONet  # isort:skip
from ppsci.arch.deeponet import DeepOperatorLayer  # isort:skip
from ppsci.arch.embedding_koopman import LorenzEmbedding  # isort:skip
from ppsci.arch.embedding_koopman import LorenzEmbeddingLayer  # isort:skip
from ppsci.arch.embedding_koopman import RosslerEmbedding  # isort:skip
from ppsci.arch.embedding_koopman import RosslerEmbeddingLayer  # isort:skip
from ppsci.arch.embedding_koopman import CylinderEmbedding  # isort:skip
from ppsci.arch.embedding_koopman import CylinderEmbeddingLayer  # isort:skip
from ppsci.arch.gan import Generator  # isort:skip
from ppsci.arch.gan import GeneratorLayer  # isort:skip
from ppsci.arch.gan import Discriminator  # isort:skip
from ppsci.arch.gan import DiscriminatorLayer  # isort:skip
from ppsci.arch.phylstm import DeepPhyLSTM  # isort:skip
from ppsci.arch.physx_transformer import PhysformerGPT2  # isort:skip
from ppsci.arch.physx_transformer import PhysformerGPT2Layer  # isort:skip
from ppsci.arch.model_list import ModelList  # isort:skip
from ppsci.arch.afno import AFNONet  # isort:skip
from ppsci.arch.afno import PrecipNet  # isort:skip
from ppsci.arch.unetex import UNetEx  # isort:skip
from ppsci.arch.unetex import UNetExLayer  # isort:skip
from ppsci.utils import logger  # isort:skip


__all__ = [
    "Arch",
    "AMGNet",
    "MLP",
    "FullyConnectedLayer",
    "DeepONet",
    "DeepOperatorLayer",
    "LorenzEmbedding",
    "LorenzEmbeddingLayer",
    "RosslerEmbedding",
    "RosslerEmbeddingLayer",
    "CylinderEmbedding",
    "CylinderEmbeddingLayer",
    "Generator",
    "GeneratorLayer",
    "Discriminator",
    "DiscriminatorLayer",
    "DeepPhyLSTM",
    "PhysformerGPT2",
    "PhysformerGPT2Layer",
    "ModelList",
    "AFNONet",
    "PrecipNet",
    "UNetEx",
    "UNetExLayer",
    "build_model",
]


def build_model(cfg):
    """Build model

    Args:
        cfg (AttrDict): Arch config.

    Returns:
        nn.Layer: Model.
    """
    cfg = copy.deepcopy(cfg)
    arch_cls = cfg.pop("name")
    arch = eval(arch_cls)(**cfg)

    logger.debug(str(arch))

    return arch
