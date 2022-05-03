# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import types
import copy
import numpy as np
import paddle


def discretize(pde,
               time_method=None,
               time_step=None,
               space_method=None,
               space_npoints=None):
    """
        Discretize PDE and Geometry

        Parameters:
            pde (PDE): The partial differential equations.
            geo (Geometry): The geometry to be discretized.


        Returns
        --------
        pde_disc: PDE
            Reserved parameter

        geo_disc: DiscreteGeometry
            Discrte Geometry
    """

    # discretize pde
    if pde.time_dependent:
        pde_disc = pde.discretize(time_method, time_step)
    else:
        pde_disc = pde

    # discretize and padding geometry
    user = pde.geometry.user

    pde_disc.geometry = pde_disc.geometry.discretize(space_method,
                                                     space_npoints)
    pde_disc.geometry.user = user

    nproc = paddle.distributed.get_world_size()
    pde_disc.geometry.padding(nproc)
