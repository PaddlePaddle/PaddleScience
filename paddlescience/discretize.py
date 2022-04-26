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
    pde_disc.geometry = pde_disc.geometry.discretize(space_method,
                                                     space_npoints)
    nproc = paddle.distributed.get_world_size()
    pde_disc.geometry.padding(nproc)

    # discritize rhs in equations
    pde_disc.rhs_disc = list()
    for rhs in pde_disc.rhs:
        points_i = pde_disc.geometry.interior

        data = list()
        for n in range(len(points_i[0])):
            data.append(points_i[:, n])

        if type(rhs) == types.LambdaType:
            pde_disc.rhs_disc.append(rhs(*data))
        else:
            pde_disc.rhs_disc.append(rhs)

    # discretize weight in equations
    weight = pde_disc.weight
    if (weight is None) or np.isscalar(weight):
        pde_disc.weight_disc = [weight for _ in range(len(pde.equations))]
    else:
        pass
        # TODO: points dependent value

    # discritize weight and rhs in boundary condition
    for name_b, bc in pde_disc.bc.items():
        points_b = pde_disc.geometry.boundary[name_b]

        data = list()
        for n in range(len(points_b[0])):
            data.append(points_b[:, n])

        # boundary weight
        for b in bc:
            # compute weight lambda with cordinates
            if type(b.weight) == types.LambdaType:
                b.weight_disc = b.weight(*data)
            else:
                b.weight_disc = b.weight

        # boundary rhs
        for b in bc:
            if type(b.rhs) == types.LambdaType:
                b.rhs_disc = b.rhs(*data)
            else:
                b.rhs_disc = b.rhs

    # discretize rhs in initial condition
    for ic in pde_disc.ic:
        points_i = pde_disc.geometry.interior

        data = list()
        for n in range(len(points_i[0])):
            data.append(points_i[:, n])

        rhs = ic.rhs
        if type(rhs) == types.LambdaType:
            ic.rhs_disc = rhs(*data)
        else:
            ic.rhs_disc = rhs

    return pde_disc
