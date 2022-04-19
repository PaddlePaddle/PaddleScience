# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


def discretize(pde,
               time_nsteps=None,
               space_method="sampling",
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

    # PDE
    if pde is not None:
        pde_disc = pde.discretize(time_nsteps)
        pde_disc.geometry = pde.geometry.discretize(space_method,
                                                    space_npoints)

        # print(pde_disc.geometry.interior)
        # print(pde_disc.geometry.boundary)

        # TODO: discretize apis

        # pde_disc.discretize_bc(pde_disc.geometry_disc)

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

        return pde_disc
    else:
        return None
