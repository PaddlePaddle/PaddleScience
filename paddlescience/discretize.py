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


def discretize(pde, geo, time_nsteps=None, space_nsteps=None):
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

    # Geometry
    geo_disc = geo.discretize(time_nsteps, space_nsteps)

    return pde, geo_disc


def sampling_discretize(pde,
                        geo,
                        time_nsteps=None,
                        space_point_size=None,
                        space_nsteps=None):
    # Geometry
    geo_disc = geo.sampling_discretize(time_nsteps, space_point_size,
                                       space_nsteps)

    return pde, geo_disc
