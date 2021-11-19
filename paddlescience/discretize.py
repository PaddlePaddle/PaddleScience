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


def discretize(pde, geo, time_steps=None, space_steps=None):

    # Geometry
    geo_disc = geo.discretize(time_steps, space_steps)
    # geo_disc.to_tensor()

    geo_disc.set_time_steps(time_steps)
    geo_disc.set_space_steps(space_steps)

    # PDE
    # pde = pde.discretize()
    # pde.to_tensor()

    return pde, geo_disc
