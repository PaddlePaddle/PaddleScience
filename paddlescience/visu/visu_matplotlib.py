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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def save_mpl(geo_disc, data, filename="output.npy"):
    with open(filename, "wb") as fp:
        np.save(fp, geo_disc.space_domain)
        np.save(fp, geo_disc.space_steps)
        np.save(fp, data)


def plot_mpl(filename):
    with open(filename, "rb") as fp:
        space_domain = np.load(fp)
        space_steps = np.load(fp)
        data = np.load(fp)
    xx = np.reshape(space_domain[:, 0], space_steps)
    yy = np.reshape(space_domain[:, 1], space_steps)
    zz = np.reshape(data, space_steps)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(
        xx, yy, zz, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # plt.show()
