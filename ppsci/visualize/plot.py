"""Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import paddle
from matplotlib import pyplot as plt

from ppsci.utils import logger

cnames = [
    "bisque",
    "black",
    "blanchedalmond",
    "blue",
    "blueviolet",
    "brown",
    "burlywood",
    "cadetblue",
    "chartreuse",
    "chocolate",
    "coral",
    "cornflowerblue",
    "cornsilk",
    "crimson",
    "cyan",
    "darkblue",
    "darkcyan",
    "darkgoldenrod",
    "darkgray",
    "darkgreen",
    "darkkhaki",
    "darkmagenta",
    "darkolivegreen",
    "darkorange",
    "darkorchid",
    "darkred",
    "darksalmon",
    "darkseagreen",
    "darkslateblue",
    "darkslategray",
    "darkturquoise",
    "darkviolet",
    "deeppink",
    "deepskyblue",
    "dimgray",
    "dodgerblue",
    "firebrick",
    "floralwhite",
    "forestgreen",
    "fuchsia",
    "gainsboro",
    "ghostwhite",
    "gold",
    "goldenrod",
    "gray",
    "green",
    "greenyellow",
    "honeydew",
    "hotpink",
    "indianred",
    "indigo",
    "ivory",
    "khaki",
    "lavender",
    "lavenderblush",
    "lawngreen",
    "lemonchiffon",
    "lightblue",
    "lightcoral",
    "lightcyan",
    "lightgoldenrodyellow",
    "lightgreen",
    "lightgray",
    "lightpink",
    "lightsalmon",
    "lightseagreen",
    "lightskyblue",
    "lightslategray",
    "lightsteelblue",
    "lightyellow",
    "lime",
    "limegreen",
    "linen",
    "magenta",
    "maroon",
    "mediumaquamarine",
    "mediumblue",
    "mediumorchid",
    "mediumpurple",
    "mediumseagreen",
    "mediumslateblue",
    "mediumspringgreen",
    "mediumturquoise",
    "mediumvioletred",
    "midnightblue",
    "mintcream",
    "mistyrose",
    "moccasin",
    "navajowhite",
    "navy",
    "oldlace",
    "olive",
    "olivedrab",
    "orange",
    "orangered",
    "orchid",
    "palegoldenrod",
    "palegreen",
    "paleturquoise",
    "palevioletred",
    "papayawhip",
    "peachpuff",
    "peru",
    "pink",
    "plum",
    "powderblue",
    "purple",
    "red",
    "rosybrown",
    "royalblue",
    "saddlebrown",
    "salmon",
    "sandybrown",
    "seagreen",
    "seashell",
    "sienna",
    "silver",
    "skyblue",
    "slateblue",
    "slategray",
    "snow",
    "springgreen",
    "steelblue",
    "tan",
    "teal",
    "thistle",
    "tomato",
    "turquoise",
    "violet",
    "wheat",
    "white",
    "whitesmoke",
    "yellow",
    "yellowgreen",
]


def save_prediction_plot(filename, data_dict, coord_key, value_keys):
    if not isinstance(coord_key, str):
        raise ValueError(f"Type of coord({len(coord_key)}) should be str.")

    if value_keys is None:
        raise ValueError(f"value_keys can not be None.")
    coord = data_dict[coord_key]
    sorted_index = np.argsort(coord, axis=0).squeeze()
    coord = coord[sorted_index]
    value = [data_dict[k] for k in value_keys] if value_keys else None

    if isinstance(coord[0], paddle.Tensor):
        coord = coord.numpy()

    if isinstance(value[0], paddle.Tensor):
        value = [x.numpy() for x in value]

    plt_nrow = min(1, int(np.sqrt(len(value_keys)) + 0.5))
    plt_ncol = len(value_keys) // plt_nrow
    if plt_ncol < plt_nrow:
        plt_ncol, plt_nrow = plt_nrow, plt_ncol
        # 子画布排列时，列数大于等于行数

    fig, a = plt.subplots(plt_nrow, plt_ncol, squeeze=False)
    for plt_idx, _value in enumerate(value):
        plt_i = plt_idx // plt_ncol
        plt_j = plt_idx % plt_ncol
        a[plt_i][plt_j].plot(
            coord,
            _value[sorted_index],
            color=cnames[plt_idx],
            label=value_keys[plt_idx],
        )
        a[plt_i][plt_j].set_title(value_keys[plt_idx])
        a[plt_i][plt_j].grid()
        a[plt_i][plt_j].legend()

    fig.savefig(filename, dpi=300)

    logger.info(f"Prediction plot result is saved to {filename}")
