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
import sympy
from sympy.parsing import sympy_parser as sp_parser

from ppsci import geometry
from ppsci.data import dataset
from ppsci.validate import base


class GeometryValidator(base.Validator):
    def __init__(
        self,
        label_expr,
        label_dict,
        geom,
        dataloader_cfg,
        loss,
        random="pseudo",
        criteria=None,
        evenly=False,
        metric=None,
        with_initial=False,
        name=None,
    ):
        self.label_expr = label_expr
        for label_name, label_expr in self.label_expr.items():
            if isinstance(label_expr, str):
                self.label_expr[label_name] = sp_parser.parse_expr(label_expr)

        self.label_dict = label_dict
        self.input_keys = geom.dim_keys
        self.output_keys = list(label_dict.keys())

        nx = dataloader_cfg["total_size"]
        self.num_timestamp = 1
        # TODO(sensen): simplify code below
        if isinstance(geom, geometry.TimeXGeometry):
            if geom.timedomain.num_timestamp is not None:
                if with_initial:
                    self.num_timestamp = geom.timedomain.num_timestamp
                    assert (
                        nx % self.num_timestamp == 0
                    ), f"{nx} % {self.num_timestamp} != 0"
                    nx //= self.num_timestamp
                    input = geom.sample_interior(
                        nx * (geom.timedomain.num_timestamp - 1),
                        random,
                        criteria,
                        evenly,
                    )
                    initial = geom.sample_initial_interior(nx, random, criteria, evenly)
                    input = {
                        key: np.vstack((initial[key], input[key])) for key in input
                    }
                else:
                    # exclude t0
                    self.num_timestamp = geom.timedomain.num_timestamp - 1
                    assert (
                        nx % self.num_timestamp == 0
                    ), f"{nx} % {self.num_timestamp} != 0"
                    nx //= self.num_timestamp
                    input = geom.sample_interior(
                        nx * (geom.timedomain.num_timestamp - 1),
                        random,
                        criteria,
                        evenly,
                    )
            else:
                raise NotImplementedError(
                    f"TimeXGeometry with random timestamp not implemented yet."
                )
        else:
            input = geom.sample_interior(nx, random, criteria, evenly)

        label = {}
        for key, value in label_dict.items():
            if isinstance(value, (int, float)):
                label[key] = np.full_like(next(iter(input.values())), float(value))
            elif isinstance(value, sympy.Basic):
                func = sympy.lambdify(
                    sympy.symbols(geom.dim_keys),
                    value,
                    [{"amax": lambda xy, _: np.maximum(xy[0], xy[1])}, "numpy"],
                )
                label[key] = func(
                    **{k: v for k, v in input.items() if k in geom.dim_keys}
                )
            else:
                raise NotImplementedError(f"type of {type(value)} is invalid yet.")

        weight = {key: np.ones_like(next(iter(label.values()))) for key in label}

        _dataset = getattr(dataset, dataloader_cfg["dataset"])(input, label, weight)
        super().__init__(_dataset, dataloader_cfg, loss, metric, name)
