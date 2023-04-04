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

from typing import Any
from typing import Callable
from typing import Dict
from typing import Union

import types

import numpy as np
import sympy
from sympy.parsing import sympy_parser as sp_parser
from typing_extensions import Literal

from ppsci import geometry
from ppsci import loss
from ppsci.data import dataset
from ppsci.validate import base


class GeometryValidator(base.Validator):
    def __init__(
        self,
        label_expr: Dict[str, Callable],
        label_dict: Dict[str, Union[float, Callable]],
        geom: geometry.Geometry,
        dataloader_cfg: Dict[str, Any],
        loss: loss.LossBase,
        random: Literal["pseudo", "LHS"] = "pseudo",
        criteria: Callable = None,
        evenly: bool = False,
        metric=None,
        with_initial: bool = False,
        name: str = None,
    ):
        self.label_expr = label_expr
        for label_name, expr in self.label_expr.items():
            if isinstance(expr, str):
                self.label_expr[label_name] = sp_parser.parse_expr(expr)

        self.label_dict = label_dict
        self.input_keys = geom.dim_keys
        self.output_keys = list(label_dict.keys())

        nx = dataloader_cfg["total_size"]
        self.num_timestamp = 1
        # TODO(sensen): simplify code below
        if isinstance(geom, geometry.TimeXGeometry):
            if geom.timedomain.num_timestamp is not None:
                if with_initial:
                    # include t0
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
                    "TimeXGeometry with random timestamp not implemented yet."
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
            elif isinstance(value, types.FunctionType):
                func = value
                label[key] = func(input)
                if isinstance(label[key], (int, float)):
                    label[key] = np.full(
                        (next(iter(input.values())).shape[0], 1),
                        float(label[key], "float32"),
                    )
            else:
                raise NotImplementedError(f"type of {type(value)} is invalid yet.")

        weight = {key: np.ones_like(next(iter(label.values()))) for key in label}

        _dataset = getattr(dataset, dataloader_cfg["dataset"])(input, label, weight)
        super().__init__(_dataset, dataloader_cfg, loss, metric, name)
