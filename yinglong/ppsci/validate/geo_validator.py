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
from typing import Optional
from typing import Union

import numpy as np
import paddle
import sympy
from sympy.parsing import sympy_parser as sp_parser
from typing_extensions import Literal

from ppsci import geometry
from ppsci import loss
from ppsci import metric
from ppsci.data import dataset
from ppsci.validate import base


class GeometryValidator(base.Validator):
    """Validator for geometry.

    Args:
        output_expr (Dict[str, Callable]): Function in dict for computing output.
            e.g. {"u_mul_v": lambda out: out["u"] * out["v"]} means the model output u
            will be multiplied by model output v and the result will be named "u_mul_v".
        label_dict (Dict[str, Union[float, Callable]]): Function in dict for computing
            label, which will be a reference value to participate in the loss calculation.
        geom (geometry.Geometry): Geometry where data sampled from.
        dataloader_cfg (Dict[str, Any]): Dataloader config.
        loss (loss.Loss): Loss functor.
        random (Literal["pseudo", "LHS"], optional): Random method for sampling data in
            geometry. Defaults to "pseudo".
        criteria (Optional[Callable]): Criteria for refining specified domain. Defaults to None.
        evenly (bool, optional): Whether to use evenly distribution sampling. Defaults to False.
        metric (Optional[Dict[str, metric.Metric]]): Named metric functors in dict. Defaults to None.
        with_initial (bool, optional): Whether the data contains time t0. Defaults to False.
        name (Optional[str]): Name of validator. Defaults to None.

    Examples:
        >>> import ppsci
        >>> rect = ppsci.geometry.Rectangle((0, 0), (1, 1))
        >>> geom_validator = ppsci.validate.GeometryValidator(
        ...     {"u": lambda out: out["u"]},
        ...     {"u": 0},
        ...     rect,
        ...     {
        ...         "dataset": "IterableNamedArrayDataset",
        ...         "iters_per_epoch": 1,
        ...         "total_size": 32,
        ...         "batch_size": 16,
        ...     },
        ...     ppsci.loss.MSELoss("mean"),
        ... )
    """

    def __init__(
        self,
        output_expr: Dict[str, Callable],
        label_dict: Dict[str, Union[float, Callable]],
        geom: geometry.Geometry,
        dataloader_cfg: Dict[str, Any],
        loss: loss.Loss,
        random: Literal["pseudo", "LHS"] = "pseudo",
        criteria: Optional[Callable] = None,
        evenly: bool = False,
        metric: Optional[Dict[str, metric.Metric]] = None,
        with_initial: bool = False,
        name: Optional[str] = None,
    ):
        self.output_expr = output_expr
        for label_name, expr in self.output_expr.items():
            if isinstance(expr, str):
                self.output_expr[label_name] = sp_parser.parse_expr(expr)

        self.label_dict = label_dict
        self.input_keys = geom.dim_keys
        self.output_keys = list(label_dict.keys())

        nx = dataloader_cfg["total_size"]
        self.num_timestamps = 1
        # TODO(sensen): simplify code below
        if isinstance(geom, geometry.TimeXGeometry):
            if geom.timedomain.num_timestamps is not None:
                if with_initial:
                    # include t0
                    self.num_timestamps = geom.timedomain.num_timestamps
                    assert (
                        nx % self.num_timestamps == 0
                    ), f"{nx} % {self.num_timestamps} != 0"
                    nx //= self.num_timestamps
                    input = geom.sample_interior(
                        nx * (geom.timedomain.num_timestamps - 1),
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
                    self.num_timestamps = geom.timedomain.num_timestamps - 1
                    assert (
                        nx % self.num_timestamps == 0
                    ), f"{nx} % {self.num_timestamps} != 0"
                    nx //= self.num_timestamps
                    input = geom.sample_interior(
                        nx * (geom.timedomain.num_timestamps - 1),
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
                label[key] = np.full_like(next(iter(input.values())), value)
            elif isinstance(value, sympy.Basic):
                func = sympy.lambdify(
                    sympy.symbols(geom.dim_keys),
                    value,
                    [{"amax": lambda xy, _: np.maximum(xy[0], xy[1])}, "numpy"],
                )
                label[key] = func(
                    **{k: v for k, v in input.items() if k in geom.dim_keys}
                )
            elif callable(value):
                func = value
                label[key] = func(input)
                if isinstance(label[key], (int, float)):
                    label[key] = np.full(
                        (next(iter(input.values())).shape[0], 1),
                        label[key],
                        paddle.get_default_dtype(),
                    )
            else:
                raise NotImplementedError(f"type of {type(value)} is invalid yet.")

        weight = {key: np.ones_like(next(iter(label.values()))) for key in label}

        _dataset = getattr(dataset, dataloader_cfg["dataset"])(input, label, weight)
        super().__init__(_dataset, dataloader_cfg, loss, metric, name)
