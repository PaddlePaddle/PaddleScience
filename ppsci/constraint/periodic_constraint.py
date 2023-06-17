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
from ppsci.constraint import base
from ppsci.data import dataset


class PeriodicConstraint(base.Constraint):
    """Class for periodic constraint.

    Args:
        output_expr (Dict[str, Callable]): Function in dict for computing output.
            e.g. {"u_mul_v": lambda out: out["u"] * out["v"]} means the model output u
            will be multiplied by model output v and the result will be named "u_mul_v".
        label_dict (Dict[str, Union[float, Callable]]): Function in dict for computing
            label, which will be a reference value to participate in the loss calculation.
        geom (geometry.Geometry): Geometry where data sampled from.
        dataloader_cfg (Dict[str, Any]): Dataloader config.
        periodic_key (str): name of dimension which periodic constraint applied to.
        loss (loss.Loss): Loss functor.
        random (Literal["pseudo", "LHS"], optional): Random method for sampling data in
            geometry. Defaults to "pseudo".
        criteria (Optional[Callable]): Criteria for refining specified boundaries.
            Defaults to None.
        evenly (bool, optional):  Whether to use evenly distribution sampling.
            Defaults to False.
        weight_dict (Optional[Dict[str, Callable]]): Define the weight of each
            constraint variable. Defaults to None.
        name (str, optional): Name of constraint object. Defaults to "PeriodicBC".
    """

    def __init__(
        self,
        output_expr: Dict[str, Callable],
        label_dict: Dict[str, Union[float, Callable]],
        geom: geometry.Geometry,
        periodic_key: str,
        dataloader_cfg: Dict[str, Any],
        loss: loss.Loss,
        random: Literal["pseudo", "LHS"] = "pseudo",
        criteria: Optional[Callable] = None,
        evenly: bool = False,
        weight_dict: Optional[Dict[str, Callable]] = None,
        name: str = "PeriodicBC",
    ):
        self.output_expr = output_expr
        for label_name, expr in self.output_expr.items():
            if isinstance(expr, str):
                self.output_expr[label_name] = sp_parser.parse_expr(expr)

        self.input_keys = geom.dim_keys
        self.output_keys = list(output_expr.keys())

        if isinstance(criteria, str):
            criteria = eval(criteria)

        if dataloader_cfg["sampler"]["batch_size"] % 2 > 0:
            raise ValueError(
                f"batch_size({dataloader_cfg['sampler']['batch_size']}) "
                f"should be positive and even when using PeriodicConstraint"
            )
        if dataloader_cfg["sampler"]["shuffle"]:
            raise ValueError(
                f"shuffle({dataloader_cfg['sampler']['batch_size']}) "
                f"should be False when using PeriodicConstraint "
            )

        # prepare input
        _bs_half = dataloader_cfg["sampler"]["batch_size"] // 2
        input = geom.sample_boundary(
            _bs_half * dataloader_cfg["iters_per_epoch"],
            random,
            criteria,
            evenly,
        )
        input_periodic = geom.periodic_point(
            input,
            geom.geometry.dim_keys.index(periodic_key)
            if isinstance(geom, geometry.TimeXGeometry)
            else geom.dim_keys.index(periodic_key),
        )
        # concatenate original data next to periodic data, i.e.
        # [orignal1, periodic1, orignal2, periodic2, ..., orignalN, periodicN]
        mixed_input = {}
        for key in input:
            mixed_input[key] = []
            for iter_id in range(dataloader_cfg["iters_per_epoch"]):
                mixed_input[key].append(
                    input[key][iter_id * _bs_half : (iter_id + 1) * _bs_half]
                )
                mixed_input[key].append(
                    input_periodic[key][iter_id * _bs_half : (iter_id + 1) * _bs_half]
                )
            mixed_input[key] = np.vstack(mixed_input[key])

        # prepare label, keep label the same shape as input_periodic
        label = {}
        for key, value in label_dict.items():
            # set all label's to zero for dummy data.
            label[key] = np.full(
                (next(iter(mixed_input.values())).shape[0], 1),
                0,
                paddle.get_default_dtype(),
            )

        # # prepare weight, keep weight the same shape as input_periodic
        weight = {key: np.ones_like(next(iter(label.values()))) for key in label}
        if weight_dict is not None:
            for key, value in weight_dict.items():
                if isinstance(value, str):
                    value = sp_parser.parse_expr(value)

                if isinstance(value, (int, float)):
                    weight[key] = np.full_like(next(iter(label.values())), value)
                elif isinstance(value, sympy.Basic):
                    func = sympy.lambdify(
                        [sympy.Symbol(k) for k in geom.dim_keys],
                        value,
                        [{"amax": lambda xy, _: np.maximum(xy[0], xy[1])}, "numpy"],
                    )
                    weight[key] = func(**{k: mixed_input[k] for k in geom.dim_keys})
                elif callable(value):
                    func = value
                    weight[key] = func(mixed_input)
                    if isinstance(weight[key], (int, float)):
                        weight[key] = np.full_like(
                            next(iter(mixed_input.values())), weight[key]
                        )
                else:
                    raise NotImplementedError(f"type of {type(value)} is invalid yet.")

        # wrap input, label, weight into a dataset
        _dataset = getattr(dataset, dataloader_cfg["dataset"])(
            mixed_input, label, weight
        )

        # construct dataloader with dataset and dataloader_cfg
        super().__init__(_dataset, dataloader_cfg, loss, name)
