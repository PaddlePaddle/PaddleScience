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
from ppsci.utils import misc


class IntegralConstraint(base.Constraint):
    """Class for integral constraint.

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
        criteria (Optional[Callable]): Criteria for refining specified boundaries.
            Defaults to None.
        weight_dict (Optional[Dict[str, Callable]]): Define the weight of each
            constraint variable. Defaults to None.
        name (str, optional): Name of constraint object. Defaults to "IgC".

    Examples:
        >>> import ppsci
        >>> rect = ppsci.geometry.Rectangle((0, 0), (1, 1))
        >>> igc = ppsci.constraint.IntegralConstraint(
        ...     {"u": lambda out: out["u"]},
        ...     {"u": 0},
        ...     rect,
        ...     {
        ...         "dataset": "IterableNamedArrayDataset",
        ...         "iters_per_epoch": 1,
        ...         "batch_size": 16,
        ...         "integral_batch_size": 8,
        ...     },
        ...     ppsci.loss.MSELoss("mean"),
        ...     name="IgC",
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
        weight_dict: Optional[Dict[str, Callable]] = None,
        name: str = "IgC",
    ):
        self.output_expr = output_expr
        for label_name, expr in self.output_expr.items():
            if isinstance(expr, str):
                self.output_expr[label_name] = sp_parser.parse_expr(expr)

        self.label_dict = label_dict
        self.input_keys = geom.dim_keys
        self.output_keys = list(label_dict.keys())
        # "area" will be kept in "output_dict" for computation.
        if isinstance(geom, geometry.Mesh):
            self.output_keys += ["area"]

        if isinstance(criteria, str):
            criteria = eval(criteria)

        # prepare input
        input_list = []
        for _ in range(
            dataloader_cfg["batch_size"] * dataloader_cfg["iters_per_epoch"]
        ):
            input = geom.sample_boundary(
                dataloader_cfg["integral_batch_size"], random, criteria
            )
            input_list.append(input)
        input = misc.stack_dict_list(input_list)
        # shape of each input is [batch_size, integral_batch_size, ndim]

        # prepare label
        # shape of each label is [batch_size, ndim]
        label = {}
        for key, value in label_dict.items():
            if isinstance(value, (int, float)):
                label[key] = np.full(
                    (next(iter(input.values())).shape[0], 1),
                    value,
                    paddle.get_default_dtype(),
                )
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

        # prepare weight
        # shape of each weight is [batch_size, ndim]
        weight = {key: np.ones_like(next(iter(label.values()))) for key in label}
        if weight_dict is not None:
            for key, value in weight_dict.items():
                if isinstance(value, str):
                    value = sp_parser.parse_expr(value)

                if isinstance(value, (int, float)):
                    weight[key] = np.full_like(next(iter(label.values())), value)
                elif isinstance(value, sympy.Basic):
                    func = sympy.lambdify(
                        sympy.symbols(geom.dim_keys),
                        value,
                        [{"amax": lambda xy, _: np.maximum(xy[0], xy[1])}, "numpy"],
                    )
                    weight[key] = func(
                        **{k: v for k, v in input.items() if k in geom.dim_keys}
                    )
                elif callable(value):
                    func = value
                    weight[key] = func(input)
                    if isinstance(weight[key], (int, float)):
                        weight[key] = np.full_like(
                            next(iter(input.values())), weight[key]
                        )
                else:
                    raise NotImplementedError(f"type of {type(value)} is invalid yet.")

        # wrap input, label, weight into a dataset
        _dataset = getattr(dataset, dataloader_cfg["dataset"])(input, label, weight)

        # construct dataloader with dataset and dataloader_cfg
        super().__init__(_dataset, dataloader_cfg, loss, name)
