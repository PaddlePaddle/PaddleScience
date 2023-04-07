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

import types
from typing import Any
from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np
import sympy
from sympy.parsing import sympy_parser as sp_parser

from ppsci import loss
from ppsci.constraint import base
from ppsci.data import dataset
from ppsci.utils import misc


class SupervisedConstraint(base.Constraint):
    """Class for supervised constraint.

    Args:
        data_file (str): File path of data.
        input_keys (Tuple[str, ...]): List of input keys.
        label_keys (Tuple[str, ...]): List of label keys.
        alias_dict (Dict[str, str]): Dict of alias(es) for input and label keys.
        dataloader_cfg (Dict[str, Any]): Dataloader config.
        loss (loss.LossBase): Loss functor.
        weight_dict (Dict[str, Callable], optional): Define the weight of each
            constraint variable. Defaults to None.
        timestamps (Tuple[float, ...], optional): The number of repetitions of the data
            in the time dimension. Defaults to None.
        name (str, optional): Name of constraint object. Defaults to "Sup".
    """

    def __init__(
        self,
        data_file: str,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        alias_dict: Dict[str, str],
        dataloader_cfg: Dict[str, Any],
        loss: loss.LossBase,
        weight_value=1.0,
        weight_dict: Dict[str, Callable] = None,
        timestamps: Tuple[float, ...] = None,
        name: str = "Sup",
    ):
        self.input_keys = [
            alias_dict[key] if key in alias_dict else key for key in input_keys
        ]
        self.output_keys = [
            alias_dict[key] if key in alias_dict else key for key in label_keys
        ]

        if str(data_file).endswith(".csv"):
            # load data
            data = self._load_csv_file(data_file, input_keys + label_keys, alias_dict)
            if "t" not in data and timestamps is None:
                raise ValueError(
                    "Time should be given by argument timestamps or data itself."
                )
            if timestamps is not None:
                if "t" in data:
                    # filter data according to given timestamps
                    raw_time_array = data["t"]
                    mask = []
                    for ti in timestamps:
                        mask.append(
                            np.nonzero(np.isclose(raw_time_array, ti).flatten())[0]
                        )
                    data = misc.convert_to_array(
                        data, self.input_keys + self.output_keys
                    )
                    mask = np.concatenate(mask, 0)
                    data = data[mask]
                    data = misc.convert_to_dict(
                        data, self.input_keys + self.output_keys
                    )
                else:
                    # repeat data according to given timestamps
                    data = misc.convert_to_array(
                        data, self.input_keys + self.output_keys
                    )
                    data = misc.combine_array_with_time(data, timestamps)
                    self.input_keys = ["t"] + self.input_keys
                    data = misc.convert_to_dict(
                        data, self.input_keys + self.output_keys
                    )
                input = {key: data[key] for key in self.input_keys}
                label = {key: data[key] for key in self.output_keys}
                self.num_timestamp = len(timestamps)
            else:
                # use all input and label
                input = {key: data[key] for key in self.input_keys}
                label = {key: data[key] for key in self.output_keys}
                self.num_timestamp = len(np.unique(data["t"]))

            self.label_expr = {key: (lambda d, k=key: d[k]) for key in self.output_keys}
        elif type(data_file) is dict:
            data = data_file
            input = {key: data[input_keys[i]] for i, key in enumerate(self.input_keys)}
            label = {key: data[label_keys[i]] for i, key in enumerate(self.output_keys)}
            for key, value in label.items():
                if isinstance(value, (int, float)):
                    label[key] = np.full_like(next(iter(input.values())), float(value))
            self.label_expr = {key: (lambda d, k=key: d[k]) for key in self.output_keys}
        else:
            raise NotImplementedError("Only suppport .csv file now.")

        weight = {
            key: weight_value * np.ones_like(next(iter(label.values())))
            for key in label
        }

        if weight_dict is not None:
            for key, value in weight_dict.items():
                if isinstance(value, str):
                    value = sp_parser.parse_expr(value)

                if isinstance(value, (int, float)):
                    weight[key] = np.full_like(next(iter(label.values())), float(value))
                elif isinstance(value, sympy.Basic):
                    func = sympy.lambdify(
                        [sympy.Symbol(k) for k in self.input_keys],
                        value,
                        [{"amax": lambda xy, _: np.maximum(xy[0], xy[1])}, "numpy"],
                    )
                    weight[key] = func(**{k: input[k] for k in self.input_keys})
                elif isinstance(value, types.FunctionType):
                    func = value
                    weight[key] = func(input)
                    if isinstance(weight[key], (int, float)):
                        weight[key] = np.full_like(
                            next(iter(input.values())), float(weight[key])
                        )
                else:
                    raise NotImplementedError(f"type of {type(value)} is invalid yet.")

        # wrap input, label, weight into a dataset
        _dataset = getattr(dataset, dataloader_cfg["dataset"])(input, label, weight)

        # construct dataloader with dataset and dataloader_cfg
        super().__init__(_dataset, dataloader_cfg, loss, name)


class SupervisedInitialConstraint(base.Constraint):
    """Class for supervised initial constraint.

    Args:
        data_file (str): File path of data.
        input_keys (Tuple[str, ...]): List of input keys.
        label_keys (Tuple[str, ...]): List of label keys.
        t0 (float): Initial timestamp. Defaults to None.
        alias_dict (Dict[str, str]): Dict of alias for input and label keys.
        dataloader_cfg (Dict[str, Any]): Dataloader config.
        loss (loss.LossBase): Loss functor.
        weight_dict (Dict[str, Callable], optional): Weight for label. Defaults to None.
        name (str, optional): Name of constraint object. Defaults to "SupIC".
    """

    def __init__(
        self,
        data_file: str,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        t0: float,
        alias_dict: Dict[str, str],
        dataloader_cfg: Dict[str, Any],
        loss: loss.LossBase,
        weight_value=1.0,
        weight_dict: Dict[str, Callable] = None,
        name: str = "SupIC",
    ):
        self.input_keys = [
            alias_dict[key] if key in alias_dict else key for key in input_keys
        ]
        self.output_keys = [
            alias_dict[key] if key in alias_dict else key for key in label_keys
        ]
        if str(data_file).endswith(".csv"):
            # load data
            data = misc.load_csv_file(data_file, input_keys + label_keys, alias_dict)
            if "t" not in data and t0 is None:
                raise ValueError("Time should be given by arg t0 or data itself.")
            if t0 is not None:
                data = misc.convert_to_array(data, self.input_keys + self.output_keys)
                data = misc.combine_array_with_time(data, [t0])
                self.input_keys = ["t"] + self.input_keys
                data = misc.convert_to_dict(data, self.input_keys + self.output_keys)
                input = {key: data[key] for key in self.input_keys}
                label = {key: data[key] for key in self.output_keys}
            else:
                # time already in data and "t" in input_keys
                input = {key: data[key] for key in self.input_keys}
                label = {key: data[key] for key in self.output_keys}

            self.label_expr = {key: (lambda d, k=key: d[k]) for key in self.output_keys}
            self.num_timestamp = 1
        elif type(data_file) is dict:
            data = data_file
            input = {key: data[input_keys[i]] for i, key in enumerate(self.input_keys)}
            label = {key: data[label_keys[i]] for i, key in enumerate(self.output_keys)}

            for key, value in label.items():
                if isinstance(value, (int, float)):
                    label[key] = np.full_like(next(iter(input.values())), float(value))

            self.label_expr = {key: (lambda d, k=key: d[k]) for key in self.output_keys}
        else:
            raise NotImplementedError("Only suppport .csv file now.")

        weight = {
            key: weight_value * np.ones_like(next(iter(label.values())))
            for key in label
        }
        if weight_dict is not None:
            for key, value in weight_dict.items():
                if isinstance(value, str):
                    value = sp_parser.parse_expr(value)

                if isinstance(value, (int, float)):
                    weight[key] = np.full_like(next(iter(label.values())), float(value))
                elif isinstance(value, sympy.Basic):
                    func = sympy.lambdify(
                        [sympy.Symbol(k) for k in self.input_keys],
                        value,
                        [{"amax": lambda xy, _: np.maximum(xy[0], xy[1])}, "numpy"],
                    )
                    weight[key] = func(**{k: input[k] for k in self.input_keys})
                elif isinstance(value, types.FunctionType):
                    func = value
                    weight[key] = func(input)
                    if isinstance(weight[key], (int, float)):
                        weight[key] = np.full_like(
                            next(iter(input.values())), float(weight[key])
                        )
                else:
                    raise NotImplementedError(f"type of {type(value)} is invalid yet.")

        # wrap input, label, weight into a dataset
        _dataset = getattr(dataset, dataloader_cfg["dataset"])(input, label, weight)

        # construct dataloader with dataset and dataloader_cfg
        super().__init__(_dataset, dataloader_cfg, loss, name)
