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

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

from paddle import jit
from paddle import nn

if TYPE_CHECKING:
    import paddle
    from ppsci import constraint
    from ppsci import validate
    from ppsci import arch

from ppsci.autodiff import clear

__all__ = [
    "ExpressionSolver",
]


class ExpressionSolver(nn.Layer):
    """Expression computing helper, which compute named result according to corresponding
    function and related inputs.

    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.MLP(("x", "y"), ("u", "v"), 5, 128)
        >>> expr_solver = ExpressionSolver()
    """

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Use train_forward/eval_forward/visu_forward instead of forward."
        )

    @jit.to_static
    def train_forward(
        self,
        expr_dicts: Tuple[Dict[str, Callable], ...],
        input_dicts: Tuple[Dict[str, "paddle.Tensor"], ...],
        model: arch.Arch,
        constraint: Dict[str, "constraint.Constraint"],
        label_dicts: Tuple[Dict[str, "paddle.Tensor"], ...],
        weight_dicts: Tuple[Dict[str, "paddle.Tensor"], ...],
    ) -> Tuple["paddle.Tensor", ...]:
        """Forward computation for training, including model forward and equation
        forward.

        Args:
            expr_dicts (Tuple[Dict[str, Callable], ...]): Tuple of expression dicts.
            input_dicts (Tuple[Dict[str, paddle.Tensor], ...]): Tuple of input dicts.
            model (arch.Arch): NN model.
            constraint (Dict[str, "constraint.Constraint"]): Constraint dict.
            label_dicts (Tuple[Dict[str, paddle.Tensor], ...]): Tuple of label dicts.
            weight_dicts (Tuple[Dict[str, paddle.Tensor], ...]): Tuple of weight dicts.

        Returns:
            Tuple[paddle.Tensor, ...]: Tuple of losses for each constraint.
        """
        output_dicts = []
        for i, expr_dict in enumerate(expr_dicts):
            # model forward
            output_dict = model(input_dicts[i])

            # equation forward
            data_dict = {k: v for k, v in input_dicts[i].items()}
            data_dict.update(output_dict)
            for name, expr in expr_dict.items():
                output_dict[name] = expr(data_dict)

            # put field 'area' into output_dict
            if "area" in input_dicts[i]:
                output_dict["area"] = input_dicts[i]["area"]

            output_dicts.append(output_dict)

            # clear differentiation cache
            clear()

        # compute loss for each constraint according to its' own output, label and weight
        constraint_losses = []
        for i, _constraint in enumerate(constraint.values()):
            constraint_loss = _constraint.loss(
                output_dicts[i],
                label_dicts[i],
                weight_dicts[i],
            )
            constraint_losses.append(constraint_loss)
        return constraint_losses

    @jit.to_static
    def eval_forward(
        self,
        expr_dict: Dict[str, Callable],
        input_dict: Dict[str, "paddle.Tensor"],
        model: arch.Arch,
        validator: "validate.Validator",
        label_dict: Dict[str, "paddle.Tensor"],
        weight_dict: Dict[str, "paddle.Tensor"],
    ) -> Tuple[Dict[str, "paddle.Tensor"], "paddle.Tensor"]:
        """Forward computation for evaluation, including model forward and equation
        forward.

        Args:
            expr_dict (Dict[str, Callable]): Expression dict.
            input_dict (Dict[str, paddle.Tensor]): Input dict.
            model (arch.Arch): NN model.
            validator (validate.Validator): Validator.
            label_dict (Dict[str, paddle.Tensor]): Label dict.
            weight_dict (Dict[str, paddle.Tensor]): Weight dict.

        Returns:
            Tuple[Dict[str, paddle.Tensor], paddle.Tensor]: Result dict and loss for
                given validator.
        """
        # model forward
        output_dict = model(input_dict)

        # equation forward
        data_dict = {k: v for k, v in input_dict.items()}
        data_dict.update(output_dict)
        for name, expr in expr_dict.items():
            output_dict[name] = expr(data_dict)

        # put field 'area' into output_dict
        if "area" in input_dict:
            output_dict["area"] = input_dict["area"]

        # clear differentiation cache
        clear()

        # compute loss for each validator according to its' own output, label and weight
        validator_loss = validator.loss(
            output_dict,
            label_dict,
            weight_dict,
        )
        return output_dict, validator_loss

    def visu_forward(
        self,
        expr_dict: Optional[Dict[str, Callable]],
        input_dict: Dict[str, "paddle.Tensor"],
        model: arch.Arch,
    ) -> Dict[str, "paddle.Tensor"]:
        """Forward computation for visualization, including model forward and equation
        forward.

        Args:
            expr_dict (Optional[Dict[str, Callable]]): Expression dict.
            input_dict (Dict[str, paddle.Tensor]): Input dict.
            model (arch.Arch): NN model.

        Returns:
            Dict[str, paddle.Tensor]: Result dict for given expression dict.
        """
        # model forward
        output_dict = model(input_dict)

        if isinstance(expr_dict, dict):
            # equation forward
            data_dict = {k: v for k, v in input_dict.items()}
            data_dict.update(output_dict)
            for name, expr in expr_dict.items():
                output_dict[name] = expr(data_dict)

            # clear differentiation cache
            clear()

        return output_dict
