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

import os
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import Optional

import paddle

from ppsci.utils import download
from ppsci.utils import logger

if TYPE_CHECKING:
    from paddle import amp
    from paddle import nn
    from paddle import optimizer

    from ppsci import equation


__all__ = [
    "load_checkpoint",
    "save_checkpoint",
    "load_pretrain",
]


def _load_pretrain_from_path(
    path: str, model: nn.Layer, equation: Optional[Dict[str, equation.PDE]] = None
):
    """Load pretrained model from given path.

    Args:
        path (str): File path of pretrained model, i.e. `/path/to/model.pdparams`.
        model (nn.Layer): Model with parameters.
        equation (Optional[Dict[str, equation.PDE]]): Equations. Defaults to None.
    """
    if not (os.path.isdir(path) or os.path.exists(f"{path}.pdparams")):
        raise FileNotFoundError(
            f"Pretrained model path {path}.pdparams does not exists."
        )

    param_state_dict = paddle.load(f"{path}.pdparams")
    model.set_state_dict(param_state_dict)
    logger.message(f"Finish loading pretrained model from: {path}.pdparams")
    if equation is not None:
        if not os.path.exists(f"{path}.pdeqn"):
            num_learnable_params = sum(
                [len(eq.learnable_parameters) for eq in equation.values()]
            )
            if num_learnable_params > 0:
                logger.warning(
                    f"There are a total of {num_learnable_params} learnable parameters"
                    f" in the equation, but {path}.pdeqn not found."
                )
        else:
            equation_dict = paddle.load(f"{path}.pdeqn")
            for name, _equation in equation.items():
                _equation.set_state_dict(equation_dict[name])
            logger.message(
                f"Finish loading pretrained equation parameters from: {path}.pdeqn"
            )


def load_pretrain(
    model: nn.Layer, path: str, equation: Optional[Dict[str, equation.PDE]] = None
):
    """Load pretrained model from given path or url.

    Args:
        model (nn.Layer): Model with parameters.
        path (str): File path or url of pretrained model, i.e. `/path/to/model.pdparams`
            or `http://xxx.com/model.pdparams`.
        equation (Optional[Dict[str, equation.PDE]]): Equations. Defaults to None.
    """
    if path.startswith("http"):
        path = download.get_weights_path_from_url(path)

    # remove ".pdparams" in suffix of path for convenient
    if path.endswith(".pdparams"):
        path = path[:-9]
    _load_pretrain_from_path(path, model, equation)


def load_checkpoint(
    path: str,
    model: nn.Layer,
    optimizer: optimizer.Optimizer,
    grad_scaler: Optional[amp.GradScaler] = None,
    equation: Optional[Dict[str, equation.PDE]] = None,
) -> Dict[str, Any]:
    """Load from checkpoint.

    Args:
        path (str): Path for checkpoint.
        model (nn.Layer): Model with parameters.
        optimizer (optimizer.Optimizer): Optimizer for model.
        grad_scaler (Optional[amp.GradScaler]): GradScaler for AMP. Defaults to None.
        equation (Optional[Dict[str, equation.PDE]]): Equations. Defaults to None.

    Returns:
        Dict[str, Any]: Loaded metric information.
    """
    if not os.path.exists(f"{path}.pdparams"):
        raise FileNotFoundError(f"{path}.pdparams not exist.")
    if not os.path.exists(f"{path}.pdopt"):
        raise FileNotFoundError(f"{path}.pdopt not exist.")
    if grad_scaler is not None and not os.path.exists(f"{path}.pdscaler"):
        raise FileNotFoundError(f"{path}.scaler not exist.")

    # load state dict
    param_dict = paddle.load(f"{path}.pdparams")
    optim_dict = paddle.load(f"{path}.pdopt")
    metric_dict = paddle.load(f"{path}.pdstates")
    if grad_scaler is not None:
        scaler_dict = paddle.load(f"{path}.pdscaler")
    if equation is not None:
        if not os.path.exists(f"{path}.pdeqn"):
            logger.warning(f"{path}.pdeqn not found.")
            equation_dict = None
        else:
            equation_dict = paddle.load(f"{path}.pdeqn")

    # set state dict
    model.set_state_dict(param_dict)
    optimizer.set_state_dict(optim_dict)
    if grad_scaler is not None:
        grad_scaler.load_state_dict(scaler_dict)
    if equation is not None and equation_dict is not None:
        for name, _equation in equation.items():
            _equation.set_state_dict(equation_dict[name])

    logger.message(f"Finish loading checkpoint from {path}")
    return metric_dict


def save_checkpoint(
    model: nn.Layer,
    optimizer: optimizer.Optimizer,
    metric: Dict[str, float],
    grad_scaler: Optional[amp.GradScaler] = None,
    output_dir: Optional[str] = None,
    prefix: str = "model",
    equation: Optional[Dict[str, equation.PDE]] = None,
):
    """Save checkpoint, including model params, optimizer params, metric information.

    Args:
        model (nn.Layer): Model with parameters.
        optimizer (optimizer.Optimizer): Optimizer for model.
        metric (Dict[str, float]): Metric information, such as {"RMSE": 0.1, "MAE": 0.2}.
        grad_scaler (Optional[amp.GradScaler]): GradScaler for AMP. Defaults to None.
        output_dir (Optional[str]): Directory for checkpoint storage.
        prefix (str, optional): Prefix for storage. Defaults to "model".
        equation (Optional[Dict[str, equation.PDE]]): Equations. Defaults to None.
    """
    if paddle.distributed.get_rank() != 0:
        return

    if output_dir is None:
        logger.warning("output_dir is None, skip save_checkpoint")
        return

    ckpt_dir = os.path.join(output_dir, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, prefix)
    os.makedirs(ckpt_dir, exist_ok=True)

    paddle.save(model.state_dict(), f"{ckpt_path}.pdparams")
    paddle.save(optimizer.state_dict(), f"{ckpt_path}.pdopt")
    paddle.save(metric, f"{ckpt_path}.pdstates")
    if grad_scaler is not None:
        paddle.save(grad_scaler.state_dict(), f"{ckpt_path}.pdscaler")
    if equation is not None:
        num_learnable_params = sum(
            [len(eq.learnable_parameters) for eq in equation.values()]
        )
        if num_learnable_params > 0:
            paddle.save(
                {key: eq.state_dict() for key, eq in equation.items()},
                f"{ckpt_path}.pdeqn",
            )

    logger.message(f"Finish saving checkpoint to {ckpt_path}")
