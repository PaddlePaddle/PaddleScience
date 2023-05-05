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

import errno
import os
from typing import Any
from typing import Dict

import paddle

from ppsci.utils import download
from ppsci.utils import logger

__all__ = ["load_checkpoint", "save_checkpoint", "load_pretrain"]


def _mkdir_if_not_exist(path):
    """mkdir if not exists, ignore the exception when multiprocess mkdir together

    Args:
        path (str): Path for makedir
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning(f"{path} already created")
            else:
                raise OSError(f"Failed to mkdir {path}")


def _load_pretrain_from_path(model, path, equation=None):
    """Load pretrained model from given path.

    Args:
        model (nn.Layer): Model with parameters.
        path (str, optional): Pretrained model path.
        equation (Optional[Dict[str, ppsci.equation.PDE]]): Equations. Defaults to None.
    """
    if not (os.path.isdir(path) or os.path.exists(path + ".pdparams")):
        raise FileNotFoundError(
            f"Pretrained model path {path}.pdparams does not exists."
        )

    param_state_dict = paddle.load(path + ".pdparams")
    model.set_dict(param_state_dict)
    if equation is not None:
        equation_dict = paddle.load(path + ".pdeqn")
        for name, _equation in equation.items():
            _equation.set_state_dict(equation_dict[name])

    logger.info(f"Finish loading pretrained model from {path}")


def load_pretrain(model, path, equation=None):
    """Load pretrained model from given path or url.

    Args:
        model (nn.Layer): Model with parameters.
        path (str): Pretrained model url.
        equation (Optional[Dict[str, ppsci.equation.PDE]]): Equations. Defaults to None.
    """
    if path.startswith("http"):
        path = download.get_weights_path_from_url(path).replace(".pdparams", "")
    _load_pretrain_from_path(model, path, equation)


def load_checkpoint(
    path, model, optimizer, grad_scaler=None, equation=None
) -> Dict[str, Any]:
    """Load from checkpoint.

    Args:
        path (AttrDict): Path for checkpoint.
        model (nn.Layer): Model with parameters.
        optimizer (optimizer.Optimizer, optional): Optimizer for model.
        grad_scaler (Optional[amp.GradScaler]): GradScaler for AMP. Defaults to None.
        equation (Optional[Dict[str, ppsci.equation.PDE]]): Equations. Defaults to None.

    Returns:
        Dict[str, Any]: Loaded metric information.
    """
    if not os.path.exists(path + ".pdparams"):
        raise FileNotFoundError(f"{path}.pdparams not exist.")
    if not os.path.exists(path + ".pdopt"):
        raise FileNotFoundError(f"{path}.pdopt not exist.")
    if grad_scaler is not None and not os.path.exists(path + ".pdscaler"):
        raise FileNotFoundError(f"{path}.scaler not exist.")

    # load state dict
    param_dict = paddle.load(path + ".pdparams")
    optim_dict = paddle.load(path + ".pdopt")
    metric_dict = paddle.load(path + ".pdstates")
    if grad_scaler is not None:
        scaler_dict = paddle.load(path + ".pdscaler")
    # if equation is not None:
    # equation_dict = paddle.load(path + ".pdeqn")

    # set state dict
    model.set_state_dict(param_dict)
    optimizer.set_state_dict(optim_dict)
    if grad_scaler is not None:
        grad_scaler.load_state_dict(scaler_dict)
    # if equation is not None:
    #     for name, _equation in equation.items():
    #         _equation.set_state_dict(equation_dict[name])

    logger.info(f"Finish loading checkpoint from {path}")
    return metric_dict


def save_checkpoint(
    model, optimizer, grad_scaler, metric, model_dir, prefix="model", equation=None
):
    """Save checkpoint, including model params, optimizer params, metric information.

    Args:
        model (nn.Layer): Model with parameters.
        optimizer (optimizer.Optimizer): Optimizer for model.
        grad_scaler (Optional[amp.GradScaler]): GradScaler for AMP. Defaults to None.
        metric (Dict[str, Any]): Metric information, such as {"RMSE": ...}.
        model_dir (str): Directory for chekpoint storage.
        prefix (str, optional): Prefix for storage. Defaults to "ppsci".
        equation (Optional[Dict[str, ppsci.equation.PDE]]): Equations. Defaults to None.
    """
    if paddle.distributed.get_rank() != 0:
        return
    model_dir = os.path.join(model_dir, "checkpoints")
    _mkdir_if_not_exist(model_dir)
    model_path = os.path.join(model_dir, prefix)

    paddle.save(model.state_dict(), model_path + ".pdparams")
    paddle.save(optimizer.state_dict(), model_path + ".pdopt")
    paddle.save(metric, model_path + ".pdstates")
    if grad_scaler is not None:
        paddle.save(grad_scaler.state_dict(), model_path + ".pdscaler")
    if equation is not None:
        paddle.save(
            {key: eq.state_dict() for key, eq in equation.items()},
            model_path + ".pdeqn",
        )

    logger.info(f"Finish saving checkpoint to {model_path}")
