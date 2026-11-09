"""Loss and metric functions for the ClimaX ClimateBench example.

Latitude weights are ``cos(lat) / mean(cos(lat))``. Loss expressions take
``(output_dict, label_dict, *args)`` for ``ppsci.loss.FunctionalLoss``; metric
expressions take ``(output_dict, label_dict)`` and return a per-batch dict,
whose arithmetic mean over batches (Solver with ``compute_metric_by_batch=
True``) matches the per-batch averaging of the original training loop.
"""

from __future__ import annotations

from typing import Callable
from typing import Dict
from typing import Optional

import numpy as np
import paddle

from ppsci.metric import base


def _identity(t):
    return t


def mse_loss(output_dict: Dict[str, paddle.Tensor], label_dict, *args):
    """Training loss: mean squared error.

    The mean over the variable dim, then over all remaining dims, of
    ``(pred - y) ** 2``.

    Args:
        output_dict (Dict[str, paddle.Tensor]): Model output, each (B, V, H, W).
        label_dict (Dict[str, paddle.Tensor]): Label, each (B, V, H, W).

    Returns:
        Dict[str, paddle.Tensor]: Summed per-key MSE under key "mse".
    """
    losses = []
    for key in output_dict:
        loss = (output_dict[key] - label_dict[key]) ** 2
        losses.append(loss.mean(axis=1).mean())
    return {"mse": paddle.sum(paddle.stack(losses))}


def make_w_mse_loss(
    lat: np.ndarray, log_postfix: Optional[str] = None
) -> Callable[..., Dict[str, paddle.Tensor]]:
    """Build the latitude-weighted MSE loss used for validation.

    The returned expression computes ``w_mse``, the aggregate key of
    ``lat_weighted_mse_val``: the latitude-weighted MSE of the normalized
    prediction (no denormalization), averaged over variables.

    Args:
        lat (np.ndarray): Latitude axis, shape (H,).
        log_postfix (Optional[str]): Postfix appended to per-variable metric
            names. Defaults to None.

    Returns:
        Callable[..., Dict[str, paddle.Tensor]]: Loss expression for
            ``ppsci.loss.FunctionalLoss``.
    """

    def w_mse(output_dict: Dict[str, paddle.Tensor], label_dict, *args):
        loss_dict = _lat_weighted_mse_val(output_dict, label_dict, lat, log_postfix)
        return {"w_mse": loss_dict["w_mse"]}

    return w_mse


def make_lat_weighted_metrics(
    lat: np.ndarray,
    transform: Optional[Callable[[paddle.Tensor], paddle.Tensor]] = None,
    clim: Optional[paddle.Tensor] = None,
    log_postfix: Optional[str] = None,
) -> Dict[str, base.Metric]:
    """Build the latitude-weighted metric family for validation and test.

    Computes the metric list of the original validation/test step:
    ``lat_weighted_mse_val``, ``lat_weighted_rmse`` and ``lat_weighted_nrmse``
    (the latter combining the spread and grid nrmse variants).

    Args:
        lat (np.ndarray): Latitude axis, shape (H,).
        transform (Optional[Callable]): Denormalization applied to prediction
            and label before rmse/nrmse. Defaults to None (identity).
        clim (Optional[paddle.Tensor]): Normalization scalar of the test
            outputs (``dataset.y_normalization``), required by the nrmse
            variant. Defaults to None.
        log_postfix (Optional[str]): Postfix appended to per-variable metric
            names. Defaults to None.

    Returns:
        Dict[str, base.Metric]: Metric instances keyed "w_mse", "w_rmse" and
            "w_nrmse".
    """
    if transform is None:
        transform = _identity

    def w_mse_expr(output_dict, label_dict):
        return _lat_weighted_mse_val(output_dict, label_dict, lat, log_postfix)

    def w_rmse_expr(output_dict, label_dict):
        return _lat_weighted_rmse(output_dict, label_dict, lat, transform, log_postfix)

    def w_nrmse_expr(output_dict, label_dict):
        variables = list(output_dict.keys())
        pred = output_dict[variables[0]]
        y = label_dict[variables[0]]
        return _lat_weighted_nrmse(pred, y, transform, variables, lat, clim)

    return {
        "w_mse": PerBatchFunctionalMetric(w_mse_expr),
        "w_rmse": PerBatchFunctionalMetric(w_rmse_expr),
        "w_nrmse": PerBatchFunctionalMetric(w_nrmse_expr),
    }


class PerBatchFunctionalMetric(base.Metric):
    """Dict-valued per-batch metric for evaluation with Solver.

    ``ppsci.metric.FunctionalMetric`` supports dict-valued expressions only
    with ``keep_batch=False``, which conflicts with ``compute_metric_by_batch=
    True``; this wrapper instead emits each per-batch scalar as a length-1
    tensor, so Solver concatenates the values over batches and takes their
    arithmetic mean, matching the per-batch averaging of the original
    training loop.

    Args:
        metric_expr (Callable): Expression with signature
            ``(output_dict, label_dict) -> Dict[str, paddle.Tensor]`` whose
            values are per-batch scalars.
    """

    def __init__(self, metric_expr: Callable):
        super().__init__(keep_batch=True)
        self.metric_expr = metric_expr

    @paddle.no_grad()
    def forward(self, output_dict, label_dict=None) -> Dict[str, paddle.Tensor]:
        metric_dict = self.metric_expr(output_dict, label_dict)
        return {key: value.reshape([1]) for key, value in metric_dict.items()}


def _lat_weighted_mse_val(output_dict, label_dict, lat, log_postfix):
    """Latitude-weighted MSE of the normalized prediction (per-batch dict)."""
    variables = list(output_dict.keys())
    pred = output_dict[variables[0]]
    y = label_dict[variables[0]]
    error = (pred - y) ** 2
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = (
        paddle.from_numpy(w_lat)
        .unsqueeze(0)
        .unsqueeze(-1)
        .to(dtype=error.dtype, device=error.device)
    )
    loss_dict = {}
    for i, var in enumerate(variables):
        loss_dict[f"w_mse_{var}_{log_postfix}"] = (error[:, i] * w_lat).mean()
    # mean over variables (single value in this example)
    loss_dict["w_mse"] = paddle.mean(paddle.stack(list(loss_dict.values())))
    return loss_dict


def _lat_weighted_rmse(output_dict, label_dict, lat, transform, log_postfix):
    """Latitude-weighted RMSE after transform (per-batch dict)."""
    variables = list(output_dict.keys())
    pred = transform(output_dict[variables[0]])
    y = transform(label_dict[variables[0]])
    error = (pred - y) ** 2
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = (
        paddle.from_numpy(w_lat)
        .unsqueeze(0)
        .unsqueeze(-1)
        .to(dtype=error.dtype, device=error.device)
    )
    loss_dict = {}
    for i, var in enumerate(variables):
        loss_dict[f"w_rmse_{var}_{log_postfix}"] = paddle.mean(
            paddle.sqrt(paddle.mean(error[:, i] * w_lat, axis=(-2, -1)))
        )
    # mean over variables (single value in this example)
    loss_dict["w_rmse"] = paddle.mean(paddle.stack(list(loss_dict.values())))
    return loss_dict


def _lat_weighted_nrmse(pred, y, transform, variables, lat, clim):
    """Latitude-weighted nrmse: spread and grid variants plus their 1:5 combo."""
    nrmses = _lat_weighted_nrmses(pred, y, transform, variables, lat, clim)
    nrmseg = _lat_weighted_nrmseg(pred, y, transform, variables, lat, clim)
    loss_dict = {}
    for var in variables:
        loss_dict[f"w_nrmses_{var}"] = nrmses[f"w_nrmses_{var}"]
        loss_dict[f"w_nrmseg_{var}"] = nrmseg[f"w_nrmseg_{var}"]
        loss_dict[f"w_nrmse_{var}"] = (
            nrmses[f"w_nrmses_{var}"] + 5 * nrmseg[f"w_nrmseg_{var}"]
        )
    return loss_dict


def _lat_weighted_nrmses(pred, y, transform, variables, lat, clim):
    """Latitude-weighted nrmse of the spread (batch-mean) fields."""
    pred = transform(pred)
    y = transform(y)
    y_normalization = clim
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    # (H, 1): weights align with the latitude dim of the (H, W) error field
    w_lat = paddle.from_numpy(w_lat).unsqueeze(-1).to(dtype=y.dtype, device=y.device)
    loss_dict = {}
    for i, var in enumerate(variables):
        pred_ = pred[:, i]
        y_ = y[:, i]
        error = (paddle.mean(pred_, axis=0) - paddle.mean(y_, axis=0)) ** 2
        error = paddle.mean(error * w_lat)
        loss_dict[f"w_nrmses_{var}"] = paddle.sqrt(error) / y_normalization
    return loss_dict


def _lat_weighted_nrmseg(pred, y, transform, variables, lat, clim):
    """Latitude-weighted nrmse of the grid-mean (per-sample) fields."""
    pred = transform(pred)
    y = transform(y)
    y_normalization = clim
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = (
        paddle.from_numpy(w_lat)
        .unsqueeze(0)
        .unsqueeze(-1)
        .to(dtype=y.dtype, device=y.device)
    )
    loss_dict = {}
    for i, var in enumerate(variables):
        pred_ = pred[:, i]
        pred_ = paddle.mean(pred_ * w_lat, axis=(-2, -1))
        y_ = y[:, i]
        y_ = paddle.mean(y_ * w_lat, axis=(-2, -1))
        error = paddle.mean((pred_ - y_) ** 2)
        loss_dict[f"w_nrmseg_{var}"] = paddle.sqrt(error) / y_normalization
    return loss_dict
