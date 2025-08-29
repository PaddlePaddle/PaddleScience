from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
import paddle
from paddle.nn import functional as F

from ppsci.data.dataset.enso_dataset import NINO_WINDOW_T
from ppsci.data.dataset.enso_dataset import scale_back_sst


def compute_enso_score(
    y_pred: paddle.Tensor,
    y_true: paddle.Tensor,
    acc_weight: Optional[Union[str, np.ndarray, paddle.Tensor]] = None,
):
    """Compute the accuracy and Root Mean Squared Error (RMSE) of enso dataset.

    Args:
        y_pred (paddle.Tensor): The predict data.
        y_true (paddle.Tensor): The label data.
        acc_weight (Optional[Union[str, np.ndarray, paddle.Tensor]], optional): The wight of accuracy. Defaults to None.use
            default acc_weight specified at https://tianchi.aliyun.com/competition/entrance/531871/information.
    """

    pred = y_pred - y_pred.mean(axis=0, keepdim=True)  # (N, 24)
    true = y_true - y_true.mean(axis=0, keepdim=True)  # (N, 24)
    cor = (pred * true).sum(axis=0) / (
        paddle.sqrt(paddle.sum(pred**2, axis=0) * paddle.sum(true**2, axis=0))
        + 1e-6
    )

    if acc_weight is None:
        acc = cor.sum()
    else:
        nino_out_len = y_true.shape[-1]
        if acc_weight == "default":
            acc_weight = paddle.to_tensor(
                [1.5] * 4 + [2] * 7 + [3] * 7 + [4] * (nino_out_len - 18)
            )[:nino_out_len] * paddle.log(paddle.arange(nino_out_len) + 1)
        elif isinstance(acc_weight, np.ndarray):
            acc_weight = paddle.to_tensor(acc_weight[:nino_out_len])
        elif isinstance(acc_weight, paddle.Tensor):
            acc_weight = acc_weight[:nino_out_len]
        else:
            raise ValueError(f"Invalid acc_weight {acc_weight}!")
        acc_weight = acc_weight.to(y_pred)
        acc = (acc_weight * cor).sum()
    rmse = paddle.mean((y_pred - y_true) ** 2, axis=0).sqrt().sum()
    return acc, rmse


def sst_to_nino(sst: paddle.Tensor, normalize_sst: bool = True, detach: bool = True):
    """Convert sst to nino index.

    Args:
        sst (paddle.Tensor): The predict data for sst. Shape = (N, T, H, W)
        normalize_sst (bool, optional): Whether to use normalize for sst. Defaults to True.
        detach (bool, optional): Whether to detach the tensor. Defaults to True.

    Returns:
        nino_index (paddle.Tensor): The nino index. Shape = (N, T-NINO_WINDOW_T+1)
    """

    if detach:
        nino_index = sst.detach()
    else:
        nino_index = sst
    if normalize_sst:
        nino_index = scale_back_sst(nino_index)
    nino_index = nino_index[:, :, 10:13, 19:30].mean(axis=[2, 3])  # (N, 26)
    nino_index = nino_index.unfold(axis=1, size=NINO_WINDOW_T, step=1).mean(
        axis=2
    )  # (N, 24)

    return nino_index


def train_mse_func(
    output_dict: Dict[str, "paddle.Tensor"],
    label_dict: Dict[str, "paddle.Tensor"],
    *args,
) -> paddle.Tensor:
    return {
        "sst_target": F.mse_loss(output_dict["sst_target"], label_dict["sst_target"])
    }


def eval_rmse_func(
    output_dict: Dict[str, "paddle.Tensor"],
    label_dict: Dict[str, "paddle.Tensor"],
    nino_out_len: int = 12,
    *args,
) -> Dict[str, paddle.Tensor]:
    pred = output_dict["sst_target"]
    sst_target = label_dict["sst_target"]
    nino_target = label_dict["nino_target"].astype("float32")
    # mse
    mae = F.l1_loss(pred, sst_target)
    # mse
    mse = F.mse_loss(pred, sst_target)
    # rmse
    nino_preds = sst_to_nino(sst=pred[..., 0])
    nino_preds_list, nino_target_list = map(list, zip((nino_preds, nino_target)))
    nino_preds_list = paddle.concat(nino_preds_list, axis=0)
    nino_target_list = paddle.concat(nino_target_list, axis=0)

    valid_acc, valid_nino_rmse = compute_enso_score(
        y_pred=nino_preds_list, y_true=nino_target_list, acc_weight=None
    )
    valid_weighted_acc, _ = compute_enso_score(
        y_pred=nino_preds_list, y_true=nino_target_list, acc_weight="default"
    )
    valid_acc /= nino_out_len
    valid_nino_rmse /= nino_out_len
    valid_weighted_acc /= nino_out_len
    valid_loss = -valid_acc

    return {
        "valid_loss_epoch": valid_loss,
        "mse": mse,
        "mae": mae,
        "rmse": valid_nino_rmse,
        "corr_nino3.4_epoch": valid_acc,
        "corr_nino3.4_weighted_epoch": valid_weighted_acc,
    }
