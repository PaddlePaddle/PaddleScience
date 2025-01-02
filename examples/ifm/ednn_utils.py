# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
import paddle.nn.functional as F
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score


def statistical(y_true, y_pred, y_pro):
    c_mat = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = list(c_mat.flatten())
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    acc = (tp + tn) / (tn + fp + fn + tp)
    mcc = (tp * tn - fp * fn) / np.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8
    )
    auc_prc = auc(
        precision_recall_curve(y_true, y_pro, pos_label=1)[1],
        precision_recall_curve(y_true, y_pro, pos_label=1)[0],
    )
    auc_roc = roc_auc_score(y_true, y_pro)
    return tn, fp, fn, tp, se, sp, acc, mcc, auc_prc, auc_roc


class Meter:
    """Track and summarize model performance on a dataset for
    (multi-label) binary classification."""

    def __init__(self):
        self.mask = []
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true, mask):
        """Update for the result of an iteration

        Parameters
        ----------
        y_pred : float32 tensor
            Predicted molecule labels with shape (B, T),
            B for batch size and T for the number of tasks
        y_true : float32 tensor
            Ground truth molecule labels with shape (B, T)
        mask : float32 tensor
            Mask for indicating the existence of ground
            truth labels with shape (B, T)
        """
        print(
            f"y_true.shape: {y_true.shape}, after detach: {y_true.detach().cpu().shape}"
        )
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        self.mask.append(mask.detach().cpu())

    def roc_precision_recall_score(self):
        """Compute AUC_PRC for each task.
        Returns
        -------
        list of float
            rmse for all tasks
        """
        mask = paddle.concat(self.mask, axis=0)
        y_pred = paddle.concat(self.y_pred, axis=0)
        y_pred = paddle.nn.functional.sigmoid(y_pred)
        y_true = paddle.concat(self.y_true, axis=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            precision, recall, _thresholds = precision_recall_curve(
                task_y_true, task_y_pred, pos_label=1
            )
            scores.append(auc(recall, precision))
        return scores

    def roc_auc_score(self):
        """Compute roc-auc score for each task.

        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        mask = paddle.concat(self.mask, axis=0)
        y_pred = paddle.concat(self.y_pred, axis=0)
        y_true = paddle.concat(self.y_true, axis=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = paddle.nn.functional.sigmoid(y_pred)  # 求得为正例的概率
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(roc_auc_score(task_y_true, task_y_pred))
        return scores

    def l1_loss(self, reduction):
        """Compute l1 loss for each task.

        Returns
        -------
        list of float
            l1 loss for all tasks
        reduction : str
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        """
        mask = paddle.concat(self.mask, axis=0)
        y_pred = paddle.concat(self.y_pred, axis=0)
        y_true = paddle.concat(self.y_true, axis=0)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            scores.append(
                F.l1_loss(task_y_true, task_y_pred, reduction=reduction).item()
            )
        return scores

    def rmse(self):
        """Compute RMSE for each task.

        Returns
        -------
        list of float
            rmse for all tasks
        """
        mask = paddle.concat(self.mask, axis=0)
        y_pred = paddle.concat(self.y_pred, axis=0)
        y_true = paddle.concat(self.y_true, axis=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            scores.append(np.sqrt(F.mse_loss(task_y_pred, task_y_true).cpu().item()))
        return scores

    def mae(self):
        """Compute mae for each task.

        Returns
        -------
        list of float
            mae for all tasks
        """
        mask = paddle.concat(self.mask, axis=0)
        y_pred = paddle.concat(self.y_pred, axis=0)
        y_true = paddle.concat(self.y_true, axis=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            scores.append(mean_absolute_error(task_y_true, task_y_pred))
        return scores

    def r2(self):
        """Compute r2 score for each task.

        Returns
        -------
        list of float
            r2 score for all tasks
        """
        mask = paddle.concat(self.mask, axis=0)
        y_pred = paddle.concat(self.y_pred, axis=0)
        y_true = paddle.concat(self.y_true, axis=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            scores.append(r2_score(task_y_true, task_y_pred))
        return scores

    def compute_metric(self, metric_name, reduction="mean"):
        """Compute metric for each task.

        Parameters
        ----------
        metric_name : str
            Name for the metric to compute.
        reduction : str
            Only comes into effect when the metric_name is l1_loss.
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task

        Returns
        -------
        list of float
            Metric value for each task
        """
        assert metric_name in [
            "roc_auc",
            "l1",
            "rmse",
            "prc_auc",
            "mae",
            "r2",
        ], 'Expect metric name to be "roc_auc", "l1", "rmse", "prc_auc", "mae", "r2" got {}'.format(
            metric_name
        )  # assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常
        assert reduction in ["mean", "sum"]
        if metric_name == "roc_auc":
            return self.roc_auc_score()
        if metric_name == "l1":
            return self.l1_loss(reduction)
        if metric_name == "rmse":
            return self.rmse()
        if metric_name == "prc_auc":
            return self.roc_precision_recall_score()
        if metric_name == "mae":
            return self.mae()
        if metric_name == "r2":
            return self.r2()
        if metric_name == "mcc":
            return self.mcc()
