import paddle as torch
import paddle
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import numpy as np
import paddle.nn.functional as F #import torch.nn.functional as F
import paddle.nn as nn #import torch.nn as nn
import datetime
import random
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, auc, \
    mean_absolute_error, r2_score, matthews_corrcoef
import pickle

def init_parameter_uniform(parameter: paddle.base.framework.EagerParamBase, n: int) -> None:
    nn.init.uniform_(parameter, -1/np.sqrt(n), 1/np.sqrt(n))

def statistical(y_true, y_pred, y_pro):
    c_mat = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = list(c_mat.flatten())
    se = tp/(tp+fn)
    sp = tn/(tn+fp)
    acc = (tp+tn)/(tn+fp+fn+tp)
    mcc = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)+1e-8)
    auc_prc = auc(precision_recall_curve(y_true, y_pro, pos_label=1)[1],
                  precision_recall_curve(y_true, y_pro, pos_label=1)[0])
    auc_roc = roc_auc_score(y_true, y_pro)
    return tn, fp, fn, tp, se, sp, acc, mcc, auc_prc, auc_roc

class Meter(object):
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
        print(f'y_true.shape: {y_true.shape}, after detach: {y_true.detach().cpu().shape}')
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
            precision, recall, _thresholds = precision_recall_curve(task_y_true, task_y_pred, pos_label=1)
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
            scores.append(F.l1_loss(task_y_true, task_y_pred, reduction=reduction).item())
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

    def compute_metric(self, metric_name, reduction='mean'):
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
        assert metric_name in ['roc_auc', 'l1', 'rmse', 'prc_auc', 'mae', 'r2'], \
            'Expect metric name to be "roc_auc", "l1", "rmse", "prc_auc", "mae", "r2" got {}'.format(metric_name)  # assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常
        assert reduction in ['mean', 'sum']
        if metric_name == 'roc_auc':
            return self.roc_auc_score()
        if metric_name == 'l1':
            return self.l1_loss(reduction)
        if metric_name == 'rmse':
            return self.rmse()
        if metric_name == 'prc_auc':
            return self.roc_precision_recall_score()
        if metric_name == 'mae':
            return self.mae()
        if metric_name == 'r2':
            return self.r2()
        if metric_name == 'mcc':
            return self.mcc()

class MyDataset(object):
    def __init__(self, Xs, Ys):
        self.Xs = paddle.to_tensor(Xs, dtype=torch.float32)
        self.masks = paddle.to_tensor(~np.isnan(Ys) * 1.0, dtype=torch.float32)
        # convert np.nan to 0
        self.Ys = paddle.to_tensor(np.nan_to_num(Ys), dtype=torch.float32)


    def __len__(self):
        return len(self.Ys)

    def __getitem__(self, idx):

        X = self.Xs[idx]
        Y = self.Ys[idx]
        mask = self.masks[idx]

        return X, Y, mask

class EarlyStopping(object):

    def __init__(self, mode='higher', patience=10, filename=None): 
        if filename is None:
            dt = datetime.datetime.now()
            filename = '{}_early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
                dt.date(), dt.hour, dt.minute, dt.second)

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher  
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        return (score > prev_best_score)

    def _check_lower(self, score, prev_best_score):
        return (score < prev_best_score)

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):  # 当前模型如果是更优模型，则保存当前模型
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            # print(
            #     f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)

    def load_checkpoint(self, model):
        '''Load model saved with early stopping.'''
        model.set_state_dict(torch.load(self.filename)['model_state_dict'])

def collate_fn(data_batch):
    Xs, Ys, masks = map(list, zip(*data_batch))

    Xs = torch.stack(Xs, axis=0)
    Ys = torch.stack(Ys, axis=0)
    masks = torch.stack(masks, axis=0)

    return Xs, Ys, masks

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子