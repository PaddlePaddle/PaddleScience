from typing import Dict
from typing import Optional
from typing import Sequence

import numpy as np
import paddle
from paddle.nn import functional as F

from ppsci.data.dataset import sevir_dataset


def _threshold(target, pred, T):
    """
    Returns binary tensors t,p the same shape as target & pred.  t = 1 wherever
    target > t.  p =1 wherever pred > t.  p and t are set to 0 wherever EITHER
    t or p are nan.
    This is useful for counts that don't involve correct rejections.

    Args:
        target (paddle.Tensor): label
        pred (paddle.Tensor): predict
        T (numeric_type): threshold
    Returns:
        t
        p
    """

    t = (target >= T).astype("float32")
    p = (pred >= T).astype("float32")
    is_nan = paddle.logical_or(paddle.isnan(target), paddle.isnan(pred))
    t[is_nan] = 0
    p[is_nan] = 0
    return t, p


class SEVIRSkillScore:
    r"""
    The calculation of skill scores in SEVIR challenge is slightly different:
        `mCSI = sum(mCSI_t) / T`
    See https://github.com/MIT-AI-Accelerator/sevir_challenges/blob/dev/radar_nowcasting/RadarNowcastBenchmarks.ipynb for more details.

    Args:
        seq_len (int): sequence length
        layout (str): layout mode
        mode (str): Should be in ("0", "1", "2")
            "0":
                cumulates hits/misses/fas of all test pixels
                score_avg takes average over all thresholds
                return
                    score_thresh shape = (1, )
                    score_avg shape = (1, )
            "1":
                cumulates hits/misses/fas of each step
                score_avg takes average over all thresholds while keeps the seq_len dim
                return
                    score_thresh shape = (seq_len, )
                    score_avg shape = (seq_len, )
            "2":
                cumulates hits/misses/fas of each step
                score_avg takes average over all thresholds, then takes average over the seq_len dim
                return
                    score_thresh shape = (1, )
                    score_avg shape = (1, )
        preprocess_type (str): prepprocess type
        threshold_list (Sequence[int]): threshold list
    """

    full_state_update: bool = True

    def __init__(
        self,
        layout: str = "NHWT",
        mode: str = "0",
        seq_len: Optional[int] = None,
        preprocess_type: str = "sevir",
        threshold_list: Sequence[int] = (16, 74, 133, 160, 181, 219),
        metrics_list: Sequence[str] = ("csi", "bias", "sucr", "pod"),
        eps: float = 1e-4,
        dist_sync_on_step: bool = False,
    ):
        super().__init__()
        self.layout = layout
        self.preprocess_type = preprocess_type
        self.threshold_list = threshold_list
        self.metrics_list = metrics_list
        self.eps = eps
        self.mode = mode
        self.seq_len = seq_len

        self.hits = paddle.zeros(shape=[len(self.threshold_list)])
        self.misses = paddle.zeros(shape=[len(self.threshold_list)])
        self.fas = paddle.zeros(shape=[len(self.threshold_list)])

        if mode in ("0",):
            self.keep_seq_len_dim = False
        elif mode in ("1", "2"):
            self.keep_seq_len_dim = True
            assert isinstance(
                self.seq_len, int
            ), "seq_len must be provided when we need to keep seq_len dim."

        else:
            raise NotImplementedError(f"mode {mode} not supported!")

    @staticmethod
    def pod(hits, misses, fas, eps):
        return hits / (hits + misses + eps)

    @staticmethod
    def sucr(hits, misses, fas, eps):
        return hits / (hits + fas + eps)

    @staticmethod
    def csi(hits, misses, fas, eps):
        return hits / (hits + misses + fas + eps)

    @staticmethod
    def bias(hits, misses, fas, eps):
        bias = (hits + fas) / (hits + misses + eps)
        logbias = paddle.pow(bias / paddle.log(paddle.full([], 2.0)), 2.0)
        return logbias

    @property
    def hits_misses_fas_reduce_dims(self):
        if not hasattr(self, "_hits_misses_fas_reduce_dims"):
            seq_dim = self.layout.find("T")
            self._hits_misses_fas_reduce_dims = list(range(len(self.layout)))
            if self.keep_seq_len_dim:
                self._hits_misses_fas_reduce_dims.pop(seq_dim)
        return self._hits_misses_fas_reduce_dims

    def calc_seq_hits_misses_fas(self, pred, target, threshold):
        """
        Args:
            pred (paddle.Tensor): Predict data.
            target (paddle.Tensor): True data.
            threshold (int):  The threshold to calculate hits, misses and fas.

        Returns:
            hits (paddle.Tensor): Number of hits.
            misses (paddle.Tensor): Number of misses.
            fas (paddle.Tensor): Number of false positives.
                each has shape (seq_len, )
        """

        with paddle.no_grad():
            t, p = _threshold(target, pred, threshold)
            hits = paddle.sum(t * p, axis=self.hits_misses_fas_reduce_dims).astype(
                "int32"
            )
            misses = paddle.sum(
                t * (1 - p), axis=self.hits_misses_fas_reduce_dims
            ).astype("int32")
            fas = paddle.sum((1 - t) * p, axis=self.hits_misses_fas_reduce_dims).astype(
                "int32"
            )
        return hits, misses, fas

    def preprocess(self, pred, target):
        if self.preprocess_type == "sevir":
            pred = sevir_dataset.SEVIRDataset.process_data_dict_back(
                data_dict={"vil": pred.detach().astype("float32")}
            )["vil"]
            target = sevir_dataset.SEVIRDataset.process_data_dict_back(
                data_dict={"vil": target.detach().astype("float32")}
            )["vil"]
        else:
            raise NotImplementedError(f"{self.preprocess_type} not supported")
        return pred, target

    def update(self, pred: paddle.Tensor, target: paddle.Tensor):
        pred, target = self.preprocess(pred, target)
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas = self.calc_seq_hits_misses_fas(pred, target, threshold)
            self.hits[i] += hits
            self.misses[i] += misses
            self.fas[i] += fas

    def compute(self, pred: paddle.Tensor, target: paddle.Tensor):
        metrics_dict = {
            "pod": self.pod,
            "csi": self.csi,
            "sucr": self.sucr,
            "bias": self.bias,
        }
        ret = {}
        for threshold in self.threshold_list:
            ret[threshold] = {}
        ret["avg"] = {}
        for metrics in self.metrics_list:
            if self.keep_seq_len_dim:
                score_avg = np.zeros((self.seq_len,))
            else:
                score_avg = 0
            # shape = (len(threshold_list), seq_len) if self.keep_seq_len_dim,
            # else shape = (len(threshold_list),)
            scores = metrics_dict[metrics](self.hits, self.misses, self.fas, self.eps)
            scores = scores.detach().cpu().numpy()
            for i, threshold in enumerate(self.threshold_list):
                if self.keep_seq_len_dim:
                    score = scores[i]  # shape = (seq_len, )
                else:
                    score = scores[i].item()  # shape = (1, )
                if self.mode in ("0", "1"):
                    ret[threshold][metrics] = score
                elif self.mode in ("2",):
                    ret[threshold][metrics] = np.mean(score).item()
                else:
                    raise NotImplementedError(f"{self.mode} is invalid.")
                score_avg += score
            score_avg /= len(self.threshold_list)
            if self.mode in ("0", "1"):
                ret["avg"][metrics] = score_avg
            elif self.mode in ("2",):
                ret["avg"][metrics] = np.mean(score_avg).item()
            else:
                raise NotImplementedError(f"{self.mode} is invalid.")

        metrics = {}
        metrics["csi_avg_loss"] = 0
        for metric in self.metrics_list:
            for th in self.threshold_list:
                metrics[f"{metric}_{th}"] = ret[th][metric]
            metrics[f"{metric}_avg"] = ret["avg"][metric]

        metrics["csi_avg_loss"] = -metrics["csi_avg"]
        return metrics


class eval_rmse_func:
    def __init__(
        self,
        out_len=12,
        layout="NTHWC",
        metrics_mode="0",
        metrics_list=["csi", "pod", "sucr", "bias"],
        threshold_list=[16, 74, 133, 160, 181, 219],
        *args,
    ) -> Dict[str, paddle.Tensor]:
        super().__init__()
        self.out_len = out_len
        self.layout = layout
        self.metrics_mode = metrics_mode
        self.metrics_list = metrics_list
        self.threshold_list = threshold_list

        self.sevir_score = SEVIRSkillScore(
            layout=self.layout,
            mode=self.metrics_mode,
            seq_len=self.out_len,
            threshold_list=self.threshold_list,
            metrics_list=self.metrics_list,
        )

    def __call__(
        self,
        output_dict: Dict[str, "paddle.Tensor"],
        label_dict: Dict[str, "paddle.Tensor"],
    ):
        pred = output_dict["vil"]
        vil_target = label_dict["vil"]
        vil_target = vil_target.reshape([-1, *vil_target.shape[2:]])
        # mse
        mae = F.l1_loss(pred, vil_target, "none")
        mae = mae.mean(axis=tuple(range(1, mae.ndim)))
        # mse
        mse = F.mse_loss(pred, vil_target, "none")
        mse = mse.mean(axis=tuple(range(1, mse.ndim)))

        return {"mse": mse, "mae": mae}


def train_mse_func(
    output_dict: Dict[str, "paddle.Tensor"],
    label_dict: Dict[str, "paddle.Tensor"],
    *args,
) -> paddle.Tensor:
    pred = output_dict["vil"]
    vil_target = label_dict["vil"]
    target = vil_target.reshape([-1, *vil_target.shape[2:]])
    return {"vil": F.mse_loss(pred, target)}
