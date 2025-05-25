import paddle
import paddle.nn.functional as F
import paddle_geometric.graphgym.register as register
from paddle_geometric.graphgym.config import cfg


def compute_loss(pred, true):
    """Compute loss and prediction score.

    Args:
        pred (paddle.Tensor): Unnormalized prediction
        true (paddle.Tensor): Ground truth

    Returns: Loss, normalized prediction score
    """
    bce_loss = paddle.nn.BCEWithLogitsLoss(reduction=cfg.model.size_average)
    mse_loss = paddle.nn.MSELoss(reduction=cfg.model.size_average)

    # default manipulation for pred and true
    pred = paddle.squeeze(pred, axis=-1) if pred.ndim > 1 else pred
    true = paddle.squeeze(true, axis=-1) if true.ndim > 1 else true

    # Try to load customized loss
    for func in register.loss_dict.values():
        value = func(pred, true)
        if value is not None:
            return value

    if cfg.model.loss_fun == 'cross_entropy':
        # multiclass
        if pred.ndim > 1 and true.ndim == 1:
            pred = F.log_softmax(pred, axis=-1)
            return F.nll_loss(pred, true), pred
        # binary or multilabel
        else:
            true = true.astype('float32')
            return bce_loss(pred, true), F.sigmoid(pred)
    elif cfg.model.loss_fun == 'mse':
        true = true.astype('float32')
        return mse_loss(pred, true), pred
    else:
        raise ValueError(f"Loss function '{cfg.model.loss_fun}' not supported")
