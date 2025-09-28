from typing import Callable

import numpy as np
import paddle

from ppsci.arch.chemprop_molecule_utils import TrainArgs


def split(x, num_or_sections, axis=0):
    if isinstance(num_or_sections, int):
        return paddle.split(x, x.shape[axis] // num_or_sections, axis)
    else:
        return paddle.split(x, num_or_sections, axis)


def reshape(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return paddle.reshape(self, args[0])
        else:
            return paddle.reshape(self, list(args))
    elif kwargs:
        assert "shape" in kwargs
        return paddle.reshape(self, shape=kwargs["shape"])


setattr(paddle.Tensor, "reshape", reshape)


def get_loss_func(args: TrainArgs) -> Callable:
    """
    Gets the loss function corresponding to a given dataset type.

    :param args: Arguments containing the dataset type ("classification", "regression", or "multiclass").
    :return: A loss function.
    """
    supported_loss_functions = {
        "regression": {
            "mse": paddle.nn.MSELoss(reduction="none"),
            "bounded_mse": bounded_mse_loss,
            "mve": normal_mve,
            "evidential": evidential_loss,
        },
        "classification": {
            "binary_cross_entropy": paddle.nn.BCEWithLogitsLoss(reduction="none"),
            "mcc": mcc_class_loss,
            "dirichlet": dirichlet_class_loss,
        },
        "multiclass": {
            "cross_entropy": paddle.nn.CrossEntropyLoss(reduction="none"),
            "mcc": mcc_multiclass_loss,
            "dirichlet": dirichlet_multiclass_loss,
        },
        "spectra": {"sid": sid_loss, "wasserstein": wasserstein_loss},
    }
    if args.dataset_type not in supported_loss_functions.keys():
        raise ValueError(f'Dataset type "{args.dataset_type}" not supported.')
    loss_function_choices = supported_loss_functions.get(args.dataset_type, dict())
    loss_function = loss_function_choices.get(args.loss_function)
    if loss_function is not None:
        return loss_function
    else:
        raise ValueError(
            f'Loss function "{args.loss_function}" not supported with dataset type {args.dataset_type}.             Available options for that dataset type are {loss_function_choices.keys()}.'
        )


def bounded_mse_loss(
    predictions: paddle.to_tensor,
    targets: paddle.to_tensor,
    less_than_target: paddle.to_tensor,
    greater_than_target: paddle.to_tensor,
) -> paddle.to_tensor:
    """
    Loss function for use with regression when some targets are presented as inequalities.

    :param predictions: Model predictions with shape(batch_size, tasks).
    :param targets: Target values with shape(batch_size, tasks).
    :param less_than_target: A tensor with boolean values indicating whether the target is a less-than inequality.
    :param greater_than_target: A tensor with boolean values indicating whether the target is a greater-than inequality.
    :return: A tensor containing loss values of shape(batch_size, tasks).
    """
    predictions = paddle.where(
        condition=paddle.logical_and(x=predictions < targets, y=less_than_target),
        x=targets,
        y=predictions,
    )
    predictions = paddle.where(
        condition=paddle.logical_and(x=predictions > targets, y=greater_than_target),
        x=targets,
        y=predictions,
    )
    return paddle.nn.functional.mse_loss(
        input=predictions, label=targets, reduction="none"
    )


def mcc_class_loss(
    predictions: paddle.to_tensor,
    targets: paddle.to_tensor,
    data_weights: paddle.to_tensor,
    mask: paddle.to_tensor,
) -> paddle.to_tensor:
    """
    A classification loss using a soft version of the Matthews Correlation Coefficient.

    :param predictions: Model predictions with shape(batch_size, tasks).
    :param targets: Target values with shape(batch_size, tasks).
    :param data_weights: A tensor with float values indicating how heavily to weight each datapoint in training with shape(batch_size, 1)
    :param mask: A tensor with boolean values indicating whether the loss for this prediction is considered in the gradient descent with shape(batch_size, tasks).
    :return: A tensor containing loss values of shape(tasks).
    """
    TP = paddle.sum(targets * predictions * data_weights * mask, axis=0)
    FP = paddle.sum((1 - targets) * predictions * data_weights * mask, axis=0)
    FN = paddle.sum(targets * (1 - predictions) * data_weights * mask, axis=0)
    TN = paddle.sum((1 - targets) * (1 - predictions) * data_weights * mask, axis=0)
    loss = 1 - (TP * TN - FP * FN) / paddle.sqrt(
        x=(TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    )
    return loss


def mcc_multiclass_loss(
    predictions: paddle.to_tensor,
    targets: paddle.to_tensor,
    data_weights: paddle.to_tensor,
    mask: paddle.to_tensor,
) -> paddle.to_tensor:
    """
    A multiclass loss using a soft version of the Matthews Correlation Coefficient. Multiclass definition follows the version in sklearn documentation.

    :param predictions: Model predictions with shape(batch_size, classes).
    :param targets: Target values with shape(batch_size).
    :param data_weights: A tensor with float values indicating how heavily to weight each datapoint in training with shape(batch_size, 1)
    :param mask: A tensor with boolean values indicating whether the loss for this prediction is considered in the gradient descent with shape(batch_size).
    :return: A tensor value for the loss.
    """
    mask = mask.unsqueeze(axis=1)
    bin_targets = paddle.zeros_like(x=predictions)
    bin_targets[paddle.arange(end=tuple(predictions.shape)[0]), targets] = 1
    pred_classes = predictions.argmax(axis=1)
    bin_preds = paddle.zeros_like(x=predictions)
    bin_preds[paddle.arange(end=tuple(predictions.shape)[0]), pred_classes] = 1
    masked_data_weights = data_weights * mask
    t_sum = paddle.sum(bin_targets * masked_data_weights, axis=0)
    p_sum = paddle.sum(bin_preds * masked_data_weights, axis=0)
    n_correct = paddle.sum(x=bin_preds * bin_targets * masked_data_weights)
    n_samples = paddle.sum(x=predictions * masked_data_weights)
    cov_ytyp = n_correct * n_samples - paddle.dot(x=p_sum, y=t_sum)
    cov_ypyp = n_samples**2 - paddle.dot(x=p_sum, y=p_sum)
    cov_ytyt = n_samples**2 - paddle.dot(x=t_sum, y=t_sum)
    if cov_ypyp * cov_ytyt == 0:
        loss = paddle.to_tensor(data=0.0)
    else:
        loss = cov_ytyp / paddle.sqrt(x=cov_ytyt * cov_ypyp)
    return loss


def sid_loss(
    model_spectra: paddle.to_tensor,
    target_spectra: paddle.to_tensor,
    mask: paddle.to_tensor,
    threshold: float = None,
) -> paddle.to_tensor:
    """
    Loss function for use with spectra data type.

    :param model_spectra: The predicted spectra output from a model with shape (batch_size,spectrum_length).
    :param target_spectra: The target spectra with shape (batch_size,spectrum_length). Values must be normalized so that each spectrum sums to 1.
    :param mask: Tensor with boolean indications of where the spectrum output should not be excluded with shape (batch_size,spectrum_length).
    :param threshold: Loss function requires that values are positive and nonzero. Values below the threshold will be replaced with the threshold value.
    :return: A tensor containing loss values for the batch with shape (batch_size,spectrum_length).
    """
    zero_sub = paddle.zeros_like(x=model_spectra)
    one_sub = paddle.ones_like(x=model_spectra)
    if threshold is not None:
        threshold_sub = paddle.full(
            shape=tuple(model_spectra.shape), fill_value=threshold
        )
        model_spectra = paddle.where(
            condition=model_spectra < threshold, x=threshold_sub, y=model_spectra
        )
    model_spectra = paddle.where(condition=mask, x=model_spectra, y=zero_sub)
    sum_model_spectra = paddle.sum(model_spectra, axis=1, keepdim=True)
    model_spectra = paddle.divide(
        x=model_spectra, y=paddle.to_tensor(sum_model_spectra)
    )
    target_spectra = paddle.where(condition=mask, x=target_spectra, y=one_sub)
    model_spectra = paddle.where(condition=mask, x=model_spectra, y=one_sub)
    loss = paddle.multiply(
        x=paddle.log(
            x=paddle.divide(x=model_spectra, y=paddle.to_tensor(target_spectra))
        ),
        y=paddle.to_tensor(model_spectra),
    ) + paddle.multiply(
        x=paddle.log(
            x=paddle.divide(x=target_spectra, y=paddle.to_tensor(model_spectra))
        ),
        y=paddle.to_tensor(target_spectra),
    )
    return loss


def wasserstein_loss(
    model_spectra: paddle.to_tensor,
    target_spectra: paddle.to_tensor,
    mask: paddle.to_tensor,
    threshold: float = None,
) -> paddle.to_tensor:
    """
    Loss function for use with spectra data type. This loss assumes that values are evenly spaced.

    :param model_spectra: The predicted spectra output from a model with shape (batch_size,spectrum_length).
    :param target_spectra: The target spectra with shape (batch_size,spectrum_length). Values must be normalized so that each spectrum sums to 1.
    :param mask: Tensor with boolian indications of where the spectrum output should not be excluded with shape (batch_size,spectrum_length).
    :param threshold: Loss function requires that values are positive and nonzero. Values below the threshold will be replaced with the threshold value.
    :return: A tensor containing loss values for the batch with shape (batch_size,spectrum_length).
    """
    zero_sub = paddle.zeros_like(x=model_spectra)
    if threshold is not None:
        threshold_sub = paddle.full(
            shape=tuple(model_spectra.shape), fill_value=threshold
        )
        model_spectra = paddle.where(
            condition=model_spectra < threshold, x=threshold_sub, y=model_spectra
        )
    model_spectra = paddle.where(condition=mask, x=model_spectra, y=zero_sub)
    sum_model_spectra = paddle.sum(model_spectra, axis=1, keepdim=True)
    model_spectra = paddle.divide(
        x=model_spectra, y=paddle.to_tensor(sum_model_spectra)
    )
    target_cum = paddle.cumsum(target_spectra, axis=1)
    model_cum = paddle.cumsum(model_spectra, axis=1)
    loss = paddle.abs(x=target_cum - model_cum)
    return loss


def normal_mve(pred_values, targets):
    """
    Use the negative log likelihood function of a normal distribution as a loss function used for making
    simultaneous predictions of the mean and error distribution variance simultaneously.

    :param pred_values: Combined predictions of means and variances of shape(data, tasks*2).
                        Means are first in dimension 1, followed by variances.
    :return: A tensor loss value.
    """
    pred_means, pred_var = split(
        x=pred_values, num_or_sections=tuple(pred_values.shape)[1] // 2, axis=1
    )
    return paddle.log(x=2 * np.pi * pred_var) / 2 + (pred_means - targets) ** 2 / (
        2 * pred_var
    )


def dirichlet_class_loss(alphas, target_labels, lam=0):
    """
    Use Evidential Learning Dirichlet loss from Sensoy et al in classification datasets.
    :param alphas: Predicted parameters for Dirichlet in shape(datapoints, tasks*2).
                   Negative class first then positive class in dimension 1.
    :param target_labels: Digital labels to predict in shape(datapoints, tasks).
    :lambda: coefficient to weight KL term

    :return: Loss
    """
    num_tasks = tuple(target_labels.shape)[1]
    num_classes = 2
    alphas = paddle.reshape(
        x=alphas, shape=(tuple(alphas.shape)[0], num_tasks, num_classes)
    )
    y_one_hot = paddle.eye(num_rows=num_classes)[target_labels.astype(dtype="int64")]
    return dirichlet_common_loss(alphas=alphas, y_one_hot=y_one_hot, lam=lam)


def dirichlet_multiclass_loss(alphas, target_labels, lam=0):
    """
    Use Evidential Learning Dirichlet loss from Sensoy et al for multiclass datasets.
    :param alphas: Predicted parameters for Dirichlet in shape(datapoints, task, classes).
    :param target_labels: Digital labels to predict in shape(datapoints, tasks).
    :lambda: coefficient to weight KL term

    :return: Loss
    """
    num_classes = tuple(alphas.shape)[2]
    y_one_hot = paddle.eye(num_rows=num_classes)[target_labels.astype(dtype="int64")]
    return dirichlet_common_loss(alphas=alphas, y_one_hot=y_one_hot, lam=lam)


def dirichlet_common_loss(alphas, y_one_hot, lam=0):
    """
    Use Evidential Learning Dirichlet loss from Sensoy et al. This function follows
    after the classification and multiclass specific functions that reshape the
    alpha inputs and create one-hot targets.

    :param alphas: Predicted parameters for Dirichlet in shape(datapoints, task, classes).
    :param y_one_hot: Digital labels to predict in shape(datapoints, tasks, classes).
    :lambda: coefficient to weight KL term

    :return: Loss
    """
    S = paddle.sum(x=alphas, axis=-1, keepdim=True)
    p = alphas / S
    A = paddle.sum(x=(y_one_hot - p) ** 2, axis=-1, keepdim=True)
    B = paddle.sum(x=p * (1 - p) / (S + 1), axis=-1, keepdim=True)
    SOS = A + B
    alpha_hat = y_one_hot + (1 - y_one_hot) * alphas
    beta = paddle.ones_like(x=alpha_hat)
    S_alpha = paddle.sum(x=alpha_hat, axis=-1, keepdim=True)
    S_beta = paddle.sum(x=beta, axis=-1, keepdim=True)
    ln_alpha = paddle.lgamma(x=S_alpha) - paddle.sum(
        x=paddle.lgamma(x=alpha_hat), axis=-1, keepdim=True
    )
    ln_beta = paddle.sum(
        x=paddle.lgamma(x=beta), axis=-1, keepdim=True
    ) - paddle.lgamma(x=S_beta)
    dg_alpha = paddle.digamma(x=alpha_hat)
    dg_S_alpha = paddle.digamma(x=S_alpha)
    KL = (
        ln_alpha
        + ln_beta
        + paddle.sum(
            x=(alpha_hat - beta) * (dg_alpha - dg_S_alpha), axis=-1, keepdim=True
        )
    )
    KL = lam * KL
    loss = SOS + KL
    loss = paddle.mean(x=loss, axis=-1)
    return loss


def evidential_loss(pred_values, targets, lam=0, epsilon=1e-08):
    """
    Use Deep Evidential Regression negative log likelihood loss + evidential
        regularizer

    :param pred_values: Combined prediction values for mu, v, alpha, and beta parameters in shape(data, tasks*4).
                        Order in dimension 1 is mu, v, alpha, beta.
    :mu: pred mean parameter for NIG
    :v: pred lam parameter for NIG
    :alpha: predicted parameter for NIG
    :beta: Predicted parmaeter for NIG
    :targets: Outputs to predict

    :return: Loss
    """
    mu, v, alpha, beta = split(
        x=pred_values, num_or_sections=tuple(pred_values.shape)[1] // 4, axis=1
    )
    twoBlambda = 2 * beta * (1 + v)
    nll = (
        0.5 * paddle.log(x=np.pi / v)
        - alpha * paddle.log(x=twoBlambda)
        + (alpha + 0.5) * paddle.log(x=v * (targets - mu) ** 2 + twoBlambda)
        + paddle.lgamma(x=alpha)
        - paddle.lgamma(x=alpha + 0.5)
    )
    L_NLL = nll
    error = paddle.abs(x=targets - mu)
    reg = error * (2 * v + alpha)
    L_REG = reg
    loss = L_NLL + lam * (L_REG - epsilon)
    return loss
