import paddle
from ignite.metrics.confusion_matrix import ConfusionMatrix
from ignite.metrics.metrics_lambda import MetricsLambda

EPSILON = 1e-15


def DiceCoefficient(cm: ConfusionMatrix) -> MetricsLambda:
    """Calculates Dice similarity coefficient from ``ignite.metrics.ConfusionMatrix``.

    Args:
        cm: Instance of confusion matrix metric, where
            ``cm[0, 0]=TN``, ``cm[0, 1]=FP``,
            ``cm[1, 0]=FN``, ``cm[1, 1]=TP``.

    Returns:
        MetricsLambda instance computing the Dice coefficient.

    """
    cm = cm.astype(paddle.float64)
    return (2 * cm[1, 1] + EPSILON) / (2 * cm[1, 1] + cm[0, 1] + cm[1, 0] + EPSILON)


def FalseNegativeRate(cm: ConfusionMatrix) -> MetricsLambda:
    """Calculates miss rate from ``ignite.metrics.ConfusionMatrix``.

    Args:
        cm: Instance of confusion matrix metric, where
            ``cm[0, 0]=TN``, ``cm[0, 1]=FP``,
            ``cm[1, 0]=FN``, ``cm[1, 1]=TP``.

    Returns:
        MetricsLambda instance computing the true negative rate.

    """
    cm = cm.astype(paddle.float64)
    return cm[1, 0] / (cm[1, 0] + cm[1, 1] + EPSILON)


def TrueNegativeRate(cm: ConfusionMatrix) -> MetricsLambda:
    """Calculates specificity from ``ignite.metrics.ConfusionMatrix``.

    Args:
        cm: Instance of confusion matrix metric, where
            ``cm[0, 0]=TN``, ``cm[0, 1]=FP``,
            ``cm[1, 0]=FN``, ``cm[1, 1]=TP``.

    Returns:
        MetricsLambda instance computing the true negative rate.

    """
    cm = cm.astype(paddle.float64)
    return cm[0, 0] / (cm[0, 0] + cm[0, 1] + EPSILON)


def FalsePositiveRate(cm: ConfusionMatrix) -> MetricsLambda:
    """Calculates fall-out from ``ignite.metrics.ConfusionMatrix``.

    Args:
        cm: Instance of confusion matrix metric, where
            ``cm[0, 0]=TN``, ``cm[0, 1]=FP``,
            ``cm[1, 0]=FN``, ``cm[1, 1]=TP``.

    Returns:
        MetricsLambda instance computing the true positive rate.

    """
    cm = cm.astype(paddle.float64)
    return cm[0, 1] / (cm[0, 1] + cm[0, 0] + EPSILON)


def TruePositiveRate(cm: ConfusionMatrix) -> MetricsLambda:
    """Calculates sensitivity from ``ignite.metrics.ConfusionMatrix``.

    Args:
        cm: Instance of confusion matrix metric, where
            ``cm[0, 0]=TN``, ``cm[0, 1]=FP``,
            ``cm[1, 0]=FN``, ``cm[1, 1]=TP``.

    Returns:
        MetricsLambda instance computing the true positive rate.

    """
    cm = cm.astype(paddle.float64)
    return cm[1, 1] / (cm[1, 0] + cm[1, 1] + EPSILON)


def PositivePredictiveValue(cm: ConfusionMatrix) -> MetricsLambda:
    """Calculates precision from ``ignite.metrics.ConfusionMatrix``.

    Args:
        cm: Instance of confusion matrix metric, where
            ``cm[0, 0]=TN``, ``cm[0, 1]=FP``,
            ``cm[1, 0]=FN``, ``cm[1, 1]=TP``.

    Returns:
        MetricsLambda instance computing the positive predictive value.

    """
    cm = cm.astype(paddle.float64)
    return cm[1, 1] / (cm[0, 1] + cm[1, 1] + EPSILON)


def NegativePredictiveValue(cm: ConfusionMatrix) -> MetricsLambda:
    """Calculates negative predictive value from ``ignite.metrics.ConfusionMatrix``.

    Args:
        cm: Instance of confusion matrix metric, where
            ``cm[0, 0]=TN``, ``cm[0, 1]=FP``,
            ``cm[1, 0]=FN``, ``cm[1, 1]=TP``.

    Returns:
        MetricsLambda instance computing the negative predictive value.

    """
    cm = cm.astype(paddle.float64)
    return cm[0, 0] / (cm[0, 0] + cm[1, 0] + EPSILON)


Precision = PositivePredictiveValue
Recall = TruePositiveRate
Sensitivity = TruePositiveRate
Specificity = TrueNegativeRate
