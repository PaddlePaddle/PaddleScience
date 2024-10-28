r"""Metrics for multi-label classification performance evaluation."""
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Union

import paddle
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
from ignite.metrics.metric import reinit__is_reduced
from ignite.metrics.metric import sync_all_reduce


class MultiLabelScore(Metric):
    r"""Compute a score for each class label."""

    def __init__(
        self,
        score_fn: Callable,
        num_classes: int,
        output_transform: Callable = lambda x: x,
        device: Optional[Union[str, (paddle.CPUPlace, paddle.CUDAPlace, str)]] = None,
    ):
        self.score_fn = score_fn
        self.num_classes = num_classes
        self.accumulator = None
        self.num_examples = 0
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self.accumulator = paddle.zeros(shape=self.num_classes, dtype="float32")
        self.num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[paddle.Tensor]) -> None:
        y_pred, y = output
        if y_pred.ndim < 2:
            raise ValueError(
                f"MultiLabelScore.update() y_pred must have shape (N, C, ...), but given {tuple(y_pred.shape)}"
            )
        if tuple(y_pred.shape)[1] not in (1, self.num_classes):
            raise ValueError(
                f"MultiLabelScore.update() expected y_pred to have 1 or {self.num_channels} channels"
            )
        if y.ndim + 1 == y_pred.ndim:
            y = y.unsqueeze(axis=1)
        elif y.ndim != y_pred.ndim:
            raise ValueError(
                f"MultiLabelScore.update() y_pred must have shape (N, C, ...) and y must have shape (N, ...) or (N, 1, ...), but given {tuple(y.shape)} vs {tuple(y_pred.shape)}"
            )
        if tuple(y.shape) != (tuple(y_pred.shape)[0], 1) + tuple(y_pred.shape)[2:]:
            raise ValueError("y and y_pred must have compatible shapes.")
        scores = multilabel_score(self.score_fn, y_pred, y, num_classes=self.num_classes)
        self.accumulator += scores
        self.num_examples += tuple(y_pred.shape)[0]

    @sync_all_reduce("accumulator", "num_examples")
    def compute(self) -> float:
        if self.num_examples == 0:
            raise NotComputableError(
                "Loss must have at least one example before it can be computed."
            )
        return self.accumulator / self.num_examples

    @paddle.no_grad()
    def iteration_completed(self, engine: Engine) -> None:
        output = self._output_transform(engine.state.output)
        self.update(output)


def multilabel_score(
    score_fn, preds: paddle.Tensor, labels: paddle.Tensor, num_classes: Optional[int] = None
) -> paddle.Tensor:
    r"""Evaluate score for each class label."""
    assert tuple(labels.shape)[1] == 1
    if num_classes is None:
        num_classes = tuple(preds.shape)[1]
        if num_classes == 1:
            raise ValueError(
                "multilabel_score() 'num_classes' required when 'preds' is not one-hot encoded"
            )
    if tuple(preds.shape)[1] == num_classes:
        preds = preds.argmax(axis=1, keepdim=True)
    elif tuple(preds.shape)[1] != 1:
        raise ValueError("multilabel_score() 'preds' must have shape (N, C|1, ..., X)")
    result = paddle.zeros(shape=num_classes, dtype="float32")
    for label in range(num_classes):
        y_pred = preds.equal(y=label).astype(dtype="float32")
        y = labels.equal(y=label).astype(dtype="float32")
        result[label] = score_fn(y_pred, y).mean()
    return result
