r"""Output conversion modules."""

from typing import Mapping
from typing import Sequence
from typing import Union

import paddle
from deepali.core.nnutils import as_immutable_container


class ToImmutableOutput(paddle.nn.Layer):
    r"""Convert input to immutable output container.

    For use with ``paddle.utils.tensorboard.SummaryWriter.add_graph`` when model output is list or dict.
    See error message: "Encountering a dict at the output of the tracer might cause the trace to be incorrect,
    this is only valid if the container structure does not change based on the module's inputs. Consider using
    a constant container instead (e.g. for `list`, use a `tuple` instead. for `dict`, use a `NamedTuple` instead).
    If you absolutely need this and know the side effects, pass strict=False to trace() to allow this behavior."

    """

    def __init__(self, recursive: bool = True) -> None:
        super().__init__()
        self.recursive = recursive

    def forward(
        self, input: Union[paddle.Tensor, Sequence, Mapping]
    ) -> Union[paddle.Tensor, Sequence, Mapping]:
        return as_immutable_container(input, recursive=self.recursive)

    def extra_repr(self) -> str:
        return f"recursive={self.recursive!r}"
