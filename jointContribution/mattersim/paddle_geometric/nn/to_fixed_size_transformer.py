from typing import Any

import paddle
from paddle import nn

from paddle_geometric.nn.fx import Transformer  # Assuming you have similar functionality for Paddle

try:
    from paddle.jit import Program, ProgramBlock, Node  # Adjust according to Paddle's equivalent functionality
except (ImportError, ModuleNotFoundError, AttributeError):
    Program, ProgramBlock, Node = 'Program', 'ProgramBlock', 'Node'


def to_fixed_size(module: nn.Layer, batch_size: int,
                  debug: bool = False) -> Program:
    r"""Converts a model and injects a pre-computed and fixed batch size to all
    global pooling operators.

    Args:
        module (paddle.nn.Layer): The model to transform.
        batch_size (int): The fixed batch size used in global pooling modules.
        debug (bool, optional): If set to :obj:`True`, will perform
            transformation in debug mode. (default: :obj:`False`)
    """
    transformer = ToFixedSizeTransformer(module, batch_size, debug)
    return transformer.transform()


class ToFixedSizeTransformer(Transformer):
    def __init__(self, module: nn.Layer, batch_size: int, debug: bool = False):
        super().__init__(module, debug=debug)
        self.batch_size = batch_size

    def call_global_pooling_module(self, node: Node, target: Any, name: str):
        kwargs = node.kwargs.copy()
        kwargs['dim_size'] = self.batch_size
        node.kwargs = kwargs
