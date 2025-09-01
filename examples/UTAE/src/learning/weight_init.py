"""
Weight initialization utilities (Paddle Version)
"""
import paddle.nn as nn


def weight_init(model):
    """
    Initialize model weights
    """
    for layer in model.sublayers():
        if isinstance(layer, (nn.Conv2D, nn.Conv1D)):
            nn.initializer.XavierUniform()(layer.weight)
            if layer.bias is not None:
                nn.initializer.Constant(0.0)(layer.bias)
        elif isinstance(layer, (nn.BatchNorm2D, nn.BatchNorm1D, nn.GroupNorm)):
            nn.initializer.Constant(1.0)(layer.weight)
            nn.initializer.Constant(0.0)(layer.bias)
        elif isinstance(layer, nn.Linear):
            nn.initializer.XavierUniform()(layer.weight)
            if layer.bias is not None:
                nn.initializer.Constant(0.0)(layer.bias)
