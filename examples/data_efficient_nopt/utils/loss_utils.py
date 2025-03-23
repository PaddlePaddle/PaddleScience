"""
  loss functions
# """
# import logging
# import time

# import numpy as np
import paddle

# import torchvision


class LossMSE:
    """mse loss"""

    def __init__(self, params, model):
        self.params = params
        self.model = model

    def data(self, inputs, pred, target):
        if self.params.loss_style == "mean":
            loss = paddle.mean((target - pred) ** 2)
        elif self.params.loss_style == "sum":
            loss = paddle.sum((target - pred) ** 2) / pred.shape[0]
        return loss

    def bc(self, inputs, pred, targets):
        # currently no BC
        return paddle.to_tensor(0.0).astype(dtype=paddle.float32)

    def pde(self, inputs, pred, targets):
        # currently no PDE loss
        return paddle.to_tensor(0.0).astype(dtype=paddle.float32)
