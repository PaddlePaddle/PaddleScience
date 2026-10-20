"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from __future__ import annotations

import math
from typing import Literal

import paddle

__all__ = ["LoRA", "LoRARollout", "LoRAMode"]
LoRAMode = Literal["single", "from_second", "all"]


class LoRA(paddle.nn.Layer):
    """LoRA adaptation for a linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 4,
        alpha: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert r > 0, "The rank must be strictly positive."
        self.lora_alpha = alpha
        self.r = r
        self.lora_dropout = paddle.nn.Dropout(dropout)
        self.lora_A = paddle.nn.Parameter(paddle.empty((r, in_features)))
        self.lora_B = paddle.nn.Parameter(paddle.empty((out_features, r)))
        self.scaling = self.lora_alpha / self.r
        self.init_weights()

    def init_weights(self) -> None:
        """Initialise weights."""
        paddle.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        paddle.nn.init.zeros_(self.lora_B)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Compute the LoRA adaptation.

        Args:
            x (paddle.Tensor): Input to the linear layer.

        Returns:
            paddle.Tensor: Additive correction for the output of the linear layer.
        """
        x = (
            self.lora_dropout(x)
            @ self.lora_A.transpose(0, 1)
            @ self.lora_B.transpose(0, 1)
        )
        return x * self.scaling


class LoRARollout(paddle.nn.Layer):
    """Per-roll-out-step LoRA finetuning."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 8,
        dropout: float = 0.0,
        max_steps: int = 40,
        mode: LoRAMode = "single",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.max_steps = max_steps
        lora_layers = max_steps if mode == "all" else 1
        self.loras = paddle.nn.LayerList(
            [
                LoRA(in_features, out_features, r=r, alpha=alpha, dropout=dropout)
                for _ in range(lora_layers)
            ]
        )

    def forward(self, x: paddle.Tensor, step: int) -> paddle.Tensor:
        """Compute the LoRA adaptation.

        Args:
            x (paddle.Tensor): Input to the linear layer.
            step (int): Roll-out step, starting at zero.

        Returns:
            paddle.Tensor: Additive correction for the output of the linear layer.
        """
        assert step >= 0, f"Step must be non-negative, found {step}."
        if step >= self.max_steps:
            return 0
        if self.mode == "single":
            return self.loras[0](x)
        elif self.mode == "from_second":
            if step == 0:
                return 0
            else:
                return self.loras[0](x)
        elif self.mode == "all":
            return self.loras[step](x)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
