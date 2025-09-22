from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import paddle.nn as nn

from ppsci.arch.mlp import MLP


class TADF(MLP):
    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        num_layers: int,
        hidden_size: Union[int, Tuple[int, ...]],
        activation: str = "tanh",
        skip_connection: bool = False,
        weight_norm: bool = False,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        periods: Optional[Dict[str, tuple]] = None,
        fourier: Optional[Dict[str, Union[float, int]]] = None,
        random_weight: Optional[Dict[str, float]] = None,
        dropout: Optional[float] = 0.5,
    ):

        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
            num_layers=num_layers,
            hidden_size=hidden_size,
            activation=activation,
            skip_connection=skip_connection,
            weight_norm=weight_norm,
            input_dim=input_dim,
            output_dim=output_dim,
            periods=periods,
            fourier=fourier,
            random_weight=random_weight,
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward_tensor(self, x):
        y = x
        skip = None
        for i, linear in enumerate(self.linears):
            y = linear(y)
            if self.skip_connection and i % 2 == 0:
                if skip is not None:
                    skip = y
                    y = y + skip
                else:
                    skip = y
            y = self.acts[i](y)
            y = self.dropout(y)
        y = self.last_fc(y)
        return y
