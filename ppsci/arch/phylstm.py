# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn

from ppsci.arch import base


class DeepPhyLSTM(base.Arch):
    """DeepPhyLSTM init function.

    Args:
        input_size (int): The input size.
        output_size (int): The output size.
        hidden_size (int, optional): The hidden size. Defaults to 100.
        model_type (int, optional): The model type, value is 2 or 3, 2 indicates having two sub-models, 3 indicates having three submodels. Defaults to 2.

    Examples:
        >>> import paddle
        >>> import ppsci
        >>> # model_type is `2`
        >>> model = ppsci.arch.DeepPhyLSTM(
        ...     input_size=16,
        ...     output_size=1,
        ...     hidden_size=100,
        ...     model_type=2)
        >>> out = model(
        ...     {"ag":paddle.rand([64, 16, 16]),
        ...     "ag_c":paddle.rand([64, 16, 16]),
        ...     "phi":paddle.rand([1, 16, 16])})
        >>> for k, v in out.items():
        ...     print(f"{k} {v.dtype} {v.shape}")
        eta_pred paddle.float32 [64, 16, 1]
        eta_dot_pred paddle.float32 [64, 16, 1]
        g_pred paddle.float32 [64, 16, 1]
        eta_t_pred_c paddle.float32 [64, 16, 1]
        eta_dot_pred_c paddle.float32 [64, 16, 1]
        lift_pred_c paddle.float32 [64, 16, 1]
        >>> # model_type is `3`
        >>> model = ppsci.arch.DeepPhyLSTM(
        ...     input_size=16,
        ...     output_size=1,
        ...     hidden_size=100,
        ...     model_type=3)
        >>> out = model(
        ...     {"ag":paddle.rand([64, 16, 1]),
        ...     "ag_c":paddle.rand([64, 16, 1]),
        ...     "phi":paddle.rand([1, 16, 16])})
        >>> for k, v in out.items():
        ...     print(f"{k} {v.dtype} {v.shape}")
        eta_pred paddle.float32 [64, 16, 1]
        eta_dot_pred paddle.float32 [64, 16, 1]
        g_pred paddle.float32 [64, 16, 1]
        eta_t_pred_c paddle.float32 [64, 16, 1]
        eta_dot_pred_c paddle.float32 [64, 16, 1]
        lift_pred_c paddle.float32 [64, 16, 1]
        g_t_pred_c paddle.float32 [64, 16, 1]
        g_dot_pred_c paddle.float32 [64, 16, 1]
    """

    def __init__(self, input_size, output_size, hidden_size=100, model_type=2):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.model_type = model_type

        if self.model_type == 2:
            self.lstm_model = nn.Sequential(
                nn.LSTM(input_size, hidden_size),
                nn.ReLU(),
                nn.LSTM(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LSTM(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.Linear(hidden_size, 3 * output_size),
            )

            self.lstm_model_f = nn.Sequential(
                nn.LSTM(3 * output_size, hidden_size),
                nn.ReLU(),
                nn.LSTM(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LSTM(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.Linear(hidden_size, output_size),
            )
        elif self.model_type == 3:
            self.lstm_model = nn.Sequential(
                nn.LSTM(1, hidden_size),
                nn.ReLU(),
                nn.LSTM(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LSTM(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 3 * output_size),
            )

            self.lstm_model_f = nn.Sequential(
                nn.LSTM(3 * output_size, hidden_size),
                nn.ReLU(),
                nn.LSTM(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LSTM(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
            )

            self.lstm_model_g = nn.Sequential(
                nn.LSTM(2 * output_size, hidden_size),
                nn.ReLU(),
                nn.LSTM(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LSTM(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
            )
        else:
            raise ValueError(f"model_type should be 2 or 3, but got {model_type}")

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)

        if self.model_type == 2:
            result_dict = self._forward_type_2(x)
        elif self.model_type == 3:
            result_dict = self._forward_type_3(x)
        if self._output_transform is not None:
            result_dict = self._output_transform(x, result_dict)
        return result_dict

    def _forward_type_2(self, x):
        output = self.lstm_model(x["ag"])
        eta_pred = output[:, :, 0 : self.output_size]
        eta_dot_pred = output[:, :, self.output_size : 2 * self.output_size]
        g_pred = output[:, :, 2 * self.output_size :]

        # for ag_c
        output_c = self.lstm_model(x["ag_c"])
        eta_pred_c = output_c[:, :, 0 : self.output_size]
        eta_dot_pred_c = output_c[:, :, self.output_size : 2 * self.output_size]
        g_pred_c = output_c[:, :, 2 * self.output_size :]
        eta_t_pred_c = paddle.matmul(x["phi"], eta_pred_c)
        eta_tt_pred_c = paddle.matmul(x["phi"], eta_dot_pred_c)
        eta_dot1_pred_c = eta_dot_pred_c[:, :, 0:1]
        tmp = paddle.concat([eta_pred_c, eta_dot1_pred_c, g_pred_c], 2)
        f = self.lstm_model_f(tmp)
        lift_pred_c = eta_tt_pred_c + f

        return {
            "eta_pred": eta_pred,
            "eta_dot_pred": eta_dot_pred,
            "g_pred": g_pred,
            "eta_t_pred_c": eta_t_pred_c,
            "eta_dot_pred_c": eta_dot_pred_c,
            "lift_pred_c": lift_pred_c,
        }

    def _forward_type_3(self, x):
        # physics informed neural networks
        output = self.lstm_model(x["ag"])
        eta_pred = output[:, :, 0 : self.output_size]
        eta_dot_pred = output[:, :, self.output_size : 2 * self.output_size]
        g_pred = output[:, :, 2 * self.output_size :]

        output_c = self.lstm_model(x["ag_c"])
        eta_pred_c = output_c[:, :, 0 : self.output_size]
        eta_dot_pred_c = output_c[:, :, self.output_size : 2 * self.output_size]
        g_pred_c = output_c[:, :, 2 * self.output_size :]

        eta_t_pred_c = paddle.matmul(x["phi"], eta_pred_c)
        eta_tt_pred_c = paddle.matmul(x["phi"], eta_dot_pred_c)
        g_t_pred_c = paddle.matmul(x["phi"], g_pred_c)

        f = self.lstm_model_f(paddle.concat([eta_pred_c, eta_dot_pred_c, g_pred_c], 2))
        lift_pred_c = eta_tt_pred_c + f

        eta_dot1_pred_c = eta_dot_pred_c[:, :, 0:1]
        g_dot_pred_c = self.lstm_model_g(paddle.concat([eta_dot1_pred_c, g_pred_c], 2))

        return {
            "eta_pred": eta_pred,
            "eta_dot_pred": eta_dot_pred,
            "g_pred": g_pred,
            "eta_t_pred_c": eta_t_pred_c,
            "eta_dot_pred_c": eta_dot_pred_c,
            "lift_pred_c": lift_pred_c,
            "g_t_pred_c": g_t_pred_c,
            "g_dot_pred_c": g_dot_pred_c,
        }
