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

from typing import Dict
from typing import List
from typing import Tuple

import paddle
import paddle.nn as nn

from ppsci.arch import activation as act_mod
from ppsci.arch import base


class Generator(base.Arch):
    """Generator Net of GAN. Attention, the net using a kind of variant of ResBlock which is
        unique to "tempoGAN" example but not an open source network.

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("input1", "input2").
        output_keys (Tuple[str, ...]): Name of output keys, such as ("output1", "output2").
        in_channel (int): Number of input channels of the first conv layer.
        out_channels_tuple (Tuple[Tuple[int, ...], ...]): Number of output channels of all conv layers,
            such as [[out_res0_conv0, out_res0_conv1], [out_res1_conv0, out_res1_conv1]]
        kernel_sizes_tuple (Tuple[Tuple[int, ...], ...]): Number of kernel_size of all conv layers,
            such as [[kernel_size_res0_conv0, kernel_size_res0_conv1], [kernel_size_res1_conv0, kernel_size_res1_conv1]]
        strides_tuple (Tuple[Tuple[int, ...], ...]): Number of stride of all conv layers,
            such as [[stride_res0_conv0, stride_res0_conv1], [stride_res1_conv0, stride_res1_conv1]]
        use_bns_tuple (Tuple[Tuple[bool, ...], ...]): Whether to use the batch_norm layer after each conv layer.
        acts_tuple (Tuple[Tuple[str, ...], ...]): Whether to use the activation layer after each conv layer. If so, witch activation to use,
            such as [[act_res0_conv0, act_res0_conv1], [act_res1_conv0, act_res1_conv1]]

    Examples:
        >>> import ppsci
        >>> in_channel = 1
        >>> rb_channel0 = (2, 8, 8)
        >>> rb_channel1 = (128, 128, 128)
        >>> rb_channel2 = (32, 8, 8)
        >>> rb_channel3 = (2, 1, 1)
        >>> out_channels_tuple = (rb_channel0, rb_channel1, rb_channel2, rb_channel3)
        >>> kernel_sizes_tuple = (((5, 5), ) * 2 + ((1, 1), ), ) * 4
        >>> strides_tuple = ((1, 1, 1), ) * 4
        >>> use_bns_tuple = ((True, True, True), ) * 3 + ((False, False, False), )
        >>> acts_tuple = (("relu", None, None), ) * 4
        >>> model = ppsci.arch.Generator(("in",), ("out",), in_channel, out_channels_tuple, kernel_sizes_tuple, strides_tuple, use_bns_tuple, acts_tuple)
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        in_channel: int,
        out_channels_tuple: Tuple[Tuple[int, ...], ...],
        kernel_sizes_tuple: Tuple[Tuple[int, ...], ...],
        strides_tuple: Tuple[Tuple[int, ...], ...],
        use_bns_tuple: Tuple[Tuple[bool, ...], ...],
        acts_tuple: Tuple[Tuple[str, ...], ...],
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.in_channel = in_channel
        self.out_channels_tuple = out_channels_tuple
        self.kernel_sizes_tuple = kernel_sizes_tuple
        self.strides_tuple = strides_tuple
        self.use_bns_tuple = use_bns_tuple
        self.acts_tuple = acts_tuple

        self.init_block_i(0)
        self.layers_res_block0 = nn.LayerList(self.layers_res_block)
        self.init_block_i(1)
        self.layers_res_block1 = nn.LayerList(self.layers_res_block)
        self.init_block_i(2)
        self.layers_res_block2 = nn.LayerList(self.layers_res_block)
        self.init_block_i(3)
        self.layers_res_block3 = nn.LayerList(self.layers_res_block)
        self.layers = [
            self.layers_res_block0,
            self.layers_res_block1,
            self.layers_res_block2,
            self.layers_res_block3,
        ]

    def init_block_i(self, i):
        self.layers_res_block = []
        in_channel = self.in_channel if i == 0 else self.out_channels_tuple[i - 1][-1]
        self.init_res_block(
            in_channel,
            self.out_channels_tuple[i],
            self.kernel_sizes_tuple[i],
            self.strides_tuple[i],
            self.use_bns_tuple[i],
            self.acts_tuple[i],
        )

    def init_conv_block(
        self, in_channel, out_channel, kernel_size, stride, use_bn, act
    ):
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Normal(mean=0.0, std=0.04)
        )
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(value=0.1))
        self.layers_res_block.append(
            nn.Conv2D(
                in_channel,
                out_channel,
                kernel_size,
                stride,
                padding="SAME",
                weight_attr=weight_attr,
                bias_attr=bias_attr,
            )
        )
        self.layers_res_block.append(nn.BatchNorm2D(out_channel) if use_bn else None)
        self.layers_res_block.append(act_mod.get_activation(act) if act else None)

    def init_res_block(
        self, in_channel, out_channels, kernel_sizes, strides, use_bns, acts
    ):
        self.init_conv_block(
            in_channel,
            out_channels[0],
            kernel_sizes[0],
            strides[0],
            use_bns[0],
            acts[0],
        )
        self.init_conv_block(
            out_channels[0],
            out_channels[1],
            kernel_sizes[1],
            strides[1],
            use_bns[1],
            acts[1],
        )

        self.init_conv_block(
            in_channel,
            out_channels[2],
            kernel_sizes[2],
            strides[2],
            use_bns[2],
            acts[2],
        )

        self.layers_res_block.append(act_mod.get_activation("relu"))

    def forward_res(self, x: paddle.Tensor) -> paddle.Tensor:
        """Forward of one res block.

        Args:
            x (paddle.Tensor): Input of the block.

        Returns:
            paddle.Tensor: Output of the block.
        """
        y1 = x
        y2 = x
        for i in range(6):
            if self.layers_res_block[i]:
                y1 = self.layers_res_block[i](y1)

        for i in range(6, 9):
            if self.layers_res_block[i]:
                y2 = self.layers_res_block[i](y2)

        y = self.layers_res_block[-1](y1 + y2)

        return y

    def forward_tensor(self, x):
        y = x

        for i in range(len(self.out_channels_tuple)):
            self.layers_res_block = self.layers[i]
            y = self.forward_res(y)

        return y

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)

        y = self.concat_to_tensor(x, self.input_keys, axis=-1)
        y = self.forward_tensor(y)
        y = self.split_to_dict(y, self.output_keys, axis=-1)

        if self._output_transform is not None:
            y = self._output_transform(y)
        return y


class Discriminator(base.Arch):
    """Discriminator Net of GAN.

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("input1", "input2").
        output_keys (Tuple[str, ...]): Name of output keys, such as ("output1", "output2").
        in_channel (int):  Number of input channels of the first conv layer.
        out_channels (Tuple[int, ...]): Number of output channels of all conv layers,
            such as (out_conv0, out_conv1, out_conv2).
        fc_channel (int):  Number of input features of linear layer. Number of output features of the layer
            is set to 1 in this Net to construct a fully_connected layer.
        kernel_sizes (Tuple[int, ...]): Number of kernel_size of all conv layers,
            such as (kernel_size_conv0, kernel_size_conv1, kernel_size_conv2).
        strides (Tuple[int, ...]): Number of stride of all conv layers,
            such as (stride_conv0, stride_conv1, stride_conv2).
        use_bns (Tuple[bool, ...]): Whether to use the batch_norm layer after each conv layer.
        acts (Tuple[str, ...]): Whether to use the activation layer after each conv layer. If so, witch activation to use,
            such as (act_conv0, act_conv1, act_conv2).

    Examples:
        >>> import ppsci
        >>> in_channel = 2
        >>> in_channel_tempo = 3
        >>> out_channels = (32, 64, 128, 256)
        >>> fc_channel = 65536
        >>> kernel_sizes = ((4, 4), (4, 4), (4, 4), (4, 4))
        >>> strides = (2, 2, 2, 1)
        >>> use_bns = (False, True, True, True)
        >>> acts = ("leaky_relu", "leaky_relu", "leaky_relu", "leaky_relu", None)
        >>> output_keys_disc = ("out_1", "out_2", "out_3", "out_4", "out_5", "out_6", "out_7", "out_8", "out_9", "out_10")
        >>> model = ppsci.arch.Discriminator(("in_1","in_2"), output_keys_disc, in_channel, out_channels, fc_channel, kernel_sizes, strides, use_bns, acts)
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        in_channel: int,
        out_channels: Tuple[int, ...],
        fc_channel: int,
        kernel_sizes: Tuple[int, ...],
        strides: Tuple[int, ...],
        use_bns: Tuple[bool, ...],
        acts: Tuple[str, ...],
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.in_channel = in_channel
        self.out_channels = out_channels
        self.fc_channel = fc_channel
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.use_bns = use_bns
        self.acts = acts

        self.layers_conv2d = []
        self.layers_bn = []
        self.layers_act = []
        self.layers_fc = []

        self.init_layers()
        self.layers_conv2d = nn.LayerList(self.layers_conv2d)
        self.layers_bn = nn.LayerList(self.layers_bn)
        self.layers_act = nn.LayerList(self.layers_act)
        self.layers_fc = nn.LayerList(self.layers_fc)

    def init_layers(self):
        self.init_conv_block(
            self.in_channel,
            self.out_channels[0],
            self.kernel_sizes[0],
            self.strides[0],
            self.use_bns[0],
            self.acts[0],
        )
        self.init_conv_block(
            self.out_channels[0],
            self.out_channels[1],
            self.kernel_sizes[1],
            self.strides[1],
            self.use_bns[1],
            self.acts[1],
        )
        self.init_conv_block(
            self.out_channels[1],
            self.out_channels[2],
            self.kernel_sizes[2],
            self.strides[2],
            self.use_bns[2],
            self.acts[2],
        )
        self.init_conv_block(
            self.out_channels[2],
            self.out_channels[3],
            self.kernel_sizes[3],
            self.strides[3],
            self.use_bns[3],
            self.acts[3],
        )
        self.init_fc_block(self.fc_channel, self.acts[4])

    def init_conv_block(
        self, in_channel, out_channel, kernel_size, stride, use_bn, act
    ):
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Normal(mean=0.0, std=0.04)
        )
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(value=0.1))
        self.layers_conv2d.append(
            nn.Conv2D(
                in_channel,
                out_channel,
                kernel_size,
                stride,
                padding="SAME",
                weight_attr=weight_attr,
                bias_attr=bias_attr,
            )
        )
        self.layers_bn.append(nn.BatchNorm2D(out_channel) if use_bn else None)
        self.layers_act.append(act_mod.get_activation(act) if act else None)

    def init_fc_block(self, in_channel, act):
        self.layers_fc.append(nn.Flatten())
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Normal(mean=0.0, std=0.04)
        )
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(value=0.1))
        self.layers_fc.append(
            nn.Linear(
                in_channel,
                1,
                weight_attr=weight_attr,
                bias_attr=bias_attr,
            )
        )
        self.layers_act.append(act_mod.get_activation(act) if act else None)

    def forward_tensor(self, x):
        y = x
        y_list = []
        for i in range(len(self.layers_conv2d)):
            y = self.layers_conv2d[i](y)
            if self.layers_bn[i]:
                y = self.layers_bn[i](y)
            if self.layers_act[i]:
                y = self.layers_act[i](y)
            y_list.append(y)

        for layer_fc in self.layers_fc:
            y = layer_fc(y)
        if self.layers_act[-1]:
            y = self.layers_act[-1](y)
        y_list.append(y)

        return y_list  # y_conv1, y_conv2, y_conv3, y_conv4, y_fc(y_out)

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)

        y_list = []
        # y1_conv1, y1_conv2, y1_conv3, y1_conv4, y1_fc, y2_conv1, y2_conv2, y2_conv3, y2_conv4, y2_fc
        for k in x:
            y_list.extend(self.forward_tensor(x[k]))

        y = self.split_to_dict(y_list, self.output_keys)

        if self._output_transform is not None:
            y = self._output_transform(y)

        return y

    def split_to_dict(
        self, data_list: List[paddle.Tensor], keys: Tuple[str, ...]
    ) -> Dict[str, paddle.Tensor]:
        """Overwrite of split_to_dict() method belongs to Class base.Arch.
            Reason for overwriting is there is no concat_to_tensor() method called in "tempoGAN" example.
            That is because input in "tempoGAN" example is not in a regular format, but a format like:
            {
                "input1": paddle.concat([in1, in2], axis=1),
                "input2": paddle.concat([in1, in3], axis=1),
            }

        Args:
            data_list (List[paddle.Tensor]): The data to be splited. It should be a list of tensor(s), but not a paddle.Tensor.
            keys (Tuple[str, ...]): Keys of outputs.

        Returns:
            Dict[str, paddle.Tensor]: Dict with splited data.
        """
        if len(keys) == 1:
            return {keys[0]: data_list[0]}
        return {key: data_list[i] for i, key in enumerate(keys)}
