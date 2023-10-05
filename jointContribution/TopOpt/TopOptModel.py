import paddle
from paddle import nn

import ppsci


# NCHW data format
class TopOptNN(ppsci.arch.UNetEx):
    def __init__(
        self,
        input_key="input",
        output_key="output",
        in_channel=2,
        out_channel=1,
        kernel_size=3,
        filters=(16, 32, 64),
        layers=2,
        weight_norm=False,
        batch_norm=False,
        activation=nn.ReLU,
    ):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.filters = filters
        super().__init__(
            input_key=input_key,
            output_key=output_key,
            in_channel=in_channel,
            out_channel=out_channel,
            kernel_size=kernel_size,
            filters=filters,
            layers=layers,
            weight_norm=weight_norm,
            batch_norm=batch_norm,
            activation=activation,
        )
        # Modify Layers
        self.encoder[1] = nn.Sequential(
            nn.MaxPool2D(self.in_channel, padding="SAME"),
            self.encoder[1][0],
            nn.Dropout2D(0.1),
            self.encoder[1][1],
        )
        self.encoder[2] = nn.Sequential(
            nn.MaxPool2D(2, padding="SAME"), self.encoder[2]
        )
        # Conv2D used in reference code in decoder
        self.decoders[0] = nn.Sequential(
            nn.Conv2D(
                self.filters[-1], self.filters[-1], kernel_size=3, padding="SAME"
            ),
            nn.ReLU(),
            nn.Conv2D(
                self.filters[-1], self.filters[-1], kernel_size=3, padding="SAME"
            ),
            nn.ReLU(),
        )
        self.decoders[1] = nn.Sequential(
            nn.Conv2D(
                sum(self.filters[-2:]), self.filters[-2], kernel_size=3, padding="SAME"
            ),
            nn.ReLU(),
            nn.Dropout2D(0.1),
            nn.Conv2D(
                self.filters[-2], self.filters[-2], kernel_size=3, padding="SAME"
            ),
            nn.ReLU(),
        )
        self.decoders[2] = nn.Sequential(
            nn.Conv2D(
                sum(self.filters[:-1]), self.filters[-3], kernel_size=3, padding="SAME"
            ),
            nn.ReLU(),
            nn.Conv2D(
                self.filters[-3], self.filters[-3], kernel_size=3, padding="SAME"
            ),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Conv2D(
                self.filters[-3], self.out_channel, kernel_size=3, padding="SAME"
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x[self.input_keys[0]].squeeze(axis=0)  # squeeze additional batch dimension
        # Layer 1 (bs, 2, 40, 40) -> (bs, 16, 40, 40)
        conv1 = self.encoder[0](x)
        up_size_2 = conv1.shape[-2:]
        # Layer 2 (bs, 16, 40, 40) -> (bs, 32, 20, 20)
        conv2 = self.encoder[1](conv1)
        up_size_1 = conv2.shape[-2:]
        # Layer 3 (bs, 32, 20, 20) -> (bs, 64, 10, 10)
        conv3 = self.encoder[2](conv2)

        # Layer 4 (bs, 64, 10, 10) -> (bs, 64, 10, 10)
        conv4 = self.decoders[0](conv3)
        # upsampling (bs, 64, 10, 10) -> (bs, 64, 20, 20)
        conv4 = nn.UpsamplingNearest2D(up_size_1)(conv4)

        # concat (bs, 64, 20, 20) -> (bs, 96, 20, 20)
        conv5 = paddle.concat((conv2, conv4), axis=1)
        # Layer 5 (bs, 96, 20, 20) -> (bs, 32, 20, 20)
        conv5 = self.decoders[1](conv5)
        # upsampling (bs, 32, 20, 20) -> (bs, 32, 40, 40)
        conv5 = nn.UpsamplingNearest2D(up_size_2)(conv5)

        # concat (bs, 32, 40, 40) -> (bs, 48, 40, 40)
        conv6 = paddle.concat((conv1, conv5), axis=1)
        # Layer 6 (bs, 48, 40, 40) -> (bs, 16, 40, 40)
        conv6 = self.decoders[2](conv6)
        # Output (bs, 16, 40, 40) -> (bs, 1, 40, 40)
        out = self.output(conv6)

        return {self.output_keys[0]: out}
