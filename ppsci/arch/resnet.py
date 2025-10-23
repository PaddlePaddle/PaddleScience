import paddle
import paddle.nn as nn

from ppsci.arch import base


class ResNetBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2D(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, 1, stride),
                nn.BatchNorm2D(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(base.Arch):
    # resnet in paddle

    def __init__(
        self,
        input_keys,
        output_keys,
        num_blocks=(2, 2, 2, 2),  # ResNet18
        num_classes=1,
        in_channels=3,
        base_channels=64,
        **kwargs
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys

        self.conv1 = nn.Conv2D(in_channels, base_channels, 7, 2, 3)
        self.bn1 = nn.BatchNorm2D(base_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(3, 2, 1)

        self.layer1 = self._make_layer(base_channels, base_channels, num_blocks[0])
        self.layer2 = self._make_layer(
            base_channels, base_channels * 2, num_blocks[1], stride=2
        )
        self.layer3 = self._make_layer(
            base_channels * 2, base_channels * 4, num_blocks[2], stride=2
        )
        self.layer4 = self._make_layer(
            base_channels * 4, base_channels * 8, num_blocks[3], stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(base_channels * 8, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [ResNetBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: dict, input_keys
        if isinstance(x, dict):
            x = x[self.input_keys[0]]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        x = self.fc(x)
        return x
