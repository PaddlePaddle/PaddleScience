import paddle
import paddle.nn as nn
from ppsci.arch import base

VGG_CONFIGS = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(base.Arch):
    def __init__(self, in_channel=1, num_classes=2, config='vgg16', input_keys=None, output_keys=None):
        super().__init__()
        self.input_keys = input_keys or ["x"]
        self.output_keys = output_keys or ["logits"]
        self.features = self._make_layers(VGG_CONFIGS[config], in_channel)
        self.num_classes = num_classes
        self.classifier = None  # 延后初始化

    def _make_layers(self, cfg, in_channel):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool1D(kernel_size=2, stride=2), nn.Dropout(0.5)]
            else:
                layers += [
                    nn.Conv1D(in_channel, v, kernel_size=3, padding=1),
                    nn.BatchNorm1D(v, momentum=0.1, epsilon=1e-5),
                    nn.ReLU()
                ]
                in_channel = v
        return nn.Sequential(*layers)
    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        if isinstance(x, dict):
            x = x[self.input_keys[0]]
        if only_fc and self.classifier is not None:
            return self.classifier(x)
        x = self.features(x)
        x = paddle.flatten(x, 1)
        if self.classifier is None:
            # 动态推断flatten后的特征维度
            in_features = x.shape[1]
            self.classifier = nn.Sequential(
                nn.Linear(in_features, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, self.num_classes),
            )
            # 初始化权重
            for m in self.sublayers():
                if isinstance(m, nn.Conv1D):
                    nn.initializer.XavierUniform()(m.weight)
                    if m.bias is not None:
                        nn.initializer.Constant(0.0)(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.initializer.XavierUniform()(m.weight)
                    nn.initializer.Constant(0.0)(m.bias)
        if only_feat:
            return x
        out = self.classifier(x)
        result_dict = {self.output_keys[0]: out, 'feat': x}
        return result_dict

    def no_weight_decay(self):
        nwd = []
        for n, _ in self.named_parameters():
            if 'bn' in n or 'bias' in n:
                nwd.append(n)
        return nwd

def vgg(config='vgg16', pretrained=False, pretrained_path=None, **kwargs):
    model = VGG(config=config, **kwargs)
    return model