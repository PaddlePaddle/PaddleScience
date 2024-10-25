import paddle
from flashbert import BertConfig
from flashbert import BertModel
from paddle.nn import Linear


class Bert(paddle.nn.Layer):
    def __init__(self, hidden=512):
        super().__init__()
        self.config = BertConfig(
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            hidden_size=hidden,
            num_attention_heads=4,
            num_hidden_layers=10,
            max_position_embeddings=1,
            intermediate_size=2048,
            vocab_size=1,
        )
        self.bert = BertModel(self.config)

    def forward(self, x):
        x = x.unsqueeze(axis=0)
        x = self.bert(inputs_embeds=x)[0]
        x = x.view(-1, self.config.hidden_size)
        return x


class ConvNet(paddle.nn.Layer):
    def __init__(self, num_layers=5, hidden=512, out=None, residual=False):
        super().__init__()
        convs = []
        for i in range(num_layers):
            convs.append(
                paddle.nn.Conv1D(in_channels=hidden, out_channels=hidden, kernel_size=1)
            )
            convs.append(paddle.nn.ReLU())
        if out is not None:
            convs.append(
                paddle.nn.Conv1D(in_channels=hidden, out_channels=out, kernel_size=1)
            )
        self.convs = paddle.nn.Sequential(*convs)
        self.residual = residual

    def forward(self, x):
        x_0 = x
        x = x.t().unsqueeze(axis=0)
        x = self.convs(x)
        x = x.transpose(perm=[0, 2, 1]).squeeze()
        if self.residual:
            x = x + x_0
        return x


class ConvBert(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        hidden = 512
        self.mlp_in = Linear(4, hidden)
        self.cnn_in = ConvNet(2, hidden, residual=False)
        self.ln = paddle.nn.LayerNorm(normalized_shape=hidden)
        self.bert = Bert(hidden)
        self.cnn_out = ConvNet(2, hidden, out=1)

    def forward(self, data):
        pos = data.pos
        area = data.area
        x = paddle.concat(x=[pos, area.unsqueeze(axis=1)], axis=1)
        x = self.mlp_in(x)
        x = self.cnn_in(x)
        x = self.ln(x)
        x = self.bert(x)
        x = self.cnn_out(x)
        return x.squeeze()
