import paddle
import paddle.nn as nn


class AdaIN(nn.Layer):
    def __init__(self, embed_dim, in_channels, mlp=None, eps=1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.eps = eps

        if mlp is None:
            mlp = nn.Sequential(
                nn.Linear(embed_dim, 512), nn.GELU(), nn.Linear(512, 2 * in_channels)
            )
        self.mlp = mlp

        self.embedding = None

    def set_embedding(self, x):
        self.embedding = x.reshape(
            [
                self.embed_dim,
            ]
        )

    def forward(self, x):
        assert (
            self.embedding is not None
        ), "AdaIN: update embeddding before running forward"

        mlp = self.mlp(self.embedding)
        # torch.split and paddle.split are different, as following:
        # https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/convert_from_pytorch/api_difference/Tensor/torch.Tensor.permute.html
        weight, bias = paddle.split(mlp, (mlp.shape[0]) // self.in_channels, axis=0)

        return nn.functional.group_norm(
            x, self.in_channels, weight=weight, bias=bias, epsilon=self.eps
        )
