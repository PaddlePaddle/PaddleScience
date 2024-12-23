import paddle
from models.MLP import MLP


class NN(paddle.nn.Layer):

    def __init__(self, hparams, encoder, decoder):
        super(NN, self).__init__()
        self.nb_hidden_layers = hparams['nb_hidden_layers']
        self.size_hidden_layers = hparams['size_hidden_layers']
        self.bn_bool = hparams['bn_bool']
        self.activation = paddle.nn.ReLU()
        self.encoder = encoder
        self.decoder = decoder
        self.dim_enc = hparams['encoder'][-1]
        self.nn = MLP([self.dim_enc] + [self.size_hidden_layers] * self.
            nb_hidden_layers + [self.dim_enc], batch_norm=self.bn_bool)

    def forward(self, data):
        z = self.encoder(data.x)
        z = self.nn(z)
        z = self.decoder(z)
        return z
