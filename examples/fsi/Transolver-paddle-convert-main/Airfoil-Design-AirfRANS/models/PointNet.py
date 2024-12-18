import sys
# sys.path.append('../../utils')
from utils import paddle_aux
import paddle
import torch_geometric.nn as nng
from models.MLP import MLP


class PointNet(paddle.nn.Layer):

    def __init__(self, hparams, encoder, decoder):
        super(PointNet, self).__init__()
        self.base_nb = hparams['base_nb']
        self.in_block = MLP([hparams['encoder'][-1], self.base_nb, self.
            base_nb * 2], batch_norm=False)
        self.max_block = MLP([self.base_nb * 2, self.base_nb * 4, self.
            base_nb * 8, self.base_nb * 32], batch_norm=False)
        self.out_block = MLP([self.base_nb * (32 + 2), self.base_nb * 16, 
            self.base_nb * 8, self.base_nb * 4], batch_norm=False)
        self.encoder = encoder
        self.decoder = decoder
        self.fcfinal = paddle.nn.Linear(in_features=self.base_nb * 4,
            out_features=hparams['encoder'][-1])

    def forward(self, data):
        z, batch = data.x.float(), data.batch.long()
        z = self.encoder(z)
        z = self.in_block(z)
        global_coef = self.max_block(z)
        global_coef = nng.global_max_pool(global_coef, batch=batch)
        nb_points = paddle.zeros(shape=tuple(global_coef.shape)[0])
        for i in range(batch.max() + 1):
            nb_points[i] = (batch == i).sum()
        nb_points = nb_points.astype(dtype='int64')
        global_coef = paddle.repeat_interleave(x=global_coef, repeats=
            nb_points, axis=0)
        z = paddle.concat(x=[z, global_coef], axis=1)
        z = self.out_block(z)
        z = self.fcfinal(z)
        z = self.decoder(z)
        return z
