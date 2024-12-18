import paddle
import torch_geometric.nn as nng


class GraphSAGE(paddle.nn.Layer):

    def __init__(self, hparams, encoder, decoder):
        super(GraphSAGE, self).__init__()
        self.nb_hidden_layers = hparams['nb_hidden_layers']
        self.size_hidden_layers = hparams['size_hidden_layers']
        self.bn_bool = hparams['bn_bool']
        self.activation = paddle.nn.ReLU()
        self.encoder = encoder
        self.decoder = decoder
        self.in_layer = nng.SAGEConv(in_channels=hparams['encoder'][-1],
            out_channels=self.size_hidden_layers)
        self.hidden_layers = paddle.nn.LayerList()
        for n in range(self.nb_hidden_layers - 1):
            self.hidden_layers.append(nng.SAGEConv(in_channels=self.
                size_hidden_layers, out_channels=self.size_hidden_layers))
        self.out_layer = nng.SAGEConv(in_channels=self.size_hidden_layers,
            out_channels=hparams['decoder'][0])
        if self.bn_bool:
            self.bn = paddle.nn.LayerList()
            for n in range(self.nb_hidden_layers):
                self.bn.append(paddle.nn.BatchNorm1D(num_features=self.
                    size_hidden_layers))

    def forward(self, data):
        z, edge_index = data.x, data.edge_index
        z = self.encoder(z)
        z = self.in_layer(z, edge_index)
        if self.bn_bool:
            z = self.bn[0](z)
        z = self.activation(z)
        for n in range(self.nb_hidden_layers - 1):
            z = self.hidden_layers[n](z, edge_index)
            if self.bn_bool:
                z = self.bn[n + 1](z)
            z = self.activation(z)
        z = self.out_layer(z, edge_index)
        z = self.decoder(z)
        return z
