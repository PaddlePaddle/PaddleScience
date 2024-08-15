import paddle as pp
import paddle.nn.functional as F

from paddle import nn
from ppsci.arch.base import Arch
from paddle.nn.initializer import KaimingNormal


class graph_conv(nn.Layer):
    def __init__(self, in_dim, out_dim, dropout, num_layer=2):
        super(graph_conv, self).__init__()
        self.mlp = nn.Conv2D((num_layer + 1) * in_dim, out_dim, kernel_size=(1, 1), weight_attr=KaimingNormal())
        self.dropout = dropout
        self.num_layer = num_layer

    def forward(self, x, adj):
        # B C N T
        out = [x]
        for _ in range(self.num_layer):
            new_x = pp.matmul(adj, x)
            out.append(new_x)
            x = new_x

        h = pp.concat(out, axis=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class tempol_conv(nn.Layer):
    def __init__(self, in_dim, out_dim, hidden, num_layer=3, k_s=3, alpha=0.1):
        super(tempol_conv, self).__init__()
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.tc_convs = nn.LayerList()
        self.num_layer = num_layer
        for i in range(num_layer):
            in_channels = in_dim if i == 0 else hidden
            self.tc_convs.append(nn.Conv2D(in_channels=in_channels, out_channels=hidden,
                                 kernel_size=(1, k_s), padding=(0, i + 1), dilation=i + 1, weight_attr=KaimingNormal()))

        self.mlp = nn.Conv2D(in_channels=in_dim + hidden * num_layer, out_channels=out_dim,
                             kernel_size=(1, 1), weight_attr=KaimingNormal())

    def forward(self, x):
        # B C N T
        x_cat = [x]
        for i in range(self.num_layer):
            x = self.leakyrelu(self.tc_convs[i](x))
            x_cat.append(x)
        tc_out = self.mlp(pp.concat(x_cat, axis=1))
        return tc_out


class TGCN(Arch):
    def __init__(self, cfg, edge_index, edge_attr, adj):
        super(TGCN, self).__init__()
        # para
        in_dim = cfg.input_dim
        emb_dim = cfg.emb_dim
        hidden = cfg.hidden
        gc_layer = cfg.gc_layer
        tc_layer = cfg.tc_layer
        k_s = cfg.tc_kernel_size
        dropout = cfg.dropout
        alpha = cfg.leakyrelu_alpha

        self.input_keys = cfg.MODEL.afno.input_keys
        self.output_keys = cfg.MODEL.afno.label_keys

        self.edge_index = pp.to_tensor(data=edge_index, place=cfg.device)
        self.edge_attr = pp.to_tensor(data=edge_attr, place=cfg.device)
        self.adj = pp.to_tensor(data=adj, place=cfg.device)

        self.emb_conv = nn.Conv2D(in_channels=in_dim, out_channels=emb_dim, kernel_size=(1, 1), weight_attr=KaimingNormal())

        self.tc1_conv = tempol_conv(emb_dim, hidden, hidden, num_layer=tc_layer, k_s=k_s, alpha=alpha)
        self.sc1_conv = graph_conv(hidden, hidden, dropout, num_layer=gc_layer)
        self.bn1 = nn.BatchNorm2D(hidden)

        self.tc2_conv = tempol_conv(hidden, hidden, hidden, num_layer=tc_layer, k_s=k_s, alpha=alpha)
        self.sc2_conv = graph_conv(hidden, hidden, dropout, num_layer=gc_layer)
        self.bn2 = nn.BatchNorm2D(hidden)

        self.end_conv_1 = nn.Conv2D(in_channels=emb_dim + hidden + hidden, out_channels=2 *
                                    hidden, kernel_size=(1, 1), weight_attr=KaimingNormal())
        self.end_conv_2 = nn.Conv2D(in_channels=2 * hidden, out_channels=cfg.label_len,
                                    kernel_size=(1, cfg.input_len), weight_attr=KaimingNormal())

    def forward(self, raw):
        # emb block
        x = raw[self.input_keys[0]]
        x = x.transpose(perm=[0, 3, 2, 1])  # B in_dim N T
        emb_x = self.emb_conv(x)  # B emd_dim N T

        # TC1
        tc1_out = self.tc1_conv(emb_x)  # B hidden N T

        # SC1
        sc1_out = self.sc1_conv(tc1_out, self.adj)  # B hidden N T
        sc1_out = sc1_out + tc1_out
        sc1_out = self.bn1(sc1_out)

        # TC2
        tc2_out = self.tc2_conv(sc1_out)  # B hidden N T

        # SC2
        sc2_out = self.sc2_conv(tc2_out, self.adj)  # B hidden N T
        sc2_out = sc2_out + tc2_out
        sc2_out = self.bn2(sc2_out)

        # readout block
        x_out = F.relu(pp.concat((emb_x, sc1_out, sc2_out), axis=1))
        x_out = F.relu(self.end_conv_1(x_out))
        # transform
        x_out = self.end_conv_2(x_out)  # B T N 1

        return {self.output_keys[0]: x_out}
