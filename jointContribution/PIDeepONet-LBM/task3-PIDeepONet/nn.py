import paddle as pd


class DeepONet(pd.nn.Layer):
    def __init__(self, h, out_dims):
        super(DeepONet, self).__init__()

        ##paddle-Branch net
        self.fnn_B = pd.nn.Sequential(
            pd.nn.Linear(h, 128),
            pd.nn.Tanh(),
            pd.nn.Linear(128, 128),
            pd.nn.Tanh(),
            pd.nn.Linear(128, out_dims),
        )

        ##paddle-Trunk net
        self.fnn_T = pd.nn.Sequential(
            pd.nn.Linear(2, 128),
            pd.nn.Tanh(),
            pd.nn.Linear(128, 128),
            pd.nn.Tanh(),
            pd.nn.Linear(128, out_dims),
        )

    def forward(self, Bin, Tin):
        return self.fnn_B(Bin), self.fnn_T(Tin)
