import paddle
from Dif_op import Dif


class PIRBN(paddle.nn.Layer):
    def __init__(self, rbn):
        super().__init__()
        self.rbn = rbn

    def forward(self, input_data):
        xy, xy_b = input_data
        ### initialize the differential operators
        Dif_u = Dif(self.rbn)
        u = self.rbn(xy)
        u_b = self.rbn(xy_b)

        ### obtain partial derivatives of u with respect to x
        _, u_xx = Dif_u(xy)
        u_b_x, _ = Dif_u(xy_b)
        t = u_xx + 4 * u + paddle.sin(u)

        ### build up the PIRBN
        return [t, u_b, u_b_x]

    def get_weights(self):
        return self.rbn.get_weights()

    def set_weights(self, weights):
        self.rbn.set_weights(weights)
