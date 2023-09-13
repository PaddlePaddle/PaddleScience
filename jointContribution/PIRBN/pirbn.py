import paddle


class Dif(paddle.nn.Layer):
    """This function is to initialise for differential operator.

    Args:
        rbn (model): The Feedforward Neural Network.
    """

    def __init__(self, rbn, **kwargs):
        super().__init__(**kwargs)
        self.rbn = rbn

    def forward(self, x):
        """This function is to calculate the differential terms.

        Args:
            x (Tensor): The coordinate array

        Returns:
            Tuple[Tensor, Tensor]: The first-order derivative of the u with respect to the x; The second-order derivative of the u with respect to the x.
        """
        x.stop_gradient = False
        # Obtain the output from the RBN
        u = self.rbn(x)
        # Obtain the first-order derivative of the output with respect to the input
        u_x = paddle.grad(u, x, retain_graph=True, create_graph=True)[0]
        # Obtain the second-order derivative of the output with respect to the input
        u_xx = paddle.grad(u_x, x, retain_graph=True, create_graph=True)[0]
        return u_x, u_xx


class PIRBN(paddle.nn.Layer):
    def __init__(self, rbn):
        super().__init__()
        self.rbn = rbn

    def forward(self, input_data):
        xy, xy_b = input_data
        # initialize the differential operators
        Dif_u = Dif(self.rbn)
        u_b = self.rbn(xy_b)

        # obtain partial derivatives of u with respect to x
        _, u_xx = Dif_u(xy)

        return [u_xx, u_b]

    def cal_ntk(self, x):
        # Formula (4), Page5, \gamma variable
        gamma_g = 0.0
        gamma_b = 0.0
        n_neu = self.rbn.n_neu

        # in-domain
        n1 = x[0].shape[0]
        for i in range(n1):
            temp_x = [x[0][i, ...].unsqueeze(0), paddle.to_tensor([[0.0]])]
            y = self.forward(temp_x)
            l1t = paddle.grad(y[0], self.parameters())
            for j in l1t:
                gamma_g = gamma_g + paddle.sum(j**2) / n1
            temp = paddle.concat((l1t[0], l1t[1].reshape((1, n_neu))), axis=1)
            if i == 0:
                # Fig.1, Page8, Kg variable
                Kg = temp
            else:
                Kg = paddle.concat((Kg, temp), axis=0)

        # bound
        n2 = x[1].shape[0]
        for i in range(n2):
            temp_x = [paddle.to_tensor([[0.0]]), x[1][i, ...].unsqueeze(0)]
            y = self.forward(temp_x)
            l1t = paddle.grad(y[1], self.parameters())
            for j in l1t:
                gamma_b = gamma_b + paddle.sum(j**2) / n2

        # calculate adapt factors
        temp = gamma_g + gamma_b
        gamma_g = temp / gamma_g
        gamma_b = temp / gamma_b

        return gamma_g, gamma_b, Kg
