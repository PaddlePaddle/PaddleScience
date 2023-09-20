import paddle


class PIRBN(paddle.nn.Layer):
    def __init__(self, rbn, activation_function="gaussian_function"):
        super().__init__()
        self.rbn = rbn
        self.activation_function = activation_function

    def forward(self, input_data):
        xy, xy_b = input_data
        # initialize the differential operators
        u_b = self.rbn(xy_b, self.activation_function)

        # obtain partial derivatives of u with respect to x
        xy.stop_gradient = False
        # Obtain the output from the RBN
        u = self.rbn(xy, self.activation_function)
        # Obtain the first-order derivative of the output with respect to the input
        u_x = paddle.grad(u, xy, retain_graph=True, create_graph=True)[0]
        # Obtain the second-order derivative of the output with respect to the input
        u_xx = paddle.grad(u_x, xy, retain_graph=True, create_graph=True)[0]

        return [u_xx, u_b]

    def cal_ntk(self, x):
        # Formula (4), Page3, \lambda variable
        # Lambda represents the eigenvalues of the matrix(Kg)
        lambda_g = 0.0
        lambda_b = 0.0
        n_neu = self.rbn.n_neu

        # in-domain
        n1 = x[0].shape[0]
        for i in range(n1):
            temp_x = [x[0][i, ...].unsqueeze(0), paddle.to_tensor([[0.0]])]
            y = self.forward(temp_x)
            l1t = paddle.grad(y[0], self.parameters(), allow_unused=True)
            for j in l1t:
                if j is not None:
                    lambda_g = lambda_g + paddle.sum(j**2) / n1
            # When use tanh activation function, the value may be None
            if l1t[0] is None and l1t[1] is not None:
                temp = l1t[1].reshape((1, n_neu))
            else:
                temp = paddle.concat((l1t[0], l1t[1].reshape((1, n_neu))), axis=1)
            if i == 0:
                # Fig.1, Page5, Kg variable
                Kg = temp
            else:
                Kg = paddle.concat((Kg, temp), axis=0)

        # bound
        n2 = x[1].shape[0]
        for i in range(n2):
            temp_x = [paddle.to_tensor([[0.0]]), x[1][i, ...].unsqueeze(0)]
            y = self.forward(temp_x)
            l1t = paddle.grad(y[1], self.rbn.parameters(), allow_unused=True)
            for j in l1t:
                # When use tanh activation function, the value may be None
                if j is not None:
                    lambda_b = lambda_b + paddle.sum(j**2) / n2

        # calculate adapt factors
        temp = lambda_g + lambda_b
        lambda_g = temp / lambda_g
        lambda_b = temp / lambda_b

        return lambda_g, lambda_b, Kg
