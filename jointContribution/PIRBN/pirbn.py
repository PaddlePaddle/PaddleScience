import paddle
from jacobian_function import jacobian


class PIRBN(paddle.nn.Layer):
    def __init__(self, rbn, activation_function="gaussian"):
        super().__init__()
        self.rbn = rbn
        self.activation_function = activation_function

    def forward(self, input_data):
        xy, xy_b = input_data
        # initialize the differential operators
        u_b = self.rbn(xy_b)
        # obtain partial derivatives of u with respect to x
        xy.stop_gradient = False
        # Obtain the output from the RBN
        u = self.rbn(xy)
        # Obtain the first-order derivative of the output with respect to the input
        u_x = paddle.grad(u, xy, retain_graph=True, create_graph=True)[0]
        # Obtain the second-order derivative of the output with respect to the input
        u_xx = paddle.grad(u_x, xy, retain_graph=True, create_graph=True)[0]
        return u_xx, u_b, u

    def cal_K(self, x):
        u_xx, _, _ = self.forward(x)
        w, b = [], []

        if self.activation_function == "gaussian":
            b.append(self.rbn.activation.b)
            w.append(self.rbn.last_fc_layer.weight)
        elif self.activation_function == "tanh":
            w.append(self.rbn.hidden_layer.weight)
            b.append(self.rbn.hidden_layer.bias)
            w.append(self.rbn.last_fc_layer.weight)

        J_list = []

        for w_i in w:
            J_w = jacobian(u_xx, w_i).squeeze()
            J_list.append(J_w)

        for b_i in b:
            J_b = jacobian(u_xx, b_i).squeeze()
            J_list.append(J_b)

        n_input = x[0].shape[0]  # ns in main.py
        K = paddle.zeros((n_input, n_input))

        for J in J_list:
            K += J @ J.T

        return K

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
            for j, grad in enumerate(l1t):
                if grad is None:
                    grad = paddle.to_tensor([0.0]).broadcast_to(
                        self.parameters()[j].shape
                    )
                    l1t[j] = grad
                lambda_g = lambda_g + paddle.sum(grad**2) / n1

            # When use tanh activation function, the value may be None
            if self.activation_function == "tanh":
                temp = paddle.concat(
                    # (l1t[0], l1t[1], l1t[2].reshape((1, n_neu))), axis=1
                    # Not select last_fc_bias
                    (l1t[1], l1t[2], l1t[3].reshape((1, n_neu))),
                    axis=1,
                )
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
                lambda_b = lambda_b + paddle.sum(j**2) / n2

        # calculate adapt factors
        temp = lambda_g + lambda_b
        lambda_g = temp / lambda_g
        lambda_b = temp / lambda_b

        return lambda_g, lambda_b, Kg
