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
        ### Apply the GradientTape function
        ### Obtain the output from the RBN
        u = self.rbn(x)
        ### Obtain the first-order derivative of the output with respect to the input
        u_x = paddle.grad(u, x, retain_graph=True, create_graph=True)[0]

        ### Obtain the second-order derivative of the output with respect to the input
        u_xx = paddle.grad(u_x, x, retain_graph=True, create_graph=True)[0]
        return u_x, u_xx
