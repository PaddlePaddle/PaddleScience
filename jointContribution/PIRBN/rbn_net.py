import math

import numpy as np
import paddle


class RBN_Net(paddle.nn.Layer):
    """This class is to build a radial basis network (RBN).

    Args:
        n_in (int): Number of input of the RBN.
        n_out (int): Number of output of the RBN.
        n_neu (int): Number of neurons in the hidden layer.
        b (List[float32]|float32): Initial value for hyperparameter b.
        c (List[float32]): Initial value for hyperparameter c.
    """

    def __init__(self, n_in, n_out, n_neu, b, c, activation_function="gaussian"):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_neu = n_neu
        self.b = paddle.to_tensor(b)
        self.c = paddle.to_tensor(c)
        self.activation_function = activation_function
        self.activation = Activation(self.n_neu, self.c, n_in, activation_function)

        # LeCun normal initialization
        std = math.sqrt(1 / self.n_neu)
        ini = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Normal(mean=0.0, std=std)
        )

        if self.activation_function == "gaussian":
            # gaussian activation_function need to set self.b
            self.ini_ab()
            self.last_fc_layer = paddle.nn.Linear(
                self.n_neu,
                self.n_out,
                weight_attr=ini,
                bias_attr=False,
            )

        elif self.activation_function == "tanh":
            w = paddle.load("./PINN/w.tensor")
            b = paddle.load("./PINN/b.tensor")
            w_0 = paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(w[0]))
            b_0 = paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(b[0]))
            w_1 = paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(w[1]))
            self.hidden_layer = paddle.nn.Linear(
                self.n_in, self.n_neu, weight_attr=w_0, bias_attr=b_0  # ini,  # ini,
            )

            self.last_fc_layer = paddle.nn.Linear(
                self.n_neu,
                self.n_out,
                weight_attr=w_1,  # ,ini,
                bias_attr=False,  # ini,
            )

            # b.shape == [1]
            self.last_fc_bias = self.create_parameter(
                shape=[1, 1],
                default_initializer=paddle.nn.initializer.Assign(b[1]),
                dtype=paddle.get_default_dtype(),
            )
        else:
            raise ("Not implemented yet")

    def forward(self, x):
        if self.activation_function == "gaussian":
            y = self.activation(x)
            y = self.last_fc_layer(y)
        elif self.activation_function == "tanh":
            # input : n x 1
            # hidden layer : 1 x 61
            # last fc layer : 61 x 1
            y = self.hidden_layer(x)
            y = self.activation(y)
            y = self.last_fc_layer(y)
            y = paddle.add(y, self.last_fc_bias)

        else:
            raise ("Not implemented yet")
        return y

    def ini_ab(self):
        b = np.ones((1, self.n_neu)) * self.b
        self.activation.b = self.activation.create_parameter(
            (1, self.n_neu), default_initializer=paddle.nn.initializer.Assign(b)
        )


class Activation(paddle.nn.Layer):
    """This class is to create the hidden layer of a radial basis network.

    Args:
        n_neu (int): Number of neurons in the hidden layer.
        c (List[float32]): Initial value for hyperparameter b.
        n_in (int): Last item of input shape.
    """

    def __init__(self, n_neu, c, n_in, activation_function="gaussian"):
        super(Activation, self).__init__()
        self.activation_function = activation_function
        # PINN      y = w2 * (tanh(w1 * x + b1)) + b2         w,b are trainable parameters, b is bais
        # PIRBN     y = w * exp(b^2 * |x-c|^2)          w,b are trainable parameters, c is constant, b is not bias

        if self.activation_function == "gaussian":
            self.n_neu = n_neu
            self.c = c
            self.b = self.create_parameter(
                shape=(n_in, self.n_neu),
                dtype=paddle.get_default_dtype(),
                # Convert from tensorflow tf.random_normal_initializer(), default value mean=0.0, std=0.05
                default_initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.05),
            )

    def forward(self, inputs):
        if self.activation_function == "gaussian":
            return self.gaussian_function(inputs)
        elif self.activation_function == "tanh":
            return self.tanh_function(inputs)

    # Gaussian function，Formula (19), Page7
    def gaussian_function(self, inputs):
        temp_x = paddle.matmul(inputs, paddle.ones((1, self.n_neu)))
        x0 = (
            paddle.reshape(
                paddle.arange(self.n_neu, dtype=paddle.get_default_dtype()),
                (1, self.n_neu),
            )
            * (self.c[1] - self.c[0])
            / (self.n_neu - 1)
            + self.c[0]
        )
        x_new = temp_x - x0
        s = self.b * self.b
        return paddle.exp(-(x_new * x_new) * s)

    def tanh_function(self, inputs):
        return paddle.tanh(inputs)
