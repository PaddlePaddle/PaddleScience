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

    def __init__(self, n_in, n_out, n_neu, b, c):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_neu = n_neu
        self.b = paddle.to_tensor(b)
        self.c = paddle.to_tensor(c)

        self.layer1 = RBF_layer1(self.n_neu, self.c, n_in)

        # LeCun normal
        std = math.sqrt(1 / self.n_neu)
        self.linear = paddle.nn.Linear(
            self.n_neu,
            self.n_out,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(mean=0.0, std=std)
            ),
            bias_attr=False,
        )
        self.ini_ab()

    def forward(self, x):
        temp = self.layer1(x)
        y = self.linear(temp)
        return y

    def ini_ab(self):
        b = np.ones((1, self.n_neu)) * self.b
        self.layer1.b = self.layer1.create_parameter(
            (1, self.n_neu), default_initializer=paddle.nn.initializer.Assign(b)
        )

    def get_weights(self):
        s = self.state_dict()
        ret = [s[i] for i in s]
        return ret

    def set_weights(self, weights):
        s = self.state_dict()
        s["layer1.b"] = weights[0]
        s["linear.weight"] = weights[1]
        self.set_state_dict(s)


class RBF_layer1(paddle.nn.Layer):
    """This class is to create the hidden layer of a radial basis network.

    Args:
        n_neu (int): Number of neurons in the hidden layer.
        c (List[float32]): Initial value for hyperparameter b.
        input_shape_last (int): Last item of input shape.
    """

    def __init__(self, n_neu, c, input_shape_last):
        super(RBF_layer1, self).__init__()
        self.n_neu = n_neu
        self.c = c
        self.b = self.create_parameter(
            shape=(input_shape_last, self.n_neu),
            dtype=paddle.get_default_dtype(),
            # Convert from tensorflow tf.random_normal_initializer(), default value mean=0.0, std=0.05
            default_initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.05),
        )
        # self.b = paddle.normal(mean=0.0, std=0.05, shape=[input_shape_last, self.n_neu])

    def forward(self, inputs):  # Defines the computation from inputs to outputs
        s = self.b * self.b
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
        x_new = (temp_x - x0) * (temp_x - x0)
        return paddle.exp(-x_new * s)
