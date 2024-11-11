import paddle


def init_normal_(x, mean, std):
    initializer = paddle.nn.initializer.Normal(mean, std)
    initializer(x)


def init_constant_(x, value):
    initializer = paddle.nn.initializer.Constant(value)
    initializer(x)
