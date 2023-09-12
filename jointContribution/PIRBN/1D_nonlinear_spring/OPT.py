import Cal_jac
import numpy as np
import paddle

paddle.framework.core.set_prim_eager_enabled(True)


class Adam:
    def __init__(self, pirbn, x_train, y_train, learning_rate=0.001, maxiter=10000):
        # set attributes
        self.pirbn = pirbn
        self.learning_rate = learning_rate
        self.x_train = [
            paddle.to_tensor(x, dtype=paddle.get_default_dtype()) for x in x_train
        ]
        self.y_train = [
            paddle.to_tensor(y, dtype=paddle.get_default_dtype()) for y in y_train
        ]
        self.maxiter = maxiter
        self.his_l1 = []
        self.his_l2 = []
        self.his_l3 = []
        self.iter = 0
        self.a_g = 1.0
        self.a_b1 = 1.0
        self.a_b2 = 1.0

    def set_weights(self, flat_weights):
        # get model weights
        shapes = [w.shape for w in self.pirbn.get_weights()]
        # compute splitting indices
        split_ids = np.cumsum([np.prod(shape) for shape in [0] + shapes])
        # reshape weights
        weights = [
            flat_weights[from_id:to_id]
            .reshape(shape)
            .astype(paddle.get_default_dtype())
            for from_id, to_id, shape in zip(split_ids[:-1], split_ids[1:], shapes)
        ]
        # set weights to the model
        self.pirbn.set_weights(weights)

    def Loss(self, x, y, a_g, a_b1, a_b2):
        tmp = self.pirbn(x)
        l1 = 0.5 * paddle.mean(paddle.square(tmp[0] - y[0]))
        l2 = 0.5 * paddle.mean(paddle.square(tmp[1]))
        l3 = 0.5 * paddle.mean(paddle.square(tmp[2]))
        loss = l1 * a_g + l2 * a_b1 + l3 * a_b2
        grads = paddle.grad(
            loss, self.pirbn.parameters(), retain_graph=True, create_graph=True
        )
        return loss, grads, l1, l2, l3

    def evaluate(self, weights):
        weights = paddle.to_tensor(weights)
        # update weights
        self.set_weights(weights)
        # compute loss and gradients for weights
        loss, grads, l1, l2, l3 = self.Loss(
            self.x_train, self.y_train, self.a_g, self.a_b1, self.a_b2
        )
        l1_numpy = float(l1)
        l2_numpy = float(l2)
        l3_numpy = float(l3)
        self.his_l1.append(l1_numpy)
        self.his_l2.append(l2_numpy)
        self.his_l3.append(l3_numpy)
        if self.iter % 200 == 0:
            self.a_g, self.a_b1, self.a_b2, _ = Cal_jac.cal_adapt(
                self.pirbn, self.x_train
            )
            print(
                "\ta_g =",
                float(self.a_g),
                "\ta_b1 =",
                float(self.a_b1),
                "\ta_b2 =",
                float(self.a_b2),
            )
            print(
                "Iter: ",
                self.iter,
                "\tL1 =",
                l1_numpy,
                "\tL2 =",
                l2_numpy,
                "\tL3 =",
                l3_numpy,
            )
        self.iter = self.iter + 1
        # convert tf.Tensor to flatten ndarray
        loss = loss.astype("float64").item()
        grads = np.concatenate([g.numpy().flatten() for g in grads]).astype("float64")
        return loss, grads

    def fit(self):
        # get initial weights as a flat vector
        initial_weights = np.concatenate(
            [w.numpy().flatten() for w in self.pirbn.get_weights()]
        )
        print(f"Optimizer: Adam (maxiter={self.maxiter})")
        beta1 = 0.9
        beta2 = 0.999
        learning_rate = self.learning_rate
        eps = 1e-8
        x0 = initial_weights
        x = x0
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        b_w = 0

        for i in range(0, self.maxiter):
            loss, g = self.evaluate(x)
            m = (1 - beta1) * g + beta1 * m
            v = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
            mhat = m / (1 - beta1 ** (i + 1))  # bias correction.
            vhat = v / (1 - beta2 ** (i + 1))
            x = x - learning_rate * mhat / (np.sqrt(vhat) + eps)

        return loss, [self.his_l1, self.his_l2], b_w
