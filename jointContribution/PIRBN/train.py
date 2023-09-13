import paddle

# Used to calculate the second-order derivatives
paddle.framework.core.set_prim_eager_enabled(True)


class Trainer:
    def __init__(self, pirbn, x_train, y_train, learning_rate=0.001, maxiter=10000):
        # set attributes
        self.pirbn = pirbn

        self.learning_rate = learning_rate
        self.x_train = [
            paddle.to_tensor(x, dtype=paddle.get_default_dtype()) for x in x_train
        ]
        self.y_train = paddle.to_tensor(y_train, dtype=paddle.get_default_dtype())
        self.maxiter = maxiter
        self.his_l1 = []
        self.his_l2 = []
        self.iter = 0
        self.a_g = paddle.to_tensor(1.0)
        self.a_b = paddle.to_tensor(1.0)
        self.optimizer = paddle.optimizer.Adam(
            learning_rate=0.001, parameters=self.pirbn.parameters()
        )
        self.ntk_list = {}

    def Loss(self, x, y, a_g, a_b):
        tmp = self.pirbn(x)
        l1 = 0.5 * paddle.mean(paddle.square(tmp[0] - y[0]))
        l2 = 0.5 * paddle.mean(paddle.square(tmp[1]))
        loss = l1 * a_g + l2 * a_b
        return loss, l1, l2

    def evaluate(self):
        # compute loss
        loss, l1, l2 = self.Loss(self.x_train, self.y_train, self.a_g, self.a_b)
        l1_numpy = float(l1)
        l2_numpy = float(l2)
        self.his_l1.append(l1_numpy)
        self.his_l2.append(l2_numpy)
        if self.iter % 200 == 0:
            self.a_g, self.a_b, _ = self.pirbn.cal_ntk(self.x_train)
            print("\ta_g =", float(self.a_g), "\ta_b =", float(self.a_b))
            print("Iter: ", self.iter, "\tL1 =", l1_numpy, "\tL2 =", l2_numpy)
        self.iter = self.iter + 1
        return loss

    def fit(self):
        for i in range(0, self.maxiter):
            if i in [0, 2000, 20000]:
                self.ntk_list[i] = self.pirbn.cal_ntk(self.x_train)[2].numpy()
            loss = self.evaluate()
            loss.backward()
            self.optimizer.step()
            self.optimizer.clear_grad()
        return loss, [self.his_l1, self.his_l2]
