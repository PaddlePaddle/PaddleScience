import paddle

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
        self.loss_g = []  # eq loss
        self.loss_b = []  # boundary loss
        self.iter = 0
        self.a_g = paddle.to_tensor(1.0)
        self.a_b = paddle.to_tensor(1.0)
        self.his_a_g = []
        self.his_a_b = []
        self.optimizer = paddle.optimizer.Adam(
            learning_rate=0.001, parameters=self.pirbn.parameters()
        )
        self.ntk_list = {}
        # Update loss by calculate ntk
        self.update_loss_by_ntk = True

        # For test
        # if self.pirbn.activation_function == "tanh":
        #     self.update_loss_by_ntk = False

    def Loss(self, x, y, a_g, a_b):
        tmp = self.pirbn(x)
        loss_g = 0.5 * paddle.mean(paddle.square(tmp[0] - y[0]))
        loss_b = 0.5 * paddle.mean(paddle.square(tmp[1]))
        if self.update_loss_by_ntk:
            loss = loss_g * a_g + loss_b * a_b
        else:
            loss = loss_g + loss_b
        return loss, loss_g, loss_b

    def evaluate(self):
        # compute loss
        loss, loss_g, loss_b = self.Loss(self.x_train, self.y_train, self.a_g, self.a_b)
        loss_g_numpy = float(loss_g)
        loss_b_numpy = float(loss_b)
        # eq loss
        self.loss_g.append(loss_g_numpy)
        # boundary loss
        self.loss_b.append(loss_b_numpy)
        if self.iter % 200 == 0:
            if self.update_loss_by_ntk:
                self.a_g, self.a_b, _ = self.pirbn.cal_ntk(self.x_train)
                print("\ta_g =", float(self.a_g), "\ta_b =", float(self.a_b))
                print(
                    "Iter: ", self.iter, "\tL1 =", loss_g_numpy, "\tL2 =", loss_b_numpy
                )
            else:
                a_g, a_b, _ = self.pirbn.cal_ntk(self.x_train)
                print("\ta_g =", float(a_g), "\ta_b =", float(a_b))
                print(
                    "Iter: ", self.iter, "\tL1 =", loss_g_numpy, "\tL2 =", loss_b_numpy
                )
        self.his_a_g.append(self.a_g)
        self.his_a_b.append(self.a_b)

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
