import paddle

paddle.framework.core.set_prim_eager_enabled(True)


class Trainer:
    def __init__(
        self,
        pirbn,
        x_train,
        y_train,
        learning_rate=0.001,
        maxiter=10000,
        adaptive_weights=True,
    ):
        # set attributes
        self.pirbn = pirbn

        self.learning_rate = learning_rate
        self.x_train = [
            paddle.to_tensor(x, dtype=paddle.get_default_dtype()) for x in x_train
        ]
        self.y_train = paddle.to_tensor(y_train, dtype=paddle.get_default_dtype())

        # Normalize x
        # self.mu_X, self.sigma_X = self.x_train[0].mean(0), self.x_train[0].std(0)
        # self.mu_x, self.sigma_x = self.mu_X[0], self.sigma_X[0]
        # self.X_u = (self.x_train[1] - self.mu_X) / self.sigma_X
        # self.X_r = (self.x_train[0] - self.mu_X) / self.sigma_X
        # self.x_train = [self.X_r, self.X_u]

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
        self.adaptive_weights = (
            adaptive_weights  # Adaptive weights for physics-informed neural networks
        )

    def Loss(self, x, y, a_g, a_b):
        u_xx, u_b, _ = self.pirbn(x)
        loss_g = 0.5 * paddle.mean(paddle.square(u_xx - y))
        loss_b = 0.5 * paddle.mean(paddle.square(u_b))
        if self.adaptive_weights:
            loss = loss_g * a_g + loss_b * a_b
        else:
            loss = loss_g + 100 * loss_b
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
        if self.iter % 100 == 0:
            if self.adaptive_weights:
                self.a_g, self.a_b, _ = self.pirbn.cal_ntk(self.x_train)
                print(
                    "Iter : ",
                    self.iter,
                    "\tloss : ",
                    float(loss),
                    "\tboundary loss : ",
                    float(loss_b),
                    "\teq loss : ",
                    float(loss_g),
                )
                print("\ta_g =", float(self.a_g), "\ta_b =", float(self.a_b))
            else:
                print(
                    "Iter : ",
                    self.iter,
                    "\tloss : ",
                    float(loss),
                    "\tboundary loss : ",
                    float(loss_b),
                    "\teq loss : ",
                    float(loss_g),
                )
        self.his_a_g.append(self.a_g)
        self.his_a_b.append(self.a_b)

        self.iter = self.iter + 1
        return loss

    def fit(self, output_Kgg):
        for i in range(0, self.maxiter):
            loss = self.evaluate()
            loss.backward()
            if i in output_Kgg:
                self.ntk_list[f"{i}"] = self.pirbn.cal_K(self.x_train)
            self.optimizer.step()
            self.optimizer.clear_grad()
