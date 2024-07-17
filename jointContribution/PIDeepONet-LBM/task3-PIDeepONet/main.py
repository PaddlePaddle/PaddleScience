import time

import numpy as np
import paddle as pd
import paddle.nn.functional as F
from dataset import DataSet

pd.set_default_dtype("float32")

# from nn import DeepONet

pd.seed(1234)
np.random.seed(1234)

pd.device.set_device("gpu:0")

# resolution
h = 65
w = 65

# output dimension of Branch/Trunk
p = 100
p1 = p // 2

# batch_size
bs = 10

# size of input for Trunk net
nx = h
x_num = nx * nx

# POD modes
modes = 20
out_dims = 50
wh = 100
# coeffs for POD
# layer_pod = [h, 64, 64, 2*modes]


class DeepONet(pd.nn.Layer):
    def __init__(self):
        super(DeepONet, self).__init__()

        ##paddle-Branch net
        self.fnn_B = pd.nn.Sequential(
            pd.nn.Linear(h, wh),
            pd.nn.Tanh(),
            pd.nn.Linear(wh, wh),
            pd.nn.Tanh(),
            pd.nn.Linear(wh, out_dims),
        )

        ##paddle-Trunk net
        self.fnn_T = pd.nn.Sequential(
            pd.nn.Linear(2, wh),
            pd.nn.Tanh(),
            pd.nn.Linear(wh, wh),
            pd.nn.Tanh(),
            pd.nn.Linear(wh, wh),
            pd.nn.Tanh(),
            pd.nn.Linear(wh, out_dims),
        )

    def forward(self, Bin, Tin):
        out_B, out_T = self.fnn_B(Bin), self.fnn_T(Tin)
        out_B = pd.tile(out_B[:, None, :], repeat_times=[1, Tin.shape[1], 1])
        out_B_u, out_B_v, out_B_p = (
            out_B[:, :, :modes],
            out_B[:, :, modes : 2 * modes],
            out_B[:, :, 2 * modes :],
        )
        out_T_u, out_T_v, out_T_p = (
            out_T[:, :, :modes],
            out_T[:, :, modes : 2 * modes],
            out_T[:, :, 2 * modes :],
        )
        u_pred = pd.sum(out_B_u * out_T_u, axis=-1, keepdim=True)
        v_pred = pd.sum(out_B_v * out_T_v, axis=-1, keepdim=True)
        p_pred = pd.sum(out_B_p * out_T_p, axis=-1, keepdim=True)
        """
        u_pred = pd.einsum('bi,ni->bn', out_B_u, out_T_u)
        v_pred = pd.einsum('bi,ni->bn', out_B_v, out_T_v)
        """
        return u_pred, v_pred, p_pred

    def eq(self, Bin, Tin, rRe, data):
        u_pred, v_pred, p_pred = self.forward(Bin, Tin)
        # u_pred, v_pred = data.decoder(u_pred, v_pred)
        du = pd.grad(u_pred, Tin, create_graph=True, retain_graph=True)[0]
        dv = pd.grad(v_pred, Tin, create_graph=True, retain_graph=True)[0]
        dp = pd.grad(p_pred, Tin, create_graph=True, retain_graph=True)[0]
        u_x, u_y = du[:, :, 0:1], du[:, :, 1:2]
        v_x, v_y = dv[:, :, 0:1], dv[:, :, 1:2]
        p_x, p_y = dp[:, :, 0:1], dp[:, :, 1:2]
        ddux = pd.grad(u_x, Tin, create_graph=True, retain_graph=True)[0]
        dduy = pd.grad(u_y, Tin, create_graph=True, retain_graph=True)[0]
        ddvx = pd.grad(v_x, Tin, create_graph=True, retain_graph=True)[0]
        ddvy = pd.grad(v_y, Tin, create_graph=True, retain_graph=True)[0]
        u_xx, u_yy = ddux[:, :, 0:1], dduy[:, :, 1:2]
        v_xx, v_yy = ddvx[:, :, 0:1], ddvy[:, :, 1:2]
        eq1 = u_x + v_y
        eq2 = u_pred * u_x + v_pred * u_y - rRe * (u_xx + u_yy) + p_x
        eq3 = u_pred * v_x + v_pred * v_y - rRe * (v_xx + v_yy) + p_y
        return eq1, eq2, eq3


def prediction(model, data, x_test, f_test, u_test, v_test):
    """
    out_B, out_T = model.forward(f_test, x_test)
    out_B_u, out_B_v = out_B[:, :modes], out_B[:, modes:]
    out_T_u, out_T_v = out_T[:, :modes], out_T[:, modes:]
    u_pred = pd.einsum('bi,ni->bn', out_B_u, out_T_u)
    v_pred = pd.einsum('bi,ni->bn', out_B_v, out_T_v)
    """
    u_pred, v_pred, _ = model.forward(f_test, x_test)
    u_pred, v_pred = u_pred.numpy(), v_pred.numpy()
    # u_temp, v_temp = np.tile(u_pred[:, :, None], [1, 1, 1]), np.tile(v_pred[:, :, None], [1, 1, 1])
    # u_pred, v_pred = data.decoder(u_temp, v_temp)
    # u_pred, v_pred = data.decoder(u_pred, v_pred)
    err_u = np.mean(
        np.linalg.norm(u_pred - u_test, 2, axis=1) / np.linalg.norm(u_test, 2, axis=1)
    )
    err_v = np.mean(
        np.linalg.norm(v_pred - v_test, 2, axis=1) / np.linalg.norm(v_test, 2, axis=1)
    )
    return err_u, err_v


def main():
    data = DataSet(nx, bs, modes)
    """
    u_basis, v_basis = data.PODbasis()
    u_basis = pd.to_tensor(u_basis)
    v_basis = pd.to_tensor(v_basis)
    """
    model = DeepONet()
    x_train, x_eq_train, f_train, rRe_train, u_train, v_train, _, _ = data.minibatch()
    x_train, x_eq_train, f_train, rRe_train, u_train, v_train = (
        pd.to_tensor(x_train),
        pd.to_tensor(x_eq_train),
        pd.to_tensor(f_train),
        pd.to_tensor(rRe_train),
        pd.to_tensor(u_train),
        pd.to_tensor(v_train),
    )
    x_train = pd.tile(x_train[None, :, :], repeat_times=[bs, 1, 1])
    x_eq_train = pd.tile(x_eq_train[None, :, :], repeat_times=[bs, 1, 1])
    """
    x_num = x_train.shape[0]
    x_train = pd.tile(x_train[None, :, :], repeat_times=[bs, 1, 1])
    f_train = pd.tile(f_train[:, None, :], repeat_times=[1, x_num, 1])
    """
    x_train.stop_gradient = False
    x_eq_train.stop_gradient = False

    # optimizer
    opt = pd.optimizer.Adam(learning_rate=5.0e-4, parameters=model.parameters())

    model.train()

    x_test, f_test, u_test, v_test = data.testbatch()
    x_test, f_test = pd.to_tensor(x_test), pd.to_tensor(f_test)
    x_test = pd.tile(x_test[None, :, :], repeat_times=[f_test.shape[0], 1, 1])
    n = 0

    nmax = 150000
    start_time = time.perf_counter()
    time_step_0 = time.perf_counter()
    while n <= nmax:

        _, _, f_train, rRe_train, u_train, v_train, _, _ = data.minibatch()
        f_train, rRe_train, u_train, v_train = (
            pd.to_tensor(f_train),
            pd.to_tensor(rRe_train),
            pd.to_tensor(u_train),
            pd.to_tensor(v_train),
        )
        """
        out_B = fnn_B(f_train)
        out_T = fnn_T(x_train)
        out_B, out_T = model.forward(f_train, x_train)
        """
        """
        out_B = pd.tile(out_B[:. None, :], repeat_time=[1, x_num, 1])
        out_T = pd.tile(out_T[None, :, :], repeat_time=[bs, 1, 1])
        """
        """
        out_B_u, out_B_v = out_B[:, :modes], out_B[:, modes:]
        out_T_u, out_T_v = out_T[:, :modes], out_T[:, modes:]
        """

        # u_pred = pd.einsum('bi,ni->bn', out_B_u, out_T_u)
        # v_pred = pd.einsum('bi,ni->bn', out_B_v, out_T_v)
        u_pred, v_pred, _ = model.forward(f_train, x_train)
        eq1, eq2, eq3 = model.eq(f_train, x_eq_train, rRe_train, data)
        data_loss = F.mse_loss(u_pred, u_train) + F.mse_loss(v_pred, v_train)
        eq1_loss = F.mse_loss(eq1, pd.zeros(shape=eq1.shape))
        eq_loss = eq1_loss + 0.1 * (
            F.mse_loss(eq2, pd.zeros(shape=eq2.shape))
            + F.mse_loss(eq3, pd.zeros(eq3.shape))
        )
        loss = data_loss + eq_loss

        loss.backward()
        opt.step()
        opt.clear_grad()

        if n % 100 == 0:
            time_step_1000 = time.perf_counter()
            T = time_step_1000 - time_step_0
            err_u, err_v = prediction(model, data, x_test, f_test, u_test, v_test)
            # err_u, err_v = data_save.save(sess, x_pos, f_ph, u_ph, v_ph, u_pred, v_pred, data, num_test, h)
            print(
                "Step: %d, Loss: %.3e, err_u: %.3e, err_v: %.3e, Time (secs): %.3f"
                % (n, float(loss), err_u, err_v, T)
            )
            """
            print('Step: %d, Loss: %.3e, Loss(eq): %.3e, Loss(eq1): %.3e, err_u: %.3e, err_v: %.3e, Time (secs): %.3f'\
                   %(n, float(loss), float(eq_loss), float(eq1_loss), err_u, err_v, T))
            """
            # print('Step: %d, Loss: %.3e, Time (secs): %.3f'%(n, float(loss), T))
            time_step_0 = time.perf_counter()

        if n % 1000 == 0:
            pd.save(model.state_dict(), "./checkpoint/DeepONet.pdparams")
            pd.save(opt.state_dict(), "./checkpoint/opt.pdopt")

        n += 1

    stop_time = time.perf_counter()
    print("Training time (secs): %.3f" % (stop_time - start_time))

    start_time = time.perf_counter()
    err_u, err_v = prediction(model, data, x_test, f_test, u_test, v_test)
    stop_time = time.perf_counter()
    T = stop_time - start_time
    print("err_u: %.3e, err_v: %.3e, Inference time (secs): %.5f" % (err_u, err_v, T))


if __name__ == "__main__":
    main()
