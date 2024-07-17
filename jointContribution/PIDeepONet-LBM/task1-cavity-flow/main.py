import time

import numpy as np
import paddle as pd
import paddle.nn.functional as F
from dataset import DataSet

pd.set_default_dtype("float32")

pd.seed(1234)
np.random.seed(1234)

# resolution
h = 65
w = 65

# output dimension of Branch/Trunk
p = 100
p1 = p // 2

# batch_size
bs = 32

# size of input for Trunk net
nx = h
x_num = nx * nx

# POD modes
modes = 5
out_dims = 2 * modes
# coeffs for POD
# layer_pod = [h, 64, 64, 2*modes]


def prediction(model, data, u_basis, v_basis, f_test, u_test, v_test):
    out_B = model(f_test)
    out_B_u, out_B_v = out_B[:, :modes], out_B[:, modes:]
    u_pred = pd.einsum("bi,ni->bn", out_B_u, u_basis)
    v_pred = pd.einsum("bi,ni->bn", out_B_v, v_basis)
    u_pred, v_pred = u_pred.numpy(), v_pred.numpy()
    u_temp, v_temp = np.tile(u_pred[:, :, None], [1, 1, 1]), np.tile(
        v_pred[:, :, None], [1, 1, 1]
    )
    u_pred, v_pred = data.decoder(u_temp, v_temp)
    err_u = np.mean(
        np.linalg.norm(u_pred - u_test, 2, axis=1) / np.linalg.norm(u_test, 2, axis=1)
    )
    err_v = np.mean(
        np.linalg.norm(v_pred - v_test, 2, axis=1) / np.linalg.norm(v_test, 2, axis=1)
    )
    return err_u, err_v


def main():
    data = DataSet(nx, bs, modes)
    u_basis, v_basis = data.PODbasis()
    u_basis = pd.to_tensor(u_basis)
    v_basis = pd.to_tensor(v_basis)

    ##paddle-Branch net
    model = pd.nn.Sequential(
        pd.nn.Linear(h, 64),
        pd.nn.Tanh(),
        pd.nn.Linear(64, 64),
        pd.nn.Tanh(),
        pd.nn.Linear(64, out_dims),
    )
    # optimizer
    opt = pd.optimizer.Adam(learning_rate=1.0e-3, parameters=model.parameters())

    model.train()

    x_test, f_test, u_test, v_test = data.testbatch()
    f_test = pd.to_tensor(f_test)
    n = 0
    nmax = 20000
    start_time = time.perf_counter()
    time_step_0 = time.perf_counter()
    while n <= nmax:

        x_train, f_train, u_train, v_train, _, _ = data.minibatch()
        f_train, u_train, v_train = (
            pd.to_tensor(f_train),
            pd.to_tensor(u_train),
            pd.to_tensor(v_train),
        )
        out_B = model(f_train)
        out_B_u, out_B_v = out_B[:, :modes], out_B[:, modes:]
        u_pred = pd.einsum("bi,ni->bn", out_B_u, u_basis)
        v_pred = pd.einsum("bi,ni->bn", out_B_v, v_basis)
        loss = F.mse_loss(u_pred, u_train[:, :, 0]) + F.mse_loss(
            v_pred, v_train[:, :, 0]
        )
        loss.backward()
        opt.step()
        opt.clear_grad()

        if n % 100 == 0:
            time_step_1000 = time.perf_counter()
            T = time_step_1000 - time_step_0
            err_u, err_v = prediction(
                model, data, u_basis, v_basis, f_test, u_test, v_test
            )
            # err_u, err_v = data_save.save(sess, x_pos, f_ph, u_ph, v_ph, u_pred, v_pred, data, num_test, h)
            print(
                "Step: %d, Loss: %.3e, err_u: %.3e, err_v: %.3e, Time (secs): %.3f"
                % (n, float(loss), err_u, err_v, T)
            )
            # print('Step: %d, Loss: %.3e, Time (secs): %.3f'%(n, float(loss), T))
            time_step_0 = time.perf_counter()

        n += 1

    stop_time = time.perf_counter()
    print("Training time (secs): %.3f" % (stop_time - start_time))

    start_time = time.perf_counter()
    err_u, err_v = prediction(model, data, u_basis, v_basis, f_test, u_test, v_test)
    stop_time = time.perf_counter()
    T = stop_time - start_time
    print("err_u: %.3e, err_v: %.3e, Inference time (secs): %.3f" % (err_u, err_v, T))


if __name__ == "__main__":
    main()
