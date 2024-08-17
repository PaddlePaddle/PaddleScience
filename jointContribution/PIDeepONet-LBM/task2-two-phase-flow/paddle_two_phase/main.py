import argparse
import os
import time

import numpy as np
import paddle as pd
import paddle.nn.functional as F
import scipy.io as io
from dataset import DataSet

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default="0", type=str)
args = parser.parse_args()
device = args.local_rank

pd.set_default_dtype("float32")
os.environ["CUDA_VISIBLE_DEVICES"] = device

pd.seed(1234)
np.random.seed(1234)

in_dim = 1
# resolution
h = 200
s = 64

# output dimension of Branch/Trunk
p = 100
p1 = p // 2

# batch_size
bs = 32

# size of input for Trunk net
nx = in_dim
x_num = nx * nx

# POD modes
modes = 40
out_dims = modes
# coeffs for POD
# layer_pod = [h, 64, 64, 2*modes]


def prediction(model, data, u_basis, f_test, u_test):
    out_B = model(f_test)
    u_pred = pd.einsum("bi,ni->bn", out_B, u_basis)
    u_pred = pd.tanh(3.0 * u_pred)
    u_pred = u_pred.numpy()
    # u_temp = np.tile(u_pred[:, :, None], [1, 1, 1])
    u_test = u_test.reshape((-1, u_test.shape[1]))
    """
    u_temp, v_temp = np.tile(u_pred[:, :, None], [1, 1, 1]), np.tile(v_pred[:, :, None], [1, 1, 1])
    u_pred, v_pred = data.decoder(u_temp, v_temp)
    """
    err_u = np.mean(
        np.linalg.norm(u_pred - u_test, axis=1) / np.linalg.norm(u_test, axis=1)
    )
    u_pred, u_test = u_pred.reshape((-1, h, s)), u_test.reshape((-1, h, s))
    save_dict = {"u_pred": u_pred, "u_test": u_test}
    io.savemat("./Output/pred.mat", save_dict)
    # err_v = np.mean(np.linalg.norm(v_pred - v_test, 2, axis=1)/np.linalg.norm(v_test, 2, axis=1))
    return err_u, u_pred


def main():
    data = DataSet(nx, bs, modes)
    # _, _,_, _, _, u_basis, _ = data.load_data()
    """
    u_basis, v_basis = data.PODbasis()
    """
    u_basis = pd.to_tensor(data.u_basis)

    ##paddle-Branch net
    num_nodes = 64
    model = pd.nn.Sequential(
        pd.nn.Linear(in_dim, num_nodes),
        pd.nn.Tanh(),
        pd.nn.Linear(num_nodes, num_nodes),
        pd.nn.Tanh(),
        pd.nn.Linear(num_nodes, out_dims),
    )
    # optimizer
    opt = pd.optimizer.Adam(learning_rate=1.0e-3, parameters=model.parameters())

    model.train()

    x_test, f_test, u_test = data.testbatch()
    f_test = pd.to_tensor(f_test)
    n = 0
    nmax = 20000
    start_time = time.perf_counter()
    time_step_0 = time.perf_counter()
    while n <= nmax:

        x_train, f_train, u_train, _, _ = data.minibatch()
        f_train, u_train = pd.to_tensor(f_train), pd.to_tensor(u_train)
        out_B = model(f_train)
        out_B_u = out_B
        u_pred = pd.einsum("bi,ni->bn", out_B_u, u_basis)
        loss = F.mse_loss(u_pred, u_train[:, :, 0])
        loss.backward()
        opt.step()
        opt.clear_grad()

        if n % 100 == 0:
            time_step_1000 = time.perf_counter()
            T = time_step_1000 - time_step_0
            # err_u, _ = prediction(model, data, u_basis, f_test, u_test)
            # err_u, err_v = data_save.save(sess, x_pos, f_ph, u_ph, v_ph, u_pred, v_pred, data, num_test, h)
            # print('Step: %d, Loss: %.3e, err_u: %.3e, Time (secs): %.3f'%(n, float(loss), err_u, T))
            print("Step: %d, Loss: %.3e, Time (secs): %.3f" % (n, float(loss), T))
            time_step_0 = time.perf_counter()

        n += 1

    stop_time = time.perf_counter()
    print("Training time (secs): %.3f" % (stop_time - start_time))

    start_time = time.perf_counter()
    err_u, u_pred = prediction(model, data, u_basis, f_test, u_test)
    stop_time = time.perf_counter()
    T = stop_time - start_time
    print("err_u: %.3e, Inference time (secs): %.5f" % (err_u, T))


if __name__ == "__main__":
    main()
