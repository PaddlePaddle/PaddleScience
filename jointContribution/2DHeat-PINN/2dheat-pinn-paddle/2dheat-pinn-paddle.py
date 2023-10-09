# 导入依赖
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import qmc
import paddle
from paddle import nn

print(paddle.__version__)
#paddle.utils.run_check()
paddle.set_default_dtype("float64")

# For the use of the second derivative: paddle.cos, paddle.exp, see [4]
paddle.framework.core.set_prim_eager_enabled(True)

### 生成数据
n_bc = 4
n_data_per_bc = 25

#
engine = qmc.LatinHypercube(d=1)
data = np.zeros([4, 25, 3])

for i, j in zip(range(n_bc), [-1, +1, -1, +1]):
    points = (engine.random(n=n_data_per_bc)[:, 0] - 0.5) * 2
    if i < 2:
        data[i, :, 0] = j
        data[i, :, 1] = points
    else:
        data[i, :, 0] = points
        data[i, :, 1] = j

# 定义模型
from paddle.nn import Layer
from paddle import Tensor

class Input(Layer):
    def __init__(self, shape : tuple, device=paddle.device.get_device(), dtype=paddle.float32):
        
        super().__init__()

        self.shape = shape
        self.device = device
        self.dtype = dtype

    def forward(self, x):
        tensor_shape = tuple(x.shape[1:])

        #if x.device != self.device:
        #    raise ValueError(f"Input tensor must be on device {self.device} but got device {x.device} instead")
        if x.dtype != self.dtype:
            raise ValueError(f"Input tensor must have data type {self.dtype} but got data type {x.dtype} instead")
        if tensor_shape != self.shape:
            raise ValueError(f"Input shape must be {self.shape} but got {tensor_shape} instead")
        
        #return (x.to(self.device, self.dtype))
        return paddle.clone(x)

### 创建模型   
class TestModel(Layer):
    def __init__(self, in_shape=2, out_shape=1, n_hidden_layers=10, neuron_per_layer=20, actfn="tanh"):
        super().__init__()
        # input layer
        #self.input_layer = Input(shape=(in_shape,))
        # hidden layers
        #self.hidden = [nn.Linear(in_features=in_shape, out_features=neuron_per_layer), nn.Tanh()]
        #for i in range(n_hidden_layers-1):
        #    new_layer = nn.Linear(in_features=neuron_per_layer, out_features=neuron_per_layer)
        #    act_layer = nn.Tanh()
        #    self.hidden.append(new_layer)
        #    self.hidden.append(act_layer)
        # output layer
        #self.output_layer = nn.Linear(in_features=neuron_per_layer, out_features=out_shape)

        self.layers = nn.Sequential(
            Input(shape=(in_shape,), dtype=paddle.float64),
            nn.Linear(in_features=in_shape, out_features=neuron_per_layer), 
            nn.Tanh(),
            nn.Linear(in_features=neuron_per_layer, out_features=neuron_per_layer),
            nn.Tanh(),
            nn.Linear(in_features=neuron_per_layer, out_features=neuron_per_layer),
            nn.Tanh(),
            nn.Linear(in_features=neuron_per_layer, out_features=neuron_per_layer),
            nn.Tanh(),
            nn.Linear(in_features=neuron_per_layer, out_features=neuron_per_layer),
            nn.Tanh(),
            nn.Linear(in_features=neuron_per_layer, out_features=neuron_per_layer),
            nn.Tanh(),
            nn.Linear(in_features=neuron_per_layer, out_features=neuron_per_layer),
            nn.Tanh(),
            nn.Linear(in_features=neuron_per_layer, out_features=neuron_per_layer),
            nn.Tanh(),
            nn.Linear(in_features=neuron_per_layer, out_features=neuron_per_layer),
            nn.Tanh(),
            nn.Linear(in_features=neuron_per_layer, out_features=out_shape)
        )

    def u(self, x, y):
        #x = self.input_layer(paddle.concat([x, y], axis=1))
        #for hl in self.hidden:
        #    x = hl(x)
        #u = self.output_layer(x)
        u = self.layers(paddle.concat([x, y], axis=1))
        return u

    def f(self, x, y):
        # Gradients need to be calculated
        x.stop_gradient = False
        y.stop_gradient = False

        u0 = self.u(x, y)
        u_x = paddle.grad(u0, x, retain_graph=True, create_graph=True)[0]
        u_y = paddle.grad(u0, y, retain_graph=True, create_graph=True)[0]
        u_xx = paddle.grad(u_x, x, retain_graph=True, create_graph=True)[0]
        u_yy = paddle.grad(u_y, y, retain_graph=True, create_graph=True)[0]
        F = u_xx + u_yy
        #return paddle.mean(paddle.square(F))
        return F
    
    def mse(self, y, y_):
        return paddle.mean(paddle.square(y-y_))

    def preprocess_data(self, dataset):
        x_c, y_c, x_d, y_d, t_d = dataset
        self.x_c = paddle.to_tensor(x_c, dtype=paddle.float64)
        self.y_c = paddle.to_tensor(y_c, dtype=paddle.float64)
        self.x_d = paddle.to_tensor(x_d, dtype=paddle.float64)
        self.y_d = paddle.to_tensor(y_d, dtype=paddle.float64)
        self.t_d = paddle.to_tensor(t_d, dtype=paddle.float64)

    def forward(self, dataset):
        self.preprocess_data(dataset)
        T_ = self.u(self.x_d, self.y_d)
        L  = paddle.mean(paddle.square(self.f(self.x_c, self.y_c)))
        l  = self.mse(self.t_d, T_)
        loss = l+L
        return loss
        #return T_


# BC Values
# normalized in [0, 1]
data[0, :, 2] = 1.
data[2, :, 2] = 50/75

data = data.reshape(n_data_per_bc * n_bc, 3)
#
x_d, y_d, t_d = map(lambda x: np.expand_dims(x, axis=1), 
                    [data[:, 0], data[:, 1], data[:, 2]])

#

Nc = 10000
engine = qmc.LatinHypercube(d=2)
colloc = engine.random(n=Nc)
colloc = 2 * (colloc -0.5)
#
x_c, y_c = map(lambda x: np.expand_dims(x, axis=1), 
               [colloc[:, 0], colloc[:, 1]])

#

plt.title("Boundary Data points and Collocation points")
plt.scatter(data[:, 0], data[:, 1], marker="x", c="k", label="BDP")
plt.scatter(colloc[:, 0], colloc[:, 1], s=.2, marker=".", c="r", label="CP")
plt.show()

#

x_c, y_c, x_d, y_d, t_d =map(lambda x: paddle.to_tensor(x,dtype=paddle.float64), [x_c, y_c, x_d, y_d, t_d])
#x_c, y_c, x_d, y_d, t_d =map(lambda x: paddle.to_tensor(x,dtype=paddle.float32), [x_c, y_c, x_d, y_d, t_d])

# 创建模型
model = TestModel(2, 1, 9, 20, "tanh")
dataset = ( x_c, y_c, x_d, y_d, t_d )

# 训练
loss = 0
epochs = 1000
opt = paddle.optimizer.Adam(learning_rate=5e-4, parameters=model.parameters())
epoch = 0
loss_values = np.array([])
#
start = time.time()
#
for epoch in range(epochs):
    # forward pass and loss calculation 
    # implicit tape-based AD 
    loss = model(dataset)
   
    # compute gradients (grad)
    loss.backward()
    # update training variables / parameters
    opt.step()
    opt.clear_grad()

    loss_values = np.append(loss_values, loss)
    if epoch % 100 == 0 or epoch == epochs-1:
        print(f"{epoch:5}, {loss}")

#
end = time.time()
computation_time = {}
computation_time["pinn"] = end - start
print(f"\ncomputation time: {end-start:.3f}\n")
#
plt.semilogy(loss_values)
plt.legend()
plt.plot(loss_values)
plt.savefig("loss.png")
plt.show()

# 实现FDM
n = 100
l = 1.
r = 2*l/(n+1)
T = np.zeros([n*n, n*n])

bc = {
    "x=-l": 75.,
    "x=+l": 0.,
    "y=-l": 50.,
    "y=+l": 0.
}

B = np.zeros([n, n])
k = 0
for i in range(n):
    x = i * r
    for j in range(n):
        y = j * r
        M = np.zeros([n, n])
        M[i, j] = -4
        if i != 0: # ok i know
            M[i-1, j] = 1
        else:
            B[i, j] += -bc["y=-l"]   # b.c y = 0
        if i != n-1:
            M[i+1, j] = 1
        else:
            B[i, j] += -bc["y=+l"]   # b.c y = l
        if j != 0:
            M[i, j-1] = 1
        else:
            B[i, j] += -bc["x=-l"]   # b.c x = 0
        if j != n-1:
            M[i, j+1] = 1
        else:
            B[i, j] += -bc["x=+l"]   # b.c x = l
        #B[i, j] += -r**2 * q(x, y) * K(x, y)
        m = np.reshape(M, (1, n**2))
        T[k, :] = m
        k += 1

#
b = np.reshape(B, (n**2, 1))
start = time.time()
T = np.matmul(np.linalg.inv(T), b)
T = T.reshape([n, n])
Temperature = T
end = time.time()
computation_time["fdm"] = end - start
print(f"\ncomputation time: {end-start:.3f}\n")

### 绘图
plt.figure("", figsize=(12, 6))
#
X = np.linspace(-1, +1, n)
Y = np.linspace(-1, +1, n)
X0, Y0 = np.meshgrid(X, Y)
X = X0.reshape([n*n, 1])
Y = Y0.reshape([n*n, 1])
X_T = paddle.to_tensor(X)
Y_T = paddle.to_tensor(Y)
# 用PINN计算
S = model.u(X_T, Y_T)
S = S.numpy().reshape(n, n)
#
plt.subplot(221)
plt.pcolormesh(X0, Y0, 75.*S, cmap="magma")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("PINN")
plt.tight_layout()
plt.axis("square")
#
x = np.linspace(-1, +1, n)
y = np.linspace(-1, +1, n)
x, y = np.meshgrid(x, y)
#
plt.subplot(222)
plt.pcolormesh(x, y, T, cmap="magma")
plt.colorbar()
plt.title(r"FDM")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-1, +1)
plt.ylim(-1, +1)
plt.tight_layout()
plt.axis("square")
plt.savefig("heat01.png")
#
plt.subplot(223)
pinn_grad = np.gradient(np.gradient(S, axis=0), axis=1)
sigma_pinn = (pinn_grad**2).mean()
plt.pcolormesh(X0, Y0, pinn_grad, cmap="jet")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"" + f"\nmean squared: {sigma_pinn: .3e}")
plt.tight_layout()
plt.axis("square")
###
x = np.linspace(-1, +1, n)
y = np.linspace(-1, +1, n)
x, y = np.meshgrid(x, y)
#
plt.subplot(224)
fdm_grad = np.gradient(np.gradient(T, axis=0), axis=1)
sigma_fdm = (fdm_grad**2).mean()
plt.pcolormesh(x, y, fdm_grad, cmap="jet")
plt.colorbar()
plt.title(r"" + f"\nmean squared: {sigma_fdm: .3e}")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-1, +1)
plt.ylim(-1, +1)
plt.tight_layout()
plt.axis("square")
plt.savefig("heat01.png")
plt.show()

print("performance comparison".center(26))
print("="*26)
for method in computation_time:
    print(f"{method}\t\t{computation_time[method]:6.2f} (s)")

S_ = S.reshape([n, n])
T_ = T.reshape([n, n])

height = 3
frames_val = np.array([-.75, -.5, -.25, 0., +.25, +.5, +.75])
frames = [*map(int, (frames_val + 1)/2 * (n-1))]
fig = plt.figure("", figsize=(len(frames)*height, 2*height))

for i, var_index in enumerate(frames):
    plt.subplot(2, len(frames), i+1)
    plt.title(f"y = {frames_val[i]:.2f}")
    plt.plot(X0[var_index, :], 75.*S_[var_index,:], "r--", lw=4., label="pinn")
    plt.plot(X0[var_index, :], T_[var_index,:], "b", lw=2., label="FDM")
    plt.ylim(0., 100.)
    plt.xlim(-1., +1.)
    plt.xlabel("x")
    plt.ylabel("T")
    plt.tight_layout()
    plt.legend()

for i, var_index in enumerate(frames):
    plt.subplot(2, len(frames), len(frames) + i+1)
    plt.title(f"x = {frames_val[i]:.2f}")
    plt.plot(Y0[:, var_index], 75.*S_[:,var_index], "r--", lw=4., label="pinn")
    plt.plot(Y0[:, var_index], T_[:,var_index], "b", lw=2., label="FDM")
    plt.ylim(0., 100.)
    plt.xlim(-1., +1.)
    plt.xlabel("y")
    plt.ylabel("T")
    plt.tight_layout()
    plt.legend()

plt.savefig("profiles.png")
plt.show()

# 保存模型
obj = {'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': 1000}
path = 'model.pdparams'
paddle.save(obj, path)