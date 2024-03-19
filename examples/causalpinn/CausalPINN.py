import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import Dataset

class DataGenerator(Dataset):
    def __init__(self, t0, t1, n_t=10, n_x=64):
        'Initialization'
        self.t0 = t0
        self.t1 = t1 + 0.01 * t1
        self.n_t = n_t
        self.n_x = n_x

    def __getitem__(self, index):
        'Generate one batch of data'
        batch = self.__data_generation()
        return batch

    def __data_generation(self):
        'Generates data containing batch_size samples'
        t_r = paddle.uniform(shape=(self.n_t,), min=self.t0, max=self.t1).sort()
        points = paddle.uniform(shape=(self.n_x, 2), min=0.0, max=2.0 * np.pi)
        x_r = paddle.tile(points[:, 0:1], (1, n_t)).reshape((-1,1))  # N x T
        y_r = paddle.tile(points[:, 1:2], (1, n_t)).reshape((-1,1)) # N x T
        t_r = (paddle.tile(t_r, (1, n_x)).T).reshape((-1,1))  # N x T

        t_r.stop_gradient=False
        x_r.stop_gradient=False
        y_r.stop_gradient=False
        batch = (t_r, x_r, y_r)
        return batch

class modified_MLP_II(paddle.nn.Layer):
    def __init__(self, layers, L_x=1.0, L_y=1.0, M_t=1, M_x=1, M_y=1, activation=nn.Tanh()):
        super(modified_MLP_II, self).__init__()
        self.w_x = paddle.to_tensor(2.0 * np.pi / L_x,dtype='float32')
        self.w_y = paddle.to_tensor(2.0 * np.pi / L_y,dtype='float32')
        self.k_x = paddle.arange(1, M_x + 1).astype('float32')
        self.k_y = paddle.arange(1, M_y + 1).astype('float32')
        k_xx, k_yy = paddle.meshgrid(self.k_x, self.k_y)
        self.k_x =self.k_x[:,None]
        self.k_y =self.k_y[:,None]
        self.k_xx = k_xx.flatten()[:,None]
        self.k_yy = k_yy.flatten()[:,None]
        self.M_t = M_t
        self.U1 = paddle.nn.Linear(in_features=layers[0], out_features=layers[1])
        self.U2 = paddle.nn.Linear(in_features=layers[0], out_features=layers[1])
        self.Ws = [paddle.nn.Linear(in_features=layers[i], out_features=layers[i + 1]) for i in
                   range(0, len(layers) - 2)]
        self.F = paddle.nn.Linear(in_features=layers[-2], out_features=layers[-1])
        self.activation = activation
        self.k_t = paddle.pow(paddle.to_tensor(10), paddle.arange(0, self.M_t + 1)).astype('float32')


    def forward(self, inputs):
        t = inputs[:,2:3]
        x = inputs[:,0:1]
        y = inputs[:,1:2]
        k_t = paddle.pow(paddle.to_tensor(10), paddle.arange(0, self.M_t + 1)).astype('float32')[:,None]
        inputs=paddle.concat([paddle.ones_like((t),dtype='float32'), t @ k_t.T,                                                      
                                    paddle.cos(x@self.k_x.T * self.w_x), paddle.cos(y@self.k_y.T * self.w_y),               
                                    paddle.sin(x@self.k_x.T * self.w_x), paddle.sin(y@self.k_y.T * self.w_y),               
                                    paddle.cos(x@self.k_xx.T * self.w_x) * paddle.cos(y@self.k_yy.T * self.w_y),            
                                    paddle.cos(x@self.k_xx.T * self.w_x) * paddle.sin(y@self.k_yy.T * self.w_y),            
                                    paddle.sin(x@self.k_xx.T * self.w_x) * paddle.cos(y@self.k_yy.T * self.w_y),            
                                    paddle.sin(x@self.k_xx.T * self.w_x) * paddle.sin(y@self.k_yy.T * self.w_y)], axis=1)
        U = self.activation(self.U1(inputs))
        V = self.activation(self.U2(inputs))
        for W in self.Ws:
            outputs = self.activation(W(inputs))
            inputs = paddle.multiply(outputs, U) + paddle.multiply(1 - outputs, V)
        outputs = self.F(inputs)
        return outputs

class PINN((paddle.nn.Layer)):
    def __init__(self, w_exact, layers, M_t, M_x, M_y, state0, t0, t1, n_t, x_star, y_star,x_starall, y_starall,t_starall, tol):
        super(PINN, self).__init__()
        self.w_exact = w_exact

        self.M_t = M_t
        self.M_x = M_x
        self.M_y = M_y

        # grid
        self.n_t = n_t
        self.t0 = t0
        self.t1 = t1
        eps = 0.01 * t1
        self.t = paddle.linspace(self.t0, self.t1 + eps, n_t)
        self.x_star = x_star
        self.y_star = y_star
        self.x_starall = x_starall
        self.y_starall = y_starall
        self.t_starall=t_starall

        # initial state
        self.state0 = state0

        self.tol = tol
        self.M = paddle.triu(paddle.ones((n_t, n_t)), diagonal=1).T

        self.network = modified_MLP_II(layers, L_x=2 * np.pi, L_y=2 * np.pi, M_t=M_t, M_x=M_x, M_y=M_y,
                                       activation=nn.Tanh())

        # Use optimizers to set optimizer initialization and update functions
        self.optimizer = paddle.optimizer.Adam(learning_rate=paddle.optimizer.lr.ExponentialDecay(1e-3, gamma=0.9),
                                               parameters=self.network.parameters())

        # Logger
        self.loss_log = []
        self.loss_ics_log = []
        self.loss_u0_log = []
        self.loss_v0_log = []
        self.loss_w0_log = []
        self.loss_bcs_log = []
        self.loss_res_w_log = []
        self.loss_res_c_log = []
        self.l2_error_log = []

    def residual_net(self, t, x, y):
        u_v= self.network(paddle.concat([x, y, t], 1))
        u = u_v[:, 0:1]
        v = u_v[:, 1:2]

        u_x = paddle.grad(u, x, create_graph=True)[0]
        u_y = paddle.grad(u, y, create_graph=True)[0]
        v_x = paddle.grad(v, x, create_graph=True)[0]
        v_y = paddle.grad(v, y, create_graph=True)[0]
        w = v_x - u_y

        w_t = paddle.grad(w, t, create_graph=True)[0]
        w_x = paddle.grad(w, x, create_graph=True)[0]
        w_y = paddle.grad(w, y, create_graph=True)[0]

        w_xx = paddle.grad(w_x, x, create_graph=False)[0]
        w_yy = paddle.grad(w_y, y, create_graph=False)[0]

        res_w = w_t + u * w_x + v * w_y - nu * (w_xx + w_yy)
        res_c = u_x + v_y

        return res_w, res_c

    def residuals_and_weights(self, tol, batch):
        t_r, x_r,y_r = batch
        loss_u0, loss_v0, loss_w0 = self.loss_ics()
        L_0 = 1e5 * (loss_u0 + loss_v0 + loss_w0)
        res_w_pred, res_c_pred = self.residual_net(t_r, x_r, y_r)
        L_t = paddle.mean(res_w_pred ** 2 + 100 * res_c_pred ** 2, axis=1)
        W = paddle.exp(- tol * (self.M @ (L_t.reshape((-1,32)).T) + L_0))
        W.stop_gradient=True
        return L_0, L_t, W

    def loss_ics(self):
        # Compute forward pass
        u_v = self.network(paddle.concat([self.x_star, self.y_star, paddle.zeros_like(self.x_star,dtype='float32')], 1))
        u0_pred = u_v[:, 0:1]
        v0_pred = u_v[:, 1:2]
        v_x = paddle.grad(v0_pred, self.x_star, create_graph=False)[0]
        u_y = paddle.grad(u0_pred, self.y_star, create_graph=False)[0]
        w0_pred = v_x - u_y
        # Compute loss
        loss_u0 = paddle.mean((u0_pred.flatten() - self.state0[0, :, :].flatten()) ** 2)
        loss_v0 = paddle.mean((v0_pred.flatten() - self.state0[1, :, :].flatten()) ** 2)
        loss_w0 = paddle.mean((w0_pred.flatten() - self.state0[2, :, :].flatten()) ** 2)
        return loss_u0, loss_v0, loss_w0

    def loss(self, batch):
        L_0, L_t, W = self.residuals_and_weights(self.tol, batch)
        # Compute loss
        loss = paddle.mean(W * (L_t.reshape((-1,32)).T) + L_0)
        return loss

    def compute_l2_error(self):
        u_v = self.network(paddle.concat([x_star, y_star,t_star[:num_step],], 1))
        u = u_v[:, 0:1]
        v = u_v[:, 1:2]

        u_y = paddle.grad(u, y_star, create_graph=False)[0]
        v_x = paddle.grad(v, x_star, create_graph=False)[0]
        w_pred = v_x - u_y
        l2_error = np.linalg.norm(w_pred - self.w_exact) / np.linalg.norm(self.w_exact)
        return l2_error

    # Optimize parameters in a loop
    def train(self, dataset, nIter=10000):
        res_data = iter(dataset)
        # Main training loop
        for it in range(nIter):
            batch = next(res_data)
            loss = self.loss(batch)
            loss.backward()
            self.optimizer.step()
            self.optimizer.clear_grad()

            if it % 1000 == 0:
                print("ite:{},loss:{:.3e}".format(it,loss.item()))
                # l2_error_value = self.compute_l2_error()

                _, _, W_value = self.residuals_and_weights(self.tol, batch)

                # self.l2_error_log.append(l2_error_value)

                if W_value.min() > 0.99:
                    break

np.random.seed(1234)

data = np.load('/home/aistudio/data/data262374/NS.npy', allow_pickle=True).item()
# Test data
sol = paddle.to_tensor(data['sol'])

t_star = paddle.to_tensor(data['t']).reshape((-1,1))
x_star = paddle.to_tensor(data['x'])
y_star = paddle.to_tensor(data['y'])
nu = paddle.to_tensor(data['viscosity'])
Nt=len(t_star)
Nx=len(x_star)
sol = sol
x_star, y_star=paddle.meshgrid(x_star,y_star)
x_star=x_star.reshape((-1,1))
y_star=y_star.reshape((-1,1))
x_starall = paddle.tile(x_star, (1, Nt)).reshape((-1,1))  # N x T
y_starall = paddle.tile(y_star, (1, Nt)).reshape((-1,1)) # N x T
t_starall = (paddle.tile(t_star, (1, Nx**2)).T).reshape((-1,1))  # N x T

t_star.stop_gradient=False
x_star.stop_gradient=False
y_star.stop_gradient=False

# Create PINNs model
u0 = paddle.to_tensor(data['u0'])
v0 = paddle.to_tensor(data['v0'])
w0 = paddle.to_tensor(data['w0'])
state0 = paddle.stack([u0, v0, w0])
M_t = 2
M_x = 5
M_y = 5
d0 = 2 * M_x + 2 * M_y + 4 * M_x * M_y + M_t + 2
layers = [d0, 128, 128, 128, 128, 2]

num_step = 10
t0 = 0.0
t1 = t_star[num_step]
n_t = 32
tol = 1.0
tol_list = [1e-3, 1e-2, 1e-1, 1e0]

# Create data set
n_x = 256
dataset = DataGenerator(t0, t1, n_t, n_x)

N = 1 #20
w_pred_list = []
params_list = []
losses_list = []

# train
for k in range(N):
    # Initialize model
    print('Final Time: {}'.format(k + 1))
    w_exact = sol[num_step * k: num_step * (k + 1), :, :]
    model = PINN(w_exact, layers, M_t, M_x, M_y, state0, t0, t1, n_t, x_star, y_star,x_starall, y_starall,t_starall, tol)

    # Train
    for tol in tol_list:
        model.tol = tol
        print('tol:', model.tol)
        # Train
        model.train(dataset, nIter=100000)
    
obj = {'model': model.state_dict()}
path = '/home/aistudio/model.pdparams'
paddle.save(obj, path)