"""
https://github.com/PredictiveIntelligenceLab/PINNsNTK/blob/18ef519e1fe924e32ef96d4ba9a814b480bd9b05/PINNsNTK_Poisson1D.ipynb
"""

import timeit

import matplotlib.pyplot as plt
import numpy as np
import paddle
from Compute_Jacobian import jacobian

paddle.framework.core.set_prim_eager_enabled(True)


class PINN(paddle.nn.Layer):
    def __init__(self, layers, X_u, Y_u, X_r, Y_r):
        super().__init__()
        X_u = paddle.to_tensor(X_u, dtype=paddle.get_default_dtype())
        Y_u = paddle.to_tensor(Y_u, dtype=paddle.get_default_dtype())
        X_r = paddle.to_tensor(X_r, dtype=paddle.get_default_dtype())
        Y_r = paddle.to_tensor(Y_r, dtype=paddle.get_default_dtype())

        self.mu_X, self.sigma_X = X_r.mean(0), X_r.std(0)
        self.mu_x, self.sigma_x = self.mu_X[0], self.sigma_X[0]

        # Normalize
        self.X_u = (X_u - self.mu_X) / self.sigma_X
        self.Y_u = Y_u
        self.X_r = (X_r - self.mu_X) / self.sigma_X
        self.Y_r = Y_r

        # Initialize network weights and biases
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # Define the size of the Kernel
        self.kernel_size = X_u.shape[0]

        self.x_u_tf = "x_u_tf"
        self.u_tf = "u_tf"

        self.x_bc_tf = "x_bc_tf"
        self.u_bc_tf = "u_bc_tf"

        self.x_r_tf = "x_r_tf"
        self.r_tf = "r_tf"

        self.x_u_ntk_tf = "x_u_ntk_tf"
        self.x_r_ntk_tf = "x_r_ntk_tf"

        self.dataset = {}

        # Define optimizer with learning rate schedule
        self.global_step = 0
        starter_learning_rate = 1e-5
        self.learning_rate = paddle.optimizer.lr.ExponentialDecay(
            starter_learning_rate, 0.9
        )
        # Passing global_step to minimize() will increment it at each step.
        # To compute NTK, it is better to use SGD optimizer
        # since the corresponding gradient flow is not exactly same.
        self.train_op = paddle.optimizer.SGD(
            learning_rate=starter_learning_rate, parameters=self.parameters()
        )

        # Logger
        # Loss logger
        self.loss_bcs_log = []
        self.loss_res_log = []

        # NTK logger
        self.K_uu_log = []
        self.K_rr_log = []
        self.K_ur_log = []

        # Weights logger
        self.weights_log = []
        self.biases_log = []

        self.is_train = False

    def forward(self, input):
        self.dataset.update(input)

        # Evaluate predictions
        self.u_bc_pred = self.net_u(self.dataset[self.x_bc_tf])

        self.u_pred = self.net_u(self.dataset[self.x_u_tf])
        self.r_pred = self.net_r(self.dataset[self.x_r_tf])

        if self.is_train:
            # Boundary loss
            self.loss_bcs = paddle.mean(
                paddle.square(self.u_bc_pred - self.dataset[self.u_bc_tf])
            )

            # Residual loss
            self.loss_res = paddle.mean(
                paddle.square(self.dataset[self.r_tf] - self.r_pred)
            )

            # Total loss
            self.loss = self.loss_res + self.loss_bcs

        if self.x_u_ntk_tf in self.dataset:
            self.u_ntk_pred = self.net_u(self.dataset[self.x_u_ntk_tf])
            self.r_ntk_pred = self.net_r(self.dataset[self.x_r_ntk_tf])

            # Compute the Jacobian for weights and biases in each hidden layer
            self.J_u = self.compute_jacobian(self.u_ntk_pred)
            self.J_r = self.compute_jacobian(self.r_ntk_pred)

            # The empirical NTK = J J^T, compute NTK of PINNs
            self.K_uu = self.compute_ntk(
                self.J_u,
                self.dataset[self.x_u_ntk_tf],
                self.J_u,
                self.dataset[self.x_u_ntk_tf],
            )
            self.K_ur = self.compute_ntk(
                self.J_u,
                self.dataset[self.x_u_ntk_tf],
                self.J_r,
                self.dataset[self.x_r_ntk_tf],
            )
            self.K_rr = self.compute_ntk(
                self.J_r,
                self.dataset[self.x_r_ntk_tf],
                self.J_r,
                self.dataset[self.x_r_ntk_tf],
            )

    # Xavier initialization
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = 1.0 / np.sqrt((in_dim + out_dim) / 2.0)
        return self.create_parameter(
            shape=[in_dim, out_dim],
            default_initializer=paddle.nn.initializer.Assign(
                paddle.normal(shape=[in_dim, out_dim]) * xavier_stddev
            ),
            dtype=paddle.get_default_dtype(),
        )

    # NTK initialization
    def NTK_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        std = 1.0 / np.sqrt(in_dim)
        return self.create_parameter(
            shape=[in_dim, out_dim],
            default_initializer=paddle.nn.initializer.Assign(
                paddle.normal(shape=[in_dim, out_dim]) * std
            ),
            dtype=paddle.get_default_dtype(),
        )

    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.NTK_init(size=[layers[l], layers[l + 1]])
            b = self.create_parameter(
                shape=[1, layers[l + 1]],
                default_initializer=paddle.nn.initializer.Assign(
                    paddle.normal(shape=[1, layers[l + 1]])
                ),
                dtype=paddle.get_default_dtype(),
            )
            weights.append(W)
            biases.append(b)
            self.add_parameter("W_{l}", W)
            self.add_parameter("b_{l}", b)
        return weights, biases

    # Evaluates the forward pass
    def forward_pass(self, H):
        num_layers = len(self.layers)
        for l in range(0, num_layers - 2):
            W = self.weights[l]
            b = self.biases[l]
            H = paddle.tanh(paddle.add(paddle.matmul(H, W), b))
        W = self.weights[-1]
        b = self.biases[-1]
        H = paddle.add(paddle.matmul(H, W), b)
        return H

    # Evaluates the PDE solution
    def net_u(self, x):
        x.stop_gradient = False
        u = self.forward_pass(x)
        return u

    # Forward pass for the residual
    def net_r(self, x):
        x.stop_gradient = False
        u = self.net_u(x)

        u_x = paddle.grad(u, x, retain_graph=True, create_graph=True)[0] / self.sigma_x
        u_xx = (
            paddle.grad(u_x, x, retain_graph=True, create_graph=True)[0] / self.sigma_x
        )

        res_u = u_xx
        return res_u

    # Compute Jacobian for each weights and biases in each layer and return a list
    def compute_jacobian(self, f):
        J_list = []
        L = len(self.weights)
        for i in range(L):
            J_w = jacobian(f, self.weights[i])
            J_list.append(J_w)

        for i in range(L):
            J_b = jacobian(f, self.biases[i])
            J_list.append(J_b)
        return J_list

    # Compute the empirical NTK = J J^T
    def compute_ntk(self, J1_list, x1, J2_list, x2):
        D = x1.shape[0]
        N = len(J1_list)

        Ker = paddle.zeros((D, D))
        for k in range(N):
            J1 = paddle.reshape(J1_list[k], shape=(D, -1))
            J2 = paddle.reshape(J2_list[k], shape=(D, -1))

            K = paddle.matmul(J1, paddle.transpose(J2, [1, 0]))
            Ker = Ker + K
        return Ker

    # Trains the model by minimizing the MSE loss
    def train(self, nIter=10000, batch_size=128, log_NTK=True, log_weights=True):
        self.is_train = True

        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch boundary mini-batches
            # Define a dictionary for associating placeholders with data
            tf_dict = {
                self.x_bc_tf: self.X_u,
                self.u_bc_tf: self.Y_u,
                self.x_u_tf: self.X_u,
                self.x_r_tf: self.X_r,
                self.r_tf: self.Y_r,
            }

            self.forward(tf_dict)

            # minimize the loss
            self.loss.backward()
            self.train_op.step()
            self.train_op.clear_grad()

            # Print
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time
                # self.forward(tf_dict)
                loss_value = self.loss
                loss_bcs_value, loss_res_value = [self.loss_bcs, self.loss_res]
                self.loss_bcs_log.append(loss_bcs_value.detach().clone())
                self.loss_res_log.append(loss_res_value.detach().clone())

                print(
                    "It: %d, Loss: %.3e, Loss_bcs: %.3e, Loss_res: %.3e ,Time: %.2f"
                    % (it, loss_value, loss_bcs_value, loss_res_value, elapsed)
                )

                start_time = timeit.default_timer()

            if log_NTK:
                # provide x, x' for NTK
                if it % 100 == 0:
                    print("Compute NTK...")
                    tf_dict = {self.x_u_ntk_tf: self.X_u, self.x_r_ntk_tf: self.X_r}
                    self.forward(tf_dict)
                    K_uu_value, K_ur_value, K_rr_value = [
                        self.K_uu,
                        self.K_ur,
                        self.K_rr,
                    ]
                    self.K_uu_log.append(K_uu_value.numpy())
                    self.K_ur_log.append(K_ur_value.numpy())
                    self.K_rr_log.append(K_rr_value.numpy())

            if log_weights:
                if it % 100 == 0:
                    print("Weights stored...")
                    weights = self.weights
                    biases = self.biases

                    self.weights_log.append([item.numpy() for item in weights])
                    self.biases_log.append([item.numpy() for item in biases])

    # Evaluates predictions at test points
    def predict_u(self, X_star):
        X_star = paddle.to_tensor(X_star, paddle.get_default_dtype())
        self.is_train = False
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x_u_tf: X_star}
        self.forward(tf_dict)
        u_star = self.u_pred
        u_star = u_star.numpy()
        return u_star

    # Evaluates predictions at test points
    def predict_r(self, X_star):
        X_star = paddle.to_tensor(X_star, paddle.get_default_dtype())
        self.is_train = False
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x_r_tf: X_star}
        self.forward(tf_dict)
        r_star = self.r_pred
        r_star = r_star.numpy()
        return r_star


# Define solution and its Laplace
a = 4


def u(x, a):
    return np.sin(np.pi * a * x)


def u_xx(x, a):
    return -((np.pi * a) ** 2) * np.sin(np.pi * a * x)


# Define computational domain
bc1_coords = np.array([[0.0], [0.0]])

bc2_coords = np.array([[1.0], [1.0]])

dom_coords = np.array([[0.0], [1.0]])

# Training data on u(x) -- Dirichlet boundary conditions

nn = 50

X_bc1 = dom_coords[0, 0] * np.ones((nn // 2, 1))
X_bc2 = dom_coords[1, 0] * np.ones((nn // 2, 1))
X_u = np.vstack([X_bc1, X_bc2])

Y_u = u(X_u, a)

X_r = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
Y_r = u_xx(X_r, a)

# Define model
layers = [1, 61, 1]
# layers = [1, 512, 512, 512, 1]
model = PINN(layers, X_u, Y_u, X_r, Y_r)

# Train model
model.train(nIter=2001, batch_size=50, log_NTK=True, log_weights=True)

loss_bcs = model.loss_bcs_log
loss_res = model.loss_res_log

fig = plt.figure(figsize=(6, 5))
plt.plot(loss_res, label="$\mathcal{L}_{r}$")
plt.plot(loss_bcs, label="$\mathcal{L}_{b}$")
plt.yscale("log")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

plt.savefig("paddle_loss.png")

nn = 1000
X_star = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
u_star = u(X_star, a)
r_star = u_xx(X_star, a)

# Predictions
u_pred = model.predict_u(X_star)
r_pred = model.predict_r(X_star)
error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
error_r = np.linalg.norm(r_star - r_pred, 2) / np.linalg.norm(r_star, 2)

print("Relative L2 error_u: {:.2e}".format(error_u))
print("Relative L2 error_r: {:.2e}".format(error_r))

fig = plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(X_star, u_star, label="Exact")
plt.plot(X_star, u_pred, "--", label="Predicted")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend(loc="upper right")

plt.subplot(1, 2, 2)
plt.plot(X_star, np.abs(u_star - u_pred), label="Error")
plt.yscale("log")
plt.xlabel("$x$")
plt.ylabel("Point-wise error")
plt.tight_layout()
plt.show()

plt.savefig("paddle_out1.png")

# Create empty lists for storing the eigenvalues of NTK
lambda_K_log = []
lambda_K_uu_log = []
lambda_K_ur_log = []
lambda_K_rr_log = []

# Restore the NTK
K_uu_list = model.K_uu_log
K_ur_list = model.K_ur_log
K_rr_list = model.K_rr_log
K_list = []

for k in range(len(K_uu_list)):
    K_uu = K_uu_list[k]
    K_ur = K_ur_list[k]
    K_rr = K_rr_list[k]

    K = np.concatenate(
        [np.concatenate([K_uu, K_ur], axis=1), np.concatenate([K_ur.T, K_rr], axis=1)],
        axis=0,
    )
    K_list.append(K)

    # Compute eigenvalues
    lambda_K, _ = np.linalg.eig(K)
    lambda_K_uu, _ = np.linalg.eig(K_uu)
    lambda_K_rr, _ = np.linalg.eig(K_rr)

    # Sort in decreasing order
    lambda_K = np.sort(np.real(lambda_K))[::-1]
    lambda_K_uu = np.sort(np.real(lambda_K_uu))[::-1]
    lambda_K_rr = np.sort(np.real(lambda_K_rr))[::-1]

    # Store eigenvalues
    lambda_K_log.append(lambda_K)
    lambda_K_uu_log.append(lambda_K_uu)
    lambda_K_rr_log.append(lambda_K_rr)

fig = plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
for i in range(1, len(lambda_K_log), 10):
    plt.plot(lambda_K_log[i], "--")
plt.xscale("log")
plt.yscale("log")
plt.title(r"Eigenvalues of ${K}$")
plt.tight_layout()

plt.subplot(1, 3, 2)
for i in range(1, len(lambda_K_uu_log), 10):
    plt.plot(lambda_K_uu_log[i], "--")
plt.xscale("log")
plt.yscale("log")
plt.title(r"Eigenvalues of ${K}_{uu}$")
plt.tight_layout()

plt.subplot(1, 3, 3)
for i in range(1, len(lambda_K_log), 10):
    plt.plot(lambda_K_rr_log[i], "--")
plt.xscale("log")
plt.yscale("log")
plt.title(r"Eigenvalues of ${K}_{rr}$")
plt.tight_layout()
plt.show()

plt.savefig("paddle_out2.png")

# Change of the NTK
NTK_change_list = []
K0 = K_list[0]
for K in K_list:
    diff = np.linalg.norm(K - K0) / np.linalg.norm(K0)
    NTK_change_list.append(diff)

fig = plt.figure(figsize=(6, 5))
plt.plot(NTK_change_list)

# Change of the weights and biases
def compute_weights_diff(weights_1, weights_2):
    weights = []
    N = len(weights_1)
    for k in range(N):
        weight = weights_1[k] - weights_2[k]
        weights.append(weight)
    return weights


def compute_weights_norm(weights, biases):
    norm = 0
    for w in weights:
        norm = norm + np.sum(np.square(w))
    for b in biases:
        norm = norm + np.sum(np.square(b))
    norm = np.sqrt(norm)
    return norm


# Restore the list weights and biases
weights_log = model.weights_log
biases_log = model.biases_log

weights_0 = weights_log[0]
biases_0 = biases_log[0]

# Norm of the weights at initialization
weights_init_norm = compute_weights_norm(weights_0, biases_0)

weights_change_list = []

N = len(weights_log)
for k in range(N):
    weights_diff = compute_weights_diff(weights_log[k], weights_log[0])
    biases_diff = compute_weights_diff(biases_log[k], biases_log[0])

    weights_diff_norm = compute_weights_norm(weights_diff, biases_diff)
    weights_change = weights_diff_norm / weights_init_norm
    weights_change_list.append(weights_change)

fig = plt.figure(figsize=(6, 5))
plt.plot(weights_change_list)

# Visualise NTK after initialisation, The normalised Kg at 0th iteration.
# Krr == Kgg

# Add empty data
len1 = len(K_rr_list)
for i in range(401 - len1):
    K_rr_list.append(None)

index = [0, 1, 2, 20, 100, 400]
K = [K_rr_list[i] for i in index]
index = [i * 100 for i in index]
plt.figure(figsize=(15, 9))
plt.subplot(2, 3, 1)
plt.imshow(K[0] / (np.max(abs(K[0]))), cmap="bwr", vmax=1, vmin=-1)
plt.colorbar()
plt.title(f"Kgg / max(Kgg) at {index[0]}-th iteration")
plt.xlabel("PDE Sample point index")

# When no data
try:
    plt.subplot(2, 3, 2)
    plt.imshow(K[1] / (np.max(abs(K[1]))), cmap="bwr", vmax=1, vmin=-1)
    plt.colorbar()
    plt.title(f"Kgg / max(Kgg) at {index[1]}-th iteration")
    plt.xlabel("PDE Sample point index")

    plt.subplot(2, 3, 3)
    plt.imshow(K[2] / (np.max(abs(K[2]))), cmap="bwr", vmax=1, vmin=-1)
    plt.colorbar()
    plt.title(f"Kgg / max(Kgg) at {index[2]}-th iteration")
    plt.xlabel("PDE Sample point index")

    plt.subplot(2, 3, 4)
    plt.imshow(K[3] / (np.max(abs(K[3]))), cmap="bwr", vmax=1, vmin=-1)
    plt.colorbar()
    plt.title(f"Kgg / max(Kgg) at {index[3]}-th iteration")
    plt.xlabel("PDE Sample point index")

    plt.subplot(2, 3, 5)
    plt.imshow(K[4] / (np.max(abs(K[4]))), cmap="bwr", vmax=1, vmin=-1)
    plt.colorbar()
    plt.title(f"Kgg / max(Kgg) at {index[4]}-th iteration")
    plt.xlabel("PDE Sample point index")

    plt.subplot(2, 3, 6)
    plt.imshow(K[5] / (np.max(abs(K[5]))), cmap="bwr", vmax=1, vmin=-1)
    plt.colorbar()
    plt.title(f"Kgg / max(Kgg) at {index[5]}-th iteration")
    plt.xlabel("PDE Sample point index")
except Exception:
    print("empty value")

plt.savefig("paddle_out3.png")
