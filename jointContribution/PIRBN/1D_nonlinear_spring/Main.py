import os

import matplotlib.pyplot as plt
import numpy as np
import OPT
import paddle
import PIRBN
import rbn_net
import scipy.io

### Define the number of sample points
ns = 1001

### Define the sample points' interval
dx = 100.0 / (ns - 1)

### Initialise sample points' coordinates
xy = np.zeros((ns, 1)).astype(np.float32)
for i in range(0, ns):
    xy[i, 0] = i * dx
xy_b = np.array([[0.0]])

x = [xy, xy_b]
y = [2 * np.cos(xy) + 3 * xy * np.sin(xy) + np.sin(xy * np.sin(xy))]

### Set up radial basis network
n_in = 1
n_out = 1
n_neu = 1021
b = 1.0
c = [-1.0, 101.0]

rbn = rbn_net.RBN_Net(n_in, n_out, n_neu, b, c)

### Set up PIRBN
pirbn = PIRBN.PIRBN(rbn)

### Train the PIRBN
opt = OPT.Adam(pirbn, x, y, learning_rate=0.001, maxiter=401)
result = opt.fit()

### Visualise results
ns = 1001
dx = 100 / (ns - 1)
xy = np.zeros((ns, 1)).astype(np.float32)
for i in range(0, ns):
    xy[i, 0] = i * dx
y = rbn(paddle.to_tensor(xy))
y = y.numpy()
plt.plot(xy, y)
plt.plot(xy, xy * np.sin(xy))
plt.legend(["predict", "ground truth"])
target_dir = os.path.join(os.path.dirname(__file__), "/../target")
if not os.path.exists(target_dir):
    os.path.mkdir(target_dir)
plt.savefig(os.path.join(target_dir, "1D_nonlinear_spring.png"))

### Save data
scipy.io.savemat(os.path.join(target_dir, "1D_nonlinear_spring.mat"), {"x": xy, "y": y})
