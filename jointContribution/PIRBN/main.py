import analytical_solution
import numpy as np
import pirbn
import rbn_net
import train

# Define mu
mu = 4

# Define the number of sample points
ns = 51

# Define the sample points' interval
dx = 1.0 / (ns - 1)

# Initialise sample points' coordinates
xy = np.zeros((ns, 1)).astype(np.float32)
for i in range(0, ns):
    xy[i, 0] = i * dx
xy_b = np.array([[0.0], [1.0]])

x = [xy, xy_b]
y = [-4 * mu**2 * np.pi**2 * np.sin(2 * mu * np.pi * xy)]

# Set up radial basis network
n_in = 1
n_out = 1
n_neu = 61
b = 10.0
c = [-0.1, 1.1]

# Set up PIRBN
rbn = rbn_net.RBN_Net(n_in, n_out, n_neu, b, c)
train_obj = train.Trainer(pirbn.PIRBN(rbn), x, y, learning_rate=0.001, maxiter=20001)
train_obj.fit()

# Visualise results
analytical_solution.output_fig(train_obj, mu, b)
