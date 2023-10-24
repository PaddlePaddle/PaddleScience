import analytical_solution
import numpy as np
import pirbn
import rbn_net
import train

import ppsci

# set random seed for reproducibility
SEED = 2023
ppsci.utils.misc.set_random_seed(SEED)

# mu, Fig.1, Page5
# right_by, Formula (15) Page5
def sine_function_main(
    mu, adaptive_weights=True, right_by=0, activation_function="gaussian"
):
    # Define the number of sample points
    ns = 50

    # Define the sample points' interval
    dx = 1.0 / (ns - 1)

    # Initialise sample points' coordinates
    x_eq = np.linspace(0.0, 1.0, ns)[:, None]

    for i in range(0, ns):
        x_eq[i, 0] = i * dx + right_by
    x_bc = np.array([[right_by + 0.0], [right_by + 1.0]])
    x = [x_eq, x_bc]
    y = -4 * mu**2 * np.pi**2 * np.sin(2 * mu * np.pi * x_eq)

    # Set up radial basis network
    n_in = 1
    n_out = 1
    n_neu = 61
    b = 10.0
    c = [right_by - 0.1, right_by + 1.1]

    # Set up PIRBN
    rbn = rbn_net.RBN_Net(n_in, n_out, n_neu, b, c, activation_function)
    rbn_loss = pirbn.PIRBN(rbn, activation_function)
    maxiter = 20001
    output_Kgg = [0, int(0.1 * maxiter), maxiter - 1]
    train_obj = train.Trainer(
        rbn_loss,
        x,
        y,
        learning_rate=0.001,
        maxiter=maxiter,
        adaptive_weights=adaptive_weights,
    )
    train_obj.fit(output_Kgg)

    # Visualise results
    analytical_solution.output_fig(
        train_obj, mu, b, right_by, activation_function, output_Kgg
    )


# Fig.1
sine_function_main(mu=4, right_by=0, activation_function="tanh")
# Fig.2
sine_function_main(mu=8, right_by=0, activation_function="tanh")
# Fig.3
sine_function_main(mu=4, right_by=100, activation_function="tanh")
# Fig.6
sine_function_main(mu=8, right_by=100, activation_function="gaussian")
