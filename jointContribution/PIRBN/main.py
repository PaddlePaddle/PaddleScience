import analytical_solution
import numpy as np
import pirbn
import rbn_net
import train


# mu, Fig.1, Page5
# right_by, Formula (15) Page5
def sine_function_main(mu, right_by=0, activation_function="gaussian_function"):
    # Define the number of sample points
    ns = 51

    # Define the sample points' interval
    dx = 1.0 / (ns - 1)

    # Initialise sample points' coordinates
    xy = np.zeros((ns, 1)).astype(np.float32)
    for i in range(0, ns):
        xy[i, 0] = i * dx + right_by
    xy_b = np.array([[right_by + 0.0], [right_by + 1.0]])

    x = [xy, xy_b]
    y = [-4 * mu**2 * np.pi**2 * np.sin(2 * mu * np.pi * xy)]

    # Set up radial basis network
    n_in = 1
    n_out = 1
    n_neu = 61
    b = 10.0
    c = [right_by - 0.1, right_by + 1.1]

    # Set up PIRBN
    rbn = rbn_net.RBN_Net(n_in, n_out, n_neu, b, c)
    train_obj = train.Trainer(
        pirbn.PIRBN(rbn, activation_function), x, y, learning_rate=0.001, maxiter=20001
    )
    train_obj.fit()

    # Visualise results
    analytical_solution.output_fig(train_obj, mu, b, right_by, activation_function)


# # Fig.1
# sine_function_main(mu=4, right_by=0, activation_function="tanh")
# # Fig.2
# sine_function_main(mu=8, right_by=0, activation_function="tanh")
# # Fig.3
# sine_function_main(mu=4, right_by=100, activation_function="tanh")
# Fig.6
sine_function_main(mu=8, right_by=100, activation_function="gaussian_function")
