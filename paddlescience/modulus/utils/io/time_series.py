import matplotlib.pyplot as plt


def plot_time_series(var, base_name, time_axis="step"):
    for plot_var in var.keys():
        if plot_var != time_axis:
            plt.plot(var[time_axis][:, 0], var[plot_var][:, 0], label=plot_var)
    plt.legend()
    plt.xlabel(time_axis)
    plt.savefig(base_name + ".png")
    plt.close()
