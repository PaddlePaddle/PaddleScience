import numpy as np

if __name__ == "__main__":
    R = 0.05  # radius of pipe
    RHO = 1  # density
    L = 1.0  # length of pipe
    nuMean = 0.001  # average viscosity
    nuStd = 0.9
    dP = 0.1

    X_IN = 0  # inlet position of pipe
    X_OUT = X_IN + L  # outlet position of pipe

    yStart = -R
    yEnd = yStart + 2 * R

    nuStart = nuMean - nuMean * nuStd  # 0.0001
    nuEnd = nuMean + nuMean * nuStd  # 0.1

    N_x = 10
    N_y = 50
    N_p = 50
    N_pTest = 500
    uSolaM = np.zeros([N_y, N_x, N_p])
    data_1d_x = np.linspace(X_IN, X_OUT, N_x, endpoint=True)
    data_1d_y = np.linspace(yStart, yEnd, N_y, endpoint=True)
    data_1d_nu = np.linspace(nuStart, nuEnd, N_p, endpoint=True)
    data_1d_nuDist = np.random.normal(nuMean, 0.2 * nuMean, N_pTest)

    for i in range(N_p):
        uy = (R**2 - data_1d_y**2) * dP / (2 * L * data_1d_nu[i] * RHO)
        uSolaM[:, :, i] = np.tile(uy.reshape([N_y, 1]), N_x)

    uMax_a = np.zeros([1, N_pTest])
    for i in range(N_pTest):
        uMax_a[0, i] = (R**2) * dP / (2 * L * data_1d_nuDist[i] * RHO)
