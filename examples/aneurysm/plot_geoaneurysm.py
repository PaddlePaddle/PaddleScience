import os

import matplotlib.pyplot as plt
import numpy as np
import single_test

scale_test = np.load("aneurysm_scale0005to002_eval0to002mean001_3sigma.npz")["scale"]
os.chdir("/workspace/wangguan/PaddleScience_Surrogate/examples/aneurysm")
caseCount = [1.0, 151.0, 486.0]

W_ctl = np.zeros([len(scale_test), 1])
W_ctl_Ml = np.zeros([len(scale_test), 1])

# aneurysm
plot_x = 0.8
plot_y = 0.06
fontsize = 14
axis_limit = [0, 1, -0.15, 0.15]
path = "Cases/"
for caseIdx in caseCount:
    ## geo_case
    scale = scale_test[int(caseIdx - 1)]

    nu = 1e-3

    dP = 0.1
    mu = 0.5
    sigma = 0.1
    epochs = 500
    ## Aneurysm
    ##
    token = "aneurysm0005to002"
    token1 = "aneurysm0to002"

    ######################################################################
    # BELOW SHOULD REAMIN UNCHANGED!!!!
    ######################################################################

    print("path is", path + str(caseIdx))

    Data_CFD = np.load(path + str(caseIdx) + "CFD_contour.npz")
    Data_NN = np.load(path + str(caseIdx) + "NN_contour.npz")
    x = Data_CFD["x"]
    y = Data_CFD["y"]
    U_CFD = Data_CFD["U"]
    U = Data_NN["U"]

    device = "cpu"
    u, v, p = single_test.det_test(
        x, y, nu, dP, mu, sigma, scale, epochs, path, device, caseIdx
    )
    print("shape of u", u.shape)
    print("shape of v", v.shape)
    print("shape of p", p.shape)
    w = np.zeros_like(u)
    U = np.concatenate([u, v, w], axis=1)

    # Contour Comparison

    plt.figure()
    plt.subplot(212)
    plt.scatter(x, y, c=U[:, 0], vmin=min(U_CFD[:, 0]), vmax=max(U_CFD[:, 0]))
    plt.text(plot_x, plot_y, r"DNN", {"color": "b", "fontsize": fontsize})
    plt.axis(axis_limit)
    plt.colorbar()
    plt.subplot(211)
    plt.scatter(x, y, c=U_CFD[:, 0], vmin=min(U_CFD[:, 0]), vmax=max(U_CFD[:, 0]))
    plt.colorbar()
    plt.text(plot_x, plot_y, r"CFD", {"color": "b", "fontsize": fontsize})
    plt.axis(axis_limit)
    plt.savefig(
        "plot/" + str(int(caseIdx)) + "scale" + str(scale) + "uContour_test.png",
        bbox_inches="tight",
    )

    print(
        "path is",
        "plot/" + str(int(caseIdx)) + "scale" + str(scale) + "uContour_test.png",
    )

    plt.figure()
    plt.subplot(212)
    plt.scatter(x, y, c=U[:, 1], vmin=min(U_CFD[:, 1]), vmax=max(U_CFD[:, 1]))
    plt.text(plot_x, plot_y, r"DNN", {"color": "b", "fontsize": fontsize})
    plt.axis(axis_limit)
    plt.colorbar()
    plt.subplot(211)
    plt.scatter(x, y, c=U_CFD[:, 1], vmin=min(U_CFD[:, 1]), vmax=max(U_CFD[:, 1]))
    plt.colorbar()
    plt.text(plot_x, plot_y, r"CFD", {"color": "b", "fontsize": fontsize})
    plt.axis(axis_limit)
    plt.savefig(
        "plot/" + str(int(caseIdx)) + "scale" + str(scale) + "vContour_test.png",
        bbox_inches="tight",
    )

    plt.close("all")
    # plt.show()

    Data_CFD_wss = np.load(path + str(caseIdx) + "CFD_wss.npz")
    unique_x = Data_CFD_wss["x"]
    wall_shear_mag_up = Data_CFD_wss["wss"]
    Data_NN_wss = np.load(path + str(caseIdx) + "NN_wss.npz")
    NNwall_shear_mag_up = Data_NN_wss["wss"]
    # show plot

    plt.figure()

    plt.plot(
        unique_x,
        wall_shear_mag_up,
        label="CFD",
        color="darkblue",
        linestyle="-",
        lw=3.0,
        alpha=1.0,
    )
    plt.plot(
        unique_x,
        NNwall_shear_mag_up,
        label="DNN",
        color="red",
        linestyle="--",
        dashes=(5, 5),
        lw=2.0,
        alpha=1.0,
    )
    plt.xlabel(r"x", fontsize=16)
    plt.ylabel(r"$\tau_{c}$", fontsize=16)
    plt.legend(prop={"size": 16})
    plt.savefig(
        "plot/" + str(int(caseIdx)) + "nu" + str(nu) + "wallShear_test.png",
        bbox_inches="tight",
    )
    plt.close("all")

    ## show center wall shear
    # CFD
    W_ctl[int(caseIdx - 1)] = wall_shear_mag_up[int(len(wall_shear_mag_up) / 2)]
    # NN
    W_ctl_Ml[int(caseIdx - 1)] = NNwall_shear_mag_up[int(len(NNwall_shear_mag_up) / 2)]
