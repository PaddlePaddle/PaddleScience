import os
import pdb

import matplotlib.pyplot as plt  # For plotting
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

os.chdir("/workspace/wangguan/PaddleScience_Surrogate/examples/pipe")
data1 = np.load("train_nu.npz")
# data1 = np.load('pipe_test_1dnu.npz')
nu = data1["nu_1d"]
# nu = np.sort(nu)
print("nu is", nu)
############################
# profile viscosity

# ss
data = np.load("pred_poiseuille_para_0425.npz")
mesh = data["mesh"]
print("shape of mesh is", mesh.shape)
u = data["u"]
v = data["v"]
p = data["p"]
ut = data["ut"]
uMaxP = data["uMaxP"]
uMaxA = data["uMaxA"]
print("shape of uMaxA", uMaxA.shape)
Ny, Nx, Np = u.shape
print("mesh shape = ", mesh.shape)
print("u shape", u.shape)
idxP = 28

# plt.figure()
# plt.contourf(mesh[0,:,:, idxP], mesh[1,:,:,idxP], u[:, :, idxP])
# plt.axis('equal')
# plt.colorbar()
# plt.savefig('pipe1.png')

# idxP = np.array([0,28,49])
idxP = [3]
plot_x = 0.8
plot_y = 0.07
fontsize = 16

# y = np.linspace(-0.05,0.05,50)
# ii = 0
# for idxPi in idxP:
# 	plt.figure()
# 	for i in range(Nx):
# 	    pP, = plt.plot(y,u[:, i, idxPi])
# 	    pT, = plt.plot(y,ut[:, i, idxPi], 'r--')
# 	ii = ii+1
# plt.close('all')
# plt.legend([pP, pT], ['NN Surrogate,nu = 2.1e-4', 'Truth'],fontsize = fontsize)
# print ('max u = ', np.max(u[:, :, idxP]))
# print ('max ut = ', np.max(ut[:, :, idxP]))
# plt.savefig('pipe2.png')

# plt.text(0, 0.1, r'$\delta$',
# {'color': 'k', 'fontsize': 24, 'ha': 'center', 'va': 'center',
#'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
d = 0.1
# plot spanwise u profile along y, looping from nu_small to nu_large
# u = data['u']
idx_X = int(round(Nx / 2))
y = np.linspace(-0.05, 0.05, 50)
can = [3, 6, 14, 49]
# for idxP in range(len(nu)):
xtext = [0, 0.5, 1]
ytext = [0.45, 0.28, 0.1, 0.01]
plt.figure(1)
Re = []
plt.figure(1)
plt.clf()
for idxP in range(len(can)):
    # plt.figure(1)
    # plt.clf()
    print(f"idxP = {idxP}")
    ax1 = plt.subplot(111)
    (pT,) = plt.plot(
        y, ut[:, idx_X, can[idxP]], color="darkblue", linestyle="-", lw=3.0, alpha=1.0
    )
    (pP,) = plt.plot(
        y,
        u[:, idx_X, can[idxP]],
        color="red",
        linestyle="--",
        dashes=(5, 5),
        lw=2.0,
        alpha=1.0,
    )
    tmpRe = np.max(u[:, idx_X, can[idxP]]) * d / nu[can[idxP]]
    Re.append(tmpRe)
    # print("Re is",Re)
    nu_current = float("{0:.5f}".format(nu[can[idxP]]))
    # plt.title(r'$\nu = $' + str(nu_current))
    plt.text(
        -0.012,
        ytext[idxP],
        r"$\nu = $" + str(nu_current),
        {"color": "k", "fontsize": 16},
    )


# plt.legend([pT, pP], ['Analytical', 'NN surrogate'], fontsize = 16,loc = 10)
plt.ylabel(r"$u(y)$", fontsize=16)
plt.xlabel(r"$y$", fontsize=16)
ax1.tick_params(axis="x", labelsize=16)
ax1.tick_params(axis="y", labelsize=16)
ax1.set_xlim([-0.05, 0.05])
ax1.set_ylim([0.0, 0.62])
figureName = "pipe_uProfiles_nuIdx_.png"
plt.savefig(figureName, bbox_inches="tight")
exit()

print("Re is", Re)
np.savez("test_Re", Re=Re)
plt.figure(2)
plt.clf()
ax1 = plt.subplot(111)
sns.kdeplot(uMaxA[0, :], shade=True, label="Analytical", linestyle="-", linewidth=3)
sns.kdeplot(
    uMaxP[0, :],
    shade=False,
    label="DNN",
    linestyle="--",
    linewidth=3.5,
    color="darkred",
)
plt.legend(prop={"size": 16})
plt.xlabel(r"$u_c$", fontsize=16)
plt.ylabel(r"PDF", fontsize=16)
ax1.tick_params(axis="x", labelsize=16)
ax1.tick_params(axis="y", labelsize=16)
figureName = "pipe_unformUQ.png"
plt.savefig(figureName, bbox_inches="tight")
plt.show()
