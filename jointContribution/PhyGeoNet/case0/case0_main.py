import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import paddle
import ppsci
from paddle import nn
from paddle.io import DataLoader

import time
import tikzplotlib

sys.path.insert(0, '../source')
from dataset import VaryGeoDataset
from pyMesh import hcubeMesh, visualize2D, plotBC, plotMesh, setAxisLabel, \
    np2cuda, to4DTensor
from model import USCNN

from readOF import convertOFMeshToImage, convertOFMeshToImage_StructuredMesh
from sklearn.metrics import mean_squared_error as calMSE
import Ofpp

import random


def set_random_seed(seed: int):
    """Set numpy, random, paddle random_seed to given seed.

    Args:
        seed (int): Random seed.
    """
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_random_seed(42)


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


h = 0.01
OFBCCoord = Ofpp.parse_boundary_field('TemplateCase/30/C')
OFLOWC = OFBCCoord[b'low'][b'value']
OFUPC = OFBCCoord[b'up'][b'value']
OFLEFTC = OFBCCoord[b'left'][b'value']
OFRIGHTC = OFBCCoord[b'right'][b'value']
leftX = OFLEFTC[:, 0]
leftY = OFLEFTC[:, 1]
lowX = OFLOWC[:, 0]
lowY = OFLOWC[:, 1]
rightX = OFRIGHTC[:, 0]
rightY = OFRIGHTC[:, 1]
upX = OFUPC[:, 0]
upY = OFUPC[:, 1]
ny = len(leftX)
nx = len(lowX)
myMesh = hcubeMesh(leftX, leftY, rightX, rightY,
                   lowX, lowY, upX, upY, h, True, True,
                   tolMesh=1e-10, tolJoint=1)
batchSize = 1
NvarInput = 2
NvarOutput = 1
nEpochs = 1500
lr = 0.001
Ns = 1
nu = 0.01
model = USCNN(h, nx, ny, NvarInput, NvarOutput)

criterion = nn.MSELoss()
optimizer = ppsci.optimizer.Adam(lr)(model)
padSingleSide = 1
udfpad = nn.Pad2D([padSingleSide, padSingleSide, padSingleSide, padSingleSide], value=0)
MeshList = []
MeshList.append(myMesh)
train_set = VaryGeoDataset(MeshList)
training_data_loader = DataLoader(dataset=train_set,
                                  batch_size=batchSize)
OFPicInformative = convertOFMeshToImage_StructuredMesh(nx, ny, 'TemplateCase/30/C',
                                                       ['TemplateCase/30/T'],
                                                       [0, 1, 0, 1], 0.0, False)
OFPic = convertOFMeshToImage_StructuredMesh(nx, ny, 'TemplateCase/30/C',
                                            ['TemplateCase/30/T'],
                                            [0, 1, 0, 1], 0.0, False)
OFX = OFPic[:, :, 0]
OFY = OFPic[:, :, 1]
OFV = OFPic[:, :, 2]
OFV_sb = OFV


def dfdx(f, dydeta, dydxi, Jinv):
    dfdxi_internal = (-f[:, :, :, 4:] + 8 * f[:, :, :, 3:-1] - 8 * f[:, :, :, 1:-3] + f[:, :, :, 0:-4]) / 12 / h
    dfdxi_left = (-11 * f[:, :, :, 0:-3] + 18 * f[:, :, :, 1:-2] - 9 * f[:, :, :, 2:-1] + 2 * f[:, :, :, 3:]) / 6 / h
    dfdxi_right = (11 * f[:, :, :, 3:] - 18 * f[:, :, :, 2:-1] + 9 * f[:, :, :, 1:-2] - 2 * f[:, :, :, 0:-3]) / 6 / h
    dfdxi = paddle.concat((dfdxi_left[:, :, :, 0:2], dfdxi_internal, dfdxi_right[:, :, :, -2:]), 3)
    dfdeta_internal = (-f[:, :, 4:, :] + 8 * f[:, :, 3:-1, :] - 8 * f[:, :, 1:-3, :] + f[:, :, 0:-4, :]) / 12 / h
    dfdeta_low = (-11 * f[:, :, 0:-3, :] + 18 * f[:, :, 1:-2, :] - 9 * f[:, :, 2:-1, :] + 2 * f[:, :, 3:, :]) / 6 / h
    dfdeta_up = (11 * f[:, :, 3:, :] - 18 * f[:, :, 2:-1, :] + 9 * f[:, :, 1:-2, :] - 2 * f[:, :, 0:-3, :]) / 6 / h
    dfdeta = paddle.concat((dfdeta_low[:, :, 0:2, :], dfdeta_internal, dfdeta_up[:, :, -2:, :]), 2)
    dfdx = Jinv * (dfdxi * dydeta - dfdeta * dydxi)
    return dfdx


def dfdy(f, dxdxi, dxdeta, Jinv):
    dfdxi_internal = (-f[:, :, :, 4:] + 8 * f[:, :, :, 3:-1] - 8 * f[:, :, :, 1:-3] + f[:, :, :, 0:-4]) / 12 / h
    dfdxi_left = (-11 * f[:, :, :, 0:-3] + 18 * f[:, :, :, 1:-2] - 9 * f[:, :, :, 2:-1] + 2 * f[:, :, :, 3:]) / 6 / h
    dfdxi_right = (11 * f[:, :, :, 3:] - 18 * f[:, :, :, 2:-1] + 9 * f[:, :, :, 1:-2] - 2 * f[:, :, :, 0:-3]) / 6 / h
    dfdxi = paddle.concat((dfdxi_left[:, :, :, 0:2], dfdxi_internal, dfdxi_right[:, :, :, -2:]), 3)

    dfdeta_internal = (-f[:, :, 4:, :] + 8 * f[:, :, 3:-1, :] - 8 * f[:, :, 1:-3, :] + f[:, :, 0:-4, :]) / 12 / h
    dfdeta_low = (-11 * f[:, :, 0:-3, :] + 18 * f[:, :, 1:-2, :] - 9 * f[:, :, 2:-1, :] + 2 * f[:, :, 3:, :]) / 6 / h
    dfdeta_up = (11 * f[:, :, 3:, :] - 18 * f[:, :, 2:-1, :] + 9 * f[:, :, 1:-2, :] - 2 * f[:, :, 0:-3, :]) / 6 / h
    dfdeta = paddle.concat((dfdeta_low[:, :, 0:2, :], dfdeta_internal, dfdeta_up[:, :, -2:, :]), 2)
    dfdy = Jinv * (dfdeta * dxdxi - dfdxi * dxdeta)
    return dfdy


mRes_list = []
eV_list = []


def train(epoch):
    startTime = time.time()
    xRes = 0
    yRes = 0
    mRes = 0
    eU = 0
    eV = 0
    eP = 0
    for iteration, batch in enumerate(training_data_loader):
        [JJInv, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta] = to4DTensor(batch)
        optimizer.clear_grad()
        output = model(coord)
        output_pad = udfpad(output)
        outputV = output_pad[:, 0, :, :].reshape([output_pad.shape[0], 1,
                                                  output_pad.shape[2],
                                                  output_pad.shape[3]])
        for j in range(batchSize):
            outputV[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = 0
            outputV[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = 1
            outputV[j, 0, padSingleSide:-padSingleSide, -padSingleSide:] = 1
            outputV[j, 0, padSingleSide:-padSingleSide, 0:padSingleSide] = 1
            outputV[j, 0, 0, 0] = 0.5 * (outputV[j, 0, 0, 1] + outputV[j, 0, 1, 0])
            outputV[j, 0, 0, -1] = 0.5 * (outputV[j, 0, 0, -2] + outputV[j, 0, 1, -1])
        dvdx = dfdx(outputV, dydeta, dydxi, Jinv)
        d2vdx2 = dfdx(dvdx, dydeta, dydxi, Jinv)
        dvdy = dfdy(outputV, dxdxi, dxdeta, Jinv)
        d2vdy2 = dfdy(dvdy, dxdxi, dxdeta, Jinv)
        continuity = (d2vdy2 + d2vdx2)

        loss = criterion(continuity, continuity * 0)
        loss.backward()
        optimizer.step()
        loss_mass = criterion(continuity, continuity * 0)
        mRes += loss_mass.item()
        CNNVNumpy = outputV[0, 0, :, :].cpu().detach().numpy()
        eV = eV + np.sqrt(calMSE(OFV_sb, CNNVNumpy) / calMSE(OFV_sb, OFV_sb * 0))
    print('Epoch is ', epoch)
    print("mRes Loss is", (mRes / len(training_data_loader)))
    print("eV Loss is", (eV / len(training_data_loader)))
    mRes_list.append("{:.8g}".format(mRes))
    eV_list.append("{:.8g}".format(eV))
    if epoch % 5000 == 0 or epoch % nEpochs == 0 or np.sqrt(
            calMSE(OFV_sb, CNNVNumpy) / calMSE(OFV_sb, OFV_sb * 0)) < 0.1:
        paddle.save(model.state_dict(), f'./Result/{epoch}.pdparams')
        fig1 = plt.figure()
        ax = plt.subplot(1, 2, 1)
        visualize2D(ax, coord[0, 0, 1:-1, 1:-1].cpu().detach().numpy(),
                    coord[0, 1, 1:-1, 1:-1].cpu().detach().numpy(),
                    outputV[0, 0, 1:-1, 1:-1].cpu().detach().numpy(), 'horizontal', [0, 1])
        setAxisLabel(ax, 'p')
        ax.set_title('CNN ' + r'$T$')
        ax.set_aspect('equal')
        ax = plt.subplot(1, 2, 2)
        visualize2D(ax, coord[0, 0, 1:-1, 1:-1].cpu().detach().numpy(),
                    coord[0, 1, 1:-1, 1:-1].cpu().detach().numpy(),
                    OFV_sb[1:-1, 1:-1], 'horizontal', [0, 1])
        setAxisLabel(ax, 'p')
        ax.set_aspect('equal')
        ax.set_title('FV ' + r'$T$')
        fig1.tight_layout(pad=1)
        fig1.savefig(f'./Result/{epoch}T.pdf', bbox_inches='tight')
        plt.close(fig1)
    return (mRes / len(training_data_loader)), (eV / len(training_data_loader))


MRes = []
EV = []
TotalstartTime = time.time()
for epoch in range(1, nEpochs + 1):
    mres, ev = train(epoch)
    MRes.append(mres)
    EV.append(ev)
    if ev < 0.1:
        break
TimeSpent = time.time() - TotalstartTime

fig = plt.figure()
plt.plot(MRes, '-*', label='Equation Residual')
plt.xlabel('Epoch')
plt.ylabel('Residual')
plt.legend()
plt.yscale('log')
plt.savefig('./Result/convergence.pdf', bbox_inches='tight')
tikzplotlib_fix_ncols(fig)
tikzplotlib.save('./Result/convergence.tikz')
fig = plt.figure()
plt.plot(EV, '-x', label=r'$e_v$')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.yscale('log')
plt.savefig('./Result/error.pdf', bbox_inches='tight')
tikzplotlib_fix_ncols(fig)
tikzplotlib.save('./Result/error.tikz')
EV = np.asarray(EV)
MRes = np.asarray(MRes)
np.savetxt('./Result/EV.txt', EV)
np.savetxt('./Result/MRes.txt', MRes)
np.savetxt('./Result/TimeSpent.txt', np.zeros([2, 2]) + TimeSpent)
