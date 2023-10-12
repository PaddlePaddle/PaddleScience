import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import paddle
from paddle import nn
from paddle import optimizer as optim
import pdb
from paddle.io import DataLoader
import time
from scipy.interpolate import interp1d
import tikzplotlib

sys.path.insert(0, '../source')
from dataset import VaryGeoDataset, FixGeoDataset
from pyMesh import hcubeMesh, visualize2D, plotBC, plotMesh, setAxisLabel, \
    np2cuda, to4DTensor
from model import USCNN
from readOF import convertOFMeshToImage, convertOFMeshToImage_StructuredMesh
from sklearn.metrics import mean_squared_error as calMSE
import Ofpp

h = 0.01
r = 0.5
R = 1
dtheta = 0
OFBCCoord = Ofpp.parse_boundary_field('TemplateCase/30/C')
OFLEFTC = OFBCCoord[b'left'][b'value']
OFRIGHTC = OFBCCoord[b'right'][b'value']

leftX = r * np.cos(np.linspace(dtheta, 2 * np.pi - dtheta, 276))
leftY = r * np.sin(np.linspace(dtheta, 2 * np.pi - dtheta, 276))
rightX = R * np.cos(np.linspace(dtheta, 2 * np.pi - dtheta, 276))
rightY = R * np.sin(np.linspace(dtheta, 2 * np.pi - dtheta, 276))

lowX = np.linspace(leftX[0], rightX[0], 49)
lowY = lowX * 0 + np.sin(dtheta)
upX = np.linspace(leftX[-1], rightX[-1], 49)
upY = upX * 0 - np.sin(dtheta)

ny = len(leftX)
nx = len(lowX)

myMesh = hcubeMesh(leftX, leftY, rightX, rightY,
                   lowX, lowY, upX, upY, h, False, True,
                   tolMesh=1e-10, tolJoint=0.01)

batchSize = 1
NvarInput = 1
NvarOutput = 1
nEpochs = 1
lr = 0.001
Ns = 1
nu = 0.01

model = USCNN(h, nx, ny, NvarInput, NvarOutput)
model.set_state_dict(paddle.load('./Result/1000.pdparams'))

criterion = nn.MSELoss()
padSingleSide = 1
udfpad = nn.Pad2D([padSingleSide,padSingleSide,padSingleSide,padSingleSide],value=0)

ParaList = [1, 2, 3, 4, 5, 6, 7]
caseName = ['TemplateCase0', 'TemplateCase1', 'TemplateCase2', 'TemplateCase3',
            'TemplateCase4', 'TemplateCase5', 'TemplateCase6']
OFV_sb = []
for name in caseName:
    OFPic = convertOFMeshToImage_StructuredMesh(nx, ny, name + '/30/C',
                                                [name + '/30/T'],
                                                [0, 1, 0, 1], 0.0, False)
    OFX = OFPic[:, :, 0]
    OFY = OFPic[:, :, 1]
    OFV = OFPic[:, :, 2]

    OFV_sb_Temp = np.zeros(OFV.shape)

    for i in range(nx):
        for j in range(ny):
            dist = (myMesh.x[j, i] - OFX) ** 2 + (myMesh.y[j, i] - OFY) ** 2
            idx_min = np.where(dist == dist.min())
            OFV_sb_Temp[j, i] = OFV[idx_min]
    OFV_sb.append(OFV_sb_Temp)

test_set = FixGeoDataset(ParaList, myMesh, OFV_sb)


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


VelocityMagnitudeErrorRecord = []
for i in range(len(ParaList)):
    [Para, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta, truth] = to4DTensor(test_set[i])

    Para = Para.reshape([1, 1, Para.shape[0], Para.shape[1]])
    truth = truth.reshape([1, 1, truth.shape[0], truth.shape[1]])
    coord = coord.reshape([1, 2, coord.shape[2], coord.shape[3]])

    print('i=', str(i))
    output = model(Para)
    output_pad = udfpad(output)
    outputV = output_pad[:, 0, :, :].reshape([output_pad.shape[0], 1,
                                             output_pad.shape[2],
                                             output_pad.shape[3]])
    # Impose BC
    outputV[0, 0, -padSingleSide:, padSingleSide:-padSingleSide] = (
    outputV[0, 0, 1:2, padSingleSide:-padSingleSide])  # up outlet bc zero gradient
    outputV[0, 0, :padSingleSide, padSingleSide:-padSingleSide] = (
    outputV[0, 0, -2:-1, padSingleSide:-padSingleSide])  # down inlet bc
    outputV[0, 0, :, -padSingleSide:] = 0  # right wall bc
    outputV[0, 0, :, 0:padSingleSide] = Para[0, 0, 0, 0]  # left  wall bc

    dvdx = dfdx(outputV, dydeta, dydxi, Jinv)
    d2vdx2 = dfdx(dvdx, dydeta, dydxi, Jinv)

    dvdy = dfdy(outputV, dxdxi, dxdeta, Jinv)
    d2vdy2 = dfdy(dvdy, dxdxi, dxdeta, Jinv)
    # Calculate PDE Residual
    continuity = (d2vdy2 + d2vdx2)
    loss = criterion(continuity, continuity * 0)
    VelocityMagnitudeErrorRecord.append(paddle.sqrt(criterion(truth, outputV) / criterion(truth, truth * 0)))
    fig1 = plt.figure()
    xylabelsize = 20
    xytickssize = 20
    titlesize = 20
    ax = plt.subplot(1, 2, 1)
    _, cbar = visualize2D(ax, coord[0, 0, :, :].cpu().detach().numpy(),
                          coord[0, 1, :, :].cpu().detach().numpy(),
                          outputV[0, 0, :, :].cpu().detach().numpy(), 'horizontal', [0, max(ParaList)])
    ax.set_aspect('equal')
    setAxisLabel(ax, 'p')
    ax.set_title('PhyGeoNet ' + r'$T$', fontsize=titlesize)
    ax.set_xlabel(xlabel=r'$x$', fontsize=xylabelsize)
    ax.set_ylabel(ylabel=r'$y$', fontsize=xylabelsize)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(axis='x', labelsize=xytickssize)
    ax.tick_params(axis='y', labelsize=xytickssize)
    cbar.set_ticks([0, 1, 2, 3, 4, 5, 6, 7])
    cbar.ax.tick_params(labelsize=xytickssize)
    ax = plt.subplot(1, 2, 2)
    _, cbar = visualize2D(ax, coord[0, 0, :, :].cpu().detach().numpy(),
                          coord[0, 1, :, :].cpu().detach().numpy(),
                          truth[0, 0, :, :].cpu().detach().numpy(), 'horizontal', [0, max(ParaList)])
    ax.set_aspect('equal')
    setAxisLabel(ax, 'p')
    ax.set_title('FV ' + r'$T$', fontsize=titlesize)
    ax.set_xlabel(xlabel=r'$x$', fontsize=xylabelsize)
    ax.set_ylabel(ylabel=r'$y$', fontsize=xylabelsize)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(axis='x', labelsize=xytickssize)
    ax.tick_params(axis='y', labelsize=xytickssize)
    cbar.set_ticks([0, 1, 2, 3, 4, 5, 6, 7])
    cbar.ax.tick_params(labelsize=xytickssize)
    fig1.tight_layout(pad=1)
    fig1.savefig(f'./Result/Para{i}T.pdf', bbox_inches='tight')
    fig1.savefig(f'./Result/Para{i}T.png', bbox_inches='tight')
    plt.close(fig1)

VErrorNumpy = np.asarray([i.cpu().detach().numpy() for i in VelocityMagnitudeErrorRecord])
plt.figure()
plt.plot(np.asarray(ParaList), VErrorNumpy, '-x', label='Temperature Error')
plt.legend()
plt.xlabel('Inner circle temprature')
plt.ylabel('Error')
plt.savefig('./Result/Error.pdf', bbox_inches='tight')
tikzplotlib.save('./Result/Error.tikz')
