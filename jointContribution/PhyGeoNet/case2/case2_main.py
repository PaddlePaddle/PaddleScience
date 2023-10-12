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

sys.path.insert(0, '../source')
from dataset import FixGeoDataset
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
                   lowX, lowY, upX, upY, h, True, True,
                   tolMesh=1e-10, tolJoint=0.01)

batchSize = 2
NvarInput = 1
NvarOutput = 1
nEpochs = 1000
lr = 0.001
Ns = 1
nu = 0.01

model = USCNN(h, nx, ny, NvarInput, NvarOutput)
criterion = nn.MSELoss()
optimizer = optim.Adam(learning_rate=lr, parameters=model.parameters())
padSingleSide = 1
udfpad = nn.Pad2D([padSingleSide,padSingleSide,padSingleSide,padSingleSide],value=0)
ParaList = [1, 7]
caseName = ['TemplateCase0', 'TemplateCase6']
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
train_set = FixGeoDataset(ParaList, myMesh, OFV_sb)
training_data_loader = DataLoader(dataset=train_set,
                                  batch_size=batchSize)


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


def train(epoch):
    startTime = time.time()
    xRes = 0
    yRes = 0
    mRes = 0
    eU = 0
    eV = 0
    eP = 0
    for iteration, batch in enumerate(training_data_loader):
        [Para, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta, truth] = to4DTensor(batch)
        optimizer.clear_grad()
        output = model(Para)
        output_pad = udfpad(output)
        outputV = output_pad[:, 0, :, :].reshape([output_pad.shape[0], 1,
                                                 output_pad.shape[2],
                                                 output_pad.shape[3]])
        for j in range(batchSize):
            # Impose BC
            outputV[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = (
            outputV[j, 0, 1:2, padSingleSide:-padSingleSide])
            outputV[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = (
            outputV[j, 0, -2:-1, padSingleSide:-padSingleSide])
            outputV[j, 0, :, -padSingleSide:] = 0
            outputV[j, 0, :, 0:padSingleSide] = Para[j, 0, 0, 0]
        dvdx = dfdx(outputV, dydeta, dydxi, Jinv)
        d2vdx2 = dfdx(dvdx, dydeta, dydxi, Jinv)
        dvdy = dfdy(outputV, dxdxi, dxdeta, Jinv)
        d2vdy2 = dfdy(dvdy, dxdxi, dxdeta, Jinv)
        continuity = (d2vdy2 + d2vdx2);
        loss = criterion(continuity, continuity * 0)
        loss.backward()
        optimizer.step()
        loss_mass = criterion(continuity, continuity * 0)
        mRes += loss_mass.item()

        eV = eV + paddle.sqrt(criterion(truth, outputV) / criterion(truth, truth * 0)).item()
        if epoch % 1000 == 0 or epoch % nEpochs == 0:
            paddle.save(model.state_dict(), f'./Result/{epoch}.pdparams')
            for j in range(batchSize):
                fig1 = plt.figure()
                ax = plt.subplot(1, 2, 1)
                visualize2D(ax, coord[0, 0, :, :].cpu().detach().numpy(),
                            coord[0, 1, :, :].cpu().detach().numpy(),
                            outputV[j, 0, :, :].cpu().detach().numpy())
                ax.set_aspect('equal')
                setAxisLabel(ax, 'p')
                ax.set_title('CNN ' + r'$T$')
                ax = plt.subplot(1, 2, 2)
                visualize2D(ax, coord[0, 0, :, :].cpu().detach().numpy(),
                            coord[0, 1, :, :].cpu().detach().numpy(),
                            truth[j, 0, :, :].cpu().detach().numpy())
                ax.set_aspect('equal')
                setAxisLabel(ax, 'p')
                ax.set_title('FV ' + r'$T$')
                fig1.tight_layout(pad=1)
                fig1.savefig(f'./Result/Epoch{epoch}Para{j}T.pdf', bbox_inches='tight')
                plt.close(fig1)
    print('Epoch is ', epoch)
    print("mRes Loss is", (mRes / len(training_data_loader)))
    print("eV Loss is", (eV / len(training_data_loader)))
    return (mRes / len(training_data_loader)), (eV / len(training_data_loader))


MRes = []
EV = []
TotalstartTime = time.time()
for epoch in range(1, nEpochs + 1):
    mres, ev = train(epoch)
    MRes.append(mres)
    EV.append(ev)
TimeSpent = time.time() - TotalstartTime
plt.figure()
plt.plot(MRes, '-*', label='Equation Residual')
plt.xlabel('Epoch')
plt.ylabel('Residual')
plt.legend()
plt.yscale('log')
plt.savefig('./Result/convergence.pdf', bbox_inches='tight')
plt.figure()
plt.plot(EV, '-x', label=r'$e_v$')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.yscale('log')
plt.savefig('./Result/error.pdf', bbox_inches='tight')
EV = np.asarray(EV)
MRes = np.asarray(MRes)
np.savetxt('./Result/EV.txt', EV)
np.savetxt('./Result/MRes.txt', MRes)
np.savetxt('./Result/TimeSpent.txt', np.zeros([2, 2]) + TimeSpent)
