import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict
import paddle

paddle.device.set_device("gpu:6")
from paddle import nn
from paddle import optimizer as optim
from paddle.io import DataLoader
import ppsci
import time
import tikzplotlib

sys.path.insert(0, '../source')
from dataset import FixGeoDataset
from pyMesh import hcubeMesh, visualize2D, plotBC, plotMesh, setAxisLabel, to4DTensor

from readOF import convertOFMeshToImage, convertOFMeshToImage_StructuredMesh
from sklearn.metrics import mean_squared_error as calMSE
import Ofpp
import random


def dfdx(f, dydeta, dydxi, Jinv, h=0.01):
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


def dfdy(f, dxdxi, dxdeta, Jinv, h=0.01):
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


# initiallizer
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
padSingleSide = 1

model = ppsci.arch.USCNN(h, nx, ny, NvarInput, NvarOutput, padSingleSide)

optimizer = ppsci.optimizer.Adam(lr)(model)
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

coords = []
jinvs = []
dxdxis = []
dydxis = []
dxdetas = []
dydetas = []
truths = []
for iteration, batch in enumerate(training_data_loader):
    [Para, _, _, _, _, Jinv, dxdxi, dydxi, dxdeta, dydeta, truth] = to4DTensor(batch)
    coords.append(Para)
    jinvs.append(Jinv)
    dxdxis.append(dxdxi)
    dydxis.append(dydxi)
    dxdetas.append(dxdeta)
    dydetas.append(dydeta)
    truths.append(truth)

coords = np.concatenate(coords)
jinvs = np.concatenate(jinvs)
dxdxis = np.concatenate(dxdxis)
dydxis = np.concatenate(dydxis)
dxdetas = np.concatenate(dxdetas)
dydetas = np.concatenate(dydetas)
truths = np.concatenate(truths)
#
# train = ([coords,jinvs,dxdxis,dydxis,dxdetas,dydetas])

sup_constraint_mres = ppsci.constraint.SupervisedConstraint(
    {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": {'coords': coords, 'jinvs': jinvs, 'dxdxis': dxdxis, 'dydxis': dydxis, 'dxdetas': dxdetas,
                      'dydetas': dydetas},
        },
        "batch_size": 1,
        'iters_per_epoch': coords.shape[0],
        "num_workers": 0,
    },
    ppsci.loss.FunctionalLoss(lambda out, label, weught: out["mRes"]),
    name="mRes",
)

constraint_pde = {sup_constraint_mres.name: sup_constraint_mres}

OFPicInformative = convertOFMeshToImage_StructuredMesh(nx, ny, 'TemplateCase/30/C',
                                                       ['TemplateCase/30/T'],
                                                       [0, 1, 0, 1], 0.0, False)


def _transform_out(_input, outputV, padSingleSide=padSingleSide, k=len(training_data_loader)):
    batchSize = outputV.shape[0]
    Jinv = _input['jinvs']
    dxdxi = _input['dxdxis']
    dydxi = _input['dydxis']
    dxdeta = _input['dxdetas']
    dydeta = _input['dydetas']
    Para = _input['coords']
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
    continuity = (d2vdy2 + d2vdx2)
    return {"mRes": paddle.mean(continuity ** 2) / k}


model.register_output_transform(_transform_out)
output_dir = './output'
solver = ppsci.solver.Solver(
    model,
    constraint_pde,
    output_dir,
    optimizer,
    epochs=nEpochs,
    iters_per_epoch=coords.shape[0],
    eval_with_no_grad=True,
    device='gpu:6'
)

solver.train()

model.register_output_transform(None)

outputV = model({'coords': paddle.to_tensor(coords)})
criterion=nn.MSELoss()
eV = 0
for iteration, batch in enumerate(training_data_loader):
    [Para, _, _, _, _, Jinv, dxdxi, dydxi, dxdeta, dydeta, truth] = to4DTensor(batch)
    outputV = model({'coords': paddle.to_tensor(Para)})
    batchSize = outputV.shape[0]
    for j in range(batchSize):
        # Impose BC
        outputV[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = (
            outputV[j, 0, 1:2, padSingleSide:-padSingleSide])
        outputV[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = (
            outputV[j, 0, -2:-1, padSingleSide:-padSingleSide])
        outputV[j, 0, :, -padSingleSide:] = 0
        outputV[j, 0, :, 0:padSingleSide] = Para[j, 0, 0, 0]
    eV = eV + paddle.sqrt(criterion(truth, outputV) / criterion(truth, truth * 0)).item()
print(eV / len(training_data_loader))
solver.plot_loss_history()
