import random

import numpy as np
import Ofpp
import paddle
from dataset import VaryGeoDataset
from paddle.io import DataLoader
from pyMesh import hcubeMesh
from pyMesh import to4DTensor
from readOF import convertOFMeshToImage_StructuredMesh
from sklearn.metrics import mean_squared_error as calMSE

import ppsci


def dfdx(f, dydeta, dydxi, Jinv, h=0.01):
    dfdxi_internal = (
        (
            -f[:, :, :, 4:]
            + 8 * f[:, :, :, 3:-1]
            - 8 * f[:, :, :, 1:-3]
            + f[:, :, :, 0:-4]
        )
        / 12
        / h
    )
    dfdxi_left = (
        (
            -11 * f[:, :, :, 0:-3]
            + 18 * f[:, :, :, 1:-2]
            - 9 * f[:, :, :, 2:-1]
            + 2 * f[:, :, :, 3:]
        )
        / 6
        / h
    )
    dfdxi_right = (
        (
            11 * f[:, :, :, 3:]
            - 18 * f[:, :, :, 2:-1]
            + 9 * f[:, :, :, 1:-2]
            - 2 * f[:, :, :, 0:-3]
        )
        / 6
        / h
    )
    dfdxi = paddle.concat(
        (dfdxi_left[:, :, :, 0:2], dfdxi_internal, dfdxi_right[:, :, :, -2:]), 3
    )
    dfdeta_internal = (
        (
            -f[:, :, 4:, :]
            + 8 * f[:, :, 3:-1, :]
            - 8 * f[:, :, 1:-3, :]
            + f[:, :, 0:-4, :]
        )
        / 12
        / h
    )
    dfdeta_low = (
        (
            -11 * f[:, :, 0:-3, :]
            + 18 * f[:, :, 1:-2, :]
            - 9 * f[:, :, 2:-1, :]
            + 2 * f[:, :, 3:, :]
        )
        / 6
        / h
    )
    dfdeta_up = (
        (
            11 * f[:, :, 3:, :]
            - 18 * f[:, :, 2:-1, :]
            + 9 * f[:, :, 1:-2, :]
            - 2 * f[:, :, 0:-3, :]
        )
        / 6
        / h
    )
    dfdeta = paddle.concat(
        (dfdeta_low[:, :, 0:2, :], dfdeta_internal, dfdeta_up[:, :, -2:, :]), 2
    )
    dfdx = Jinv * (dfdxi * dydeta - dfdeta * dydxi)
    return dfdx


def dfdy(f, dxdxi, dxdeta, Jinv, h=0.01):
    dfdxi_internal = (
        (
            -f[:, :, :, 4:]
            + 8 * f[:, :, :, 3:-1]
            - 8 * f[:, :, :, 1:-3]
            + f[:, :, :, 0:-4]
        )
        / 12
        / h
    )
    dfdxi_left = (
        (
            -11 * f[:, :, :, 0:-3]
            + 18 * f[:, :, :, 1:-2]
            - 9 * f[:, :, :, 2:-1]
            + 2 * f[:, :, :, 3:]
        )
        / 6
        / h
    )
    dfdxi_right = (
        (
            11 * f[:, :, :, 3:]
            - 18 * f[:, :, :, 2:-1]
            + 9 * f[:, :, :, 1:-2]
            - 2 * f[:, :, :, 0:-3]
        )
        / 6
        / h
    )
    dfdxi = paddle.concat(
        (dfdxi_left[:, :, :, 0:2], dfdxi_internal, dfdxi_right[:, :, :, -2:]), 3
    )

    dfdeta_internal = (
        (
            -f[:, :, 4:, :]
            + 8 * f[:, :, 3:-1, :]
            - 8 * f[:, :, 1:-3, :]
            + f[:, :, 0:-4, :]
        )
        / 12
        / h
    )
    dfdeta_low = (
        (
            -11 * f[:, :, 0:-3, :]
            + 18 * f[:, :, 1:-2, :]
            - 9 * f[:, :, 2:-1, :]
            + 2 * f[:, :, 3:, :]
        )
        / 6
        / h
    )
    dfdeta_up = (
        (
            11 * f[:, :, 3:, :]
            - 18 * f[:, :, 2:-1, :]
            + 9 * f[:, :, 1:-2, :]
            - 2 * f[:, :, 0:-3, :]
        )
        / 6
        / h
    )
    dfdeta = paddle.concat(
        (dfdeta_low[:, :, 0:2, :], dfdeta_internal, dfdeta_up[:, :, -2:, :]), 2
    )
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
OFBCCoord = Ofpp.parse_boundary_field("TemplateCase/30/C")
OFLOWC = OFBCCoord[b"low"][b"value"]
OFUPC = OFBCCoord[b"up"][b"value"]
OFLEFTC = OFBCCoord[b"left"][b"value"]
OFRIGHTC = OFBCCoord[b"right"][b"value"]
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
myMesh = hcubeMesh(
    leftX,
    leftY,
    rightX,
    rightY,
    lowX,
    lowY,
    upX,
    upY,
    h,
    True,
    True,
    tolMesh=1e-10,
    tolJoint=1,
)
batchSize = 1
NvarInput = 2
NvarOutput = 1
nEpochs = 1500
lr = 0.001
Ns = 1
nu = 0.01
padSingleSide = 1
model = ppsci.arch.USCNN(h, nx, ny, NvarInput, NvarOutput, padSingleSide)
# model.set_state_dict(paddle.load('torch_init.pdparams'))


optimizer = ppsci.optimizer.Adam(lr)(model)
MeshList = []
MeshList.append(myMesh)
train_set = VaryGeoDataset(MeshList)
training_data_loader = DataLoader(dataset=train_set, batch_size=batchSize)
coords = []
jinvs = []
dxdxis = []
dydxis = []
dxdetas = []
dydetas = []
for iteration, batch in enumerate(training_data_loader):
    [_, coord, _, _, _, Jinv, dxdxi, dydxi, dxdeta, dydeta] = to4DTensor(batch)
    coords.append(coord)
    jinvs.append(Jinv)
    dxdxis.append(dxdxi)
    dydxis.append(dydxi)
    dxdetas.append(dxdeta)
    dydetas.append(dydeta)

coords = np.concatenate(coords)
jinvs = np.concatenate(jinvs)
dxdxis = np.concatenate(dxdxis)
dydxis = np.concatenate(dydxis)
dxdetas = np.concatenate(dxdetas)
dydetas = np.concatenate(dydetas)
#
# train = ([coords,jinvs,dxdxis,dydxis,dxdetas,dydetas])

sup_constraint_mres = ppsci.constraint.SupervisedConstraint(
    {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": {
                "coords": coords,
                "jinvs": jinvs,
                "dxdxis": dxdxis,
                "dydxis": dydxis,
                "dxdetas": dxdetas,
                "dydetas": dydetas,
            },
        },
        "batch_size": 1,
        "iters_per_epoch": coords.shape[0],
        "num_workers": 0,
    },
    ppsci.loss.FunctionalLoss(lambda out, label, weught: out["mRes"]),
    name="mRes",
)


constraint_pde = {sup_constraint_mres.name: sup_constraint_mres}


OFPicInformative = convertOFMeshToImage_StructuredMesh(
    nx, ny, "TemplateCase/30/C", ["TemplateCase/30/T"], [0, 1, 0, 1], 0.0, False
)
OFPic = convertOFMeshToImage_StructuredMesh(
    nx, ny, "TemplateCase/30/C", ["TemplateCase/30/T"], [0, 1, 0, 1], 0.0, False
)

OFX = OFPic[:, :, 0]
OFY = OFPic[:, :, 1]
OFV = OFPic[:, :, 2]
OFV_sb = paddle.to_tensor(OFV)


def _transform_out(_input, outputV, padSingleSide=padSingleSide):
    batchSize = outputV.shape[0]
    Jinv = _input["jinvs"]
    dxdxi = _input["dxdxis"]
    dydxi = _input["dydxis"]
    dxdeta = _input["dxdetas"]
    dydeta = _input["dydetas"]
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
    continuity = d2vdy2 + d2vdx2
    return {"mRes": paddle.mean(continuity**2)}


model.register_output_transform(_transform_out)
output_dir = "./output"
solver = ppsci.solver.Solver(
    model,
    constraint_pde,
    output_dir,
    optimizer,
    epochs=nEpochs,
    iters_per_epoch=coords.shape[0],
    eval_with_no_grad=True,
)

solver.train()

model.register_output_transform(None)

outputV = model({"coords": paddle.to_tensor(coords)})
outputV[0, 0, -padSingleSide:, padSingleSide:-padSingleSide] = 0
outputV[0, 0, :padSingleSide, padSingleSide:-padSingleSide] = 1
outputV[0, 0, padSingleSide:-padSingleSide, -padSingleSide:] = 1
outputV[0, 0, padSingleSide:-padSingleSide, 0:padSingleSide] = 1
outputV[0, 0, 0, 0] = 0.5 * (outputV[0, 0, 0, 1] + outputV[0, 0, 1, 0])
outputV[0, 0, 0, -1] = 0.5 * (outputV[0, 0, 0, -2] + outputV[0, 0, 1, -1])
CNNVNumpy = outputV[0, 0, :, :]
ev = np.sqrt(
    calMSE(OFV_sb.numpy(), CNNVNumpy.numpy())
    / calMSE(OFV_sb.numpy(), OFV_sb.numpy() * 0)
)
print(ev)
solver.plot_loss_history()
