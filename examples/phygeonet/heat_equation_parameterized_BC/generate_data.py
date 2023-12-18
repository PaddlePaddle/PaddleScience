import numpy as np
from omegaconf import DictConfig
from paddle.io import DataLoader
from paddle.io import Dataset
from pyMesh import hcubeMesh
from pyMesh import to4DTensor
from readOF import convertOFMeshToImage_StructuredMesh


class FixGeoDataset(Dataset):
    """docstring for hcubeMeshDataset"""

    def __init__(self, ParaList, mesh, OFSolutionList):
        self.ParaList = ParaList
        self.mesh = mesh
        self.OFSolutionList = OFSolutionList

    def __len__(self):
        return len(self.ParaList)

    def __getitem__(self, idx):
        mesh = self.mesh
        x = mesh.x
        y = mesh.y
        xi = mesh.xi
        eta = mesh.eta
        J = mesh.J_ho
        Jinv = mesh.Jinv_ho
        dxdxi = mesh.dxdxi_ho
        dydxi = mesh.dydxi_ho
        dxdeta = mesh.dxdeta_ho
        dydeta = mesh.dydeta_ho
        cord = np.zeros([2, x.shape[0], x.shape[1]])
        cord[0, :, :] = x
        cord[1, :, :] = y
        ParaStart = np.ones(x.shape[0]) * self.ParaList[idx]
        ParaEnd = np.zeros(x.shape[0])
        Para = np.linspace(ParaStart, ParaEnd, x.shape[1]).T
        return [
            Para,
            cord,
            xi,
            eta,
            J,
            Jinv,
            dxdxi,
            dydxi,
            dxdeta,
            dydeta,
            self.OFSolutionList[idx],
        ]


def generate_data(cfg: DictConfig):
    h = cfg.MODEL.h
    r = cfg.r
    R = cfg.R
    dtheta = cfg.dtheta
    ny = cfg.MODEL.ny
    nx = cfg.MODEL.nx
    leftX = r * np.cos(np.linspace(dtheta, 2 * np.pi - dtheta, ny))
    leftY = r * np.sin(np.linspace(dtheta, 2 * np.pi - dtheta, ny))
    rightX = R * np.cos(np.linspace(dtheta, 2 * np.pi - dtheta, ny))
    rightY = R * np.sin(np.linspace(dtheta, 2 * np.pi - dtheta, ny))
    lowX = np.linspace(leftX[0], rightX[0], nx)
    lowY = lowX * 0 + np.sin(dtheta)
    upX = np.linspace(leftX[-1], rightX[-1], nx)
    upY = upX * 0 - np.sin(dtheta)
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
        tolJoint=0.01,
    )
    ParaList = [1, 7]
    caseName = cfg.casename
    OFV_sb = []
    for name in caseName:
        OFPic = convertOFMeshToImage_StructuredMesh(
            nx, ny, name + "/30/C", [name + "/30/T"], [0, 1, 0, 1], 0.0, False
        )
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
    training_data_loader = DataLoader(dataset=train_set, batch_size=cfg.TRAIN.batchsize)
    coords = []
    jinvs = []
    dxdxis = []
    dydxis = []
    dxdetas = []
    dydetas = []
    truths = []
    for iteration, batch in enumerate(training_data_loader):
        [Para, _, _, _, _, Jinv, dxdxi, dydxi, dxdeta, dydeta, truth] = to4DTensor(
            batch
        )
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
    len_data = len(training_data_loader)
    return coords, jinvs, dxdxis, dydxis, dxdetas, dydetas, truths, len_data
