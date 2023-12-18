import numpy as np
import Ofpp
from omegaconf import DictConfig
from paddle.io import DataLoader
from paddle.io import Dataset
from pyMesh import hcubeMesh
from pyMesh import to4DTensor


class VaryGeoDataset(Dataset):
    """docstring for hcubeMeshDataset"""

    def __init__(self, MeshList):
        self.MeshList = MeshList

    def __len__(self):
        return len(self.MeshList)

    def __getitem__(self, idx):
        mesh = self.MeshList[idx]
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
        InvariantInput = np.zeros([2, J.shape[0], J.shape[1]])
        InvariantInput[0, :, :] = J
        InvariantInput[1, :, :] = Jinv
        return [InvariantInput, cord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta]


def generate_data(cfg: DictConfig):
    h = cfg.MODEL.h
    OFBCCoord = Ofpp.parse_boundary_field(cfg.boundary_dir)
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
    batchSize = cfg.TRAIN.batchsize
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
    return coords, jinvs, dxdxis, dydxis, dxdetas, dydetas, nx, ny
