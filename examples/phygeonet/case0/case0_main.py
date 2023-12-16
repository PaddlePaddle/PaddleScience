import generate_data
import hydra
import numpy as np
import Ofpp
import paddle
from omegaconf import DictConfig
from paddle.io import DataLoader
from pyMesh import hcubeMesh
from pyMesh import to4DTensor
from readOF import convertOFMeshToImage_StructuredMesh
from sklearn.metrics import mean_squared_error as calMSE

import ppsci
from ppsci.data.dataset.geo_dataset import VaryGeoDataset


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


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def train(cfg: DictConfig):
    # initiallizer
    SEED = cfg.seed
    ppsci.utils.misc.set_random_seed(SEED)
    h = cfg.MODEL.h
    (
        coords,
        jinvs,
        dxdxis,
        dydxis,
        dxdetas,
        dydetas,
        nx,
        ny,
    ) = generate_data.generate_data(cfg)

    NvarInput = cfg.MODEL.NvarInput
    NvarOutput = cfg.MODEL.NvarOutput
    padSingleSide = cfg.MODEL.padSingleSide
    model = ppsci.arch.USCNN(
        cfg.MODEL.input_keys,
        cfg.MODEL.output_keys,
        h,
        nx,
        ny,
        NvarInput,
        NvarOutput,
        padSingleSide,
    )
    # model.set_state_dict(paddle.load('torch_init.pdparams'))

    optimizer = ppsci.optimizer.Adam(cfg.TRAIN.lr)(model)

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

    def _transform_out(_input, _output, padSingleSide=padSingleSide):
        outputV = _output["outputV"]
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
    output_dir = cfg.output_dir
    solver = ppsci.solver.Solver(
        model,
        constraint_pde,
        output_dir,
        optimizer,
        epochs=cfg.TRAIN.epochs,
        iters_per_epoch=coords.shape[0],
        eval_with_no_grad=cfg.TRAIN.eval_with_no_grad,
    )

    solver.train()

    solver.plot_loss_history()


def evaluate(cfg: DictConfig):

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
    OFPic = convertOFMeshToImage_StructuredMesh(
        nx, ny, "TemplateCase/30/C", ["TemplateCase/30/T"], [0, 1, 0, 1], 0.0, False
    )

    OFV = OFPic[:, :, 2]
    OFV_sb = paddle.to_tensor(OFV)
    SEED = cfg.seed
    ppsci.utils.misc.set_random_seed(SEED)
    padSingleSide = 1
    model = ppsci.arch.USCNN(**cfg.MODEL)
    model.register_output_transform(None)
    MeshList = []
    MeshList.append(myMesh)
    train_set = VaryGeoDataset(MeshList)
    training_data_loader = DataLoader(dataset=train_set, batch_size=cfg.batchsize)
    solver = ppsci.solver.Solver(
        model,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,  ### the path of the model
    )
    coords = []
    for iteration, batch in enumerate(training_data_loader):
        [_, coord, _, _, _, _, _, _, _, _] = to4DTensor(batch)
        coords.append(coord)

    coords = np.concatenate(coords)
    outputV = solver.predict({"coords": paddle.to_tensor(coords)})
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


@hydra.main(version_base=None, config_path="./conf", config_name="case2.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
