from __future__ import annotations

import csv
import os
import re
from os import path as osp
from timeit import default_timer

import numpy as np
import paddle
import pandas as pd
from omegaconf import DictConfig
from omegaconf import OmegaConf
from paddle import nn
from paddle.optimizer.lr import LRScheduler
from src.data import instantiate_datamodule
from src.networks import instantiate_network

from ppsci.utils import logger
from ppsci.utils import save_load
from ppsci.utils.misc import AverageMeter


class StepDecay(LRScheduler):
    def __init__(
        self, learning_rate, step_size, gamma=0.1, last_epoch=-1, verbose=False
    ):
        if not isinstance(step_size, int):
            raise TypeError(
                "The type of 'step_size' must be 'int', but received %s."
                % type(step_size)
            )
        if gamma >= 1.0:
            raise ValueError("gamma should be < 1.0.")

        self.step_size = step_size
        self.gamma = gamma
        super().__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        i = self.last_epoch // self.step_size
        return self.base_lr * (self.gamma**i)


def instantiate_scheduler(cfg: DictConfig):
    if cfg.opt_scheduler == "CosineAnnealingLR":
        scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
            cfg.lr, T_max=cfg.opt_scheduler_T_max
        )
    elif cfg.opt_scheduler == "StepLR":
        scheduler = StepDecay(cfg.lr, step_size=cfg.opt_step_size, gamma=cfg.opt_gamma)
    else:
        raise ValueError(f"Got {cfg.opt_scheduler}")
    return scheduler


# loss function with rel/abs Lp loss
class LpLoss:
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * paddle.norm(
            x.reshape((num_examples, -1)) - y.reshape((num_examples, -1)), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return paddle.mean(all_norms)
            else:
                return paddle.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        diff_norms = paddle.norm(x - y, 2)
        y_norms = paddle.norm(y, self.p)

        if self.reduction:
            if self.size_average:
                return paddle.mean(diff_norms / y_norms)
            else:
                return paddle.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def write_to_vtk(
    output_dir: str,
    out_dict: dict,
    point_data_pos: str = "press on mesh points",
    mesh_path: str = None,
    track: str | None = None,
):
    def extract_numbers(s):
        return [int(digit) for digit in re.findall(r"\d+", s)]

    import meshio

    p = out_dict["pressure"]
    index = extract_numbers(mesh_path.name)[0]

    if track == "Dataset_1":
        index = str(index).zfill(3)
    elif track == "Track_B":
        index = str(index).zfill(4)

    logger.info(f"Pressure shape for mesh {index} = {p.shape}")

    if point_data_pos == "press on mesh points":
        mesh = meshio.read(mesh_path)
        mesh.point_data["p"] = p.numpy()
        if "pred wss_x" in out_dict:
            wss_x = out_dict["pred wss_x"]
            mesh.point_data["wss_x"] = wss_x.numpy()
    elif point_data_pos == "press on mesh cells":
        points = np.load(mesh_path.parent / f"centroid_{index}.npy")
        npoint = points.shape[0]
        mesh = meshio.Mesh(
            points=points, cells=[("vertex", np.arange(npoint).reshape(npoint, 1))]
        )
        mesh.point_data = {"p": p.numpy()}

    dump_path = osp.join(output_dir, f"{mesh_path.parent.name}_{index}.vtk")
    mesh.write(dump_path)
    logger.info(f"write vtk to: {dump_path}")


def train(cfg: DictConfig):
    # init model
    model = instantiate_network(cfg)

    # init dataloader
    datamodule = instantiate_datamodule(cfg)
    train_loader = paddle.io.DataLoader(
        dataset=datamodule.train_data,
        batch_size=cfg.train_batch_size,
        shuffle=False,
        collate_fn=getattr(datamodule, "collate_fn", None),
        num_workers=0,
    )

    # Initialize learning rate scheduler
    lr_scheduler = paddle.optimizer.lr.StepDecay(
        cfg.opt.lr,
        cfg.opt.step_size,
        gamma=cfg.opt.gamma,
    )
    # Initialize optimizer
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(),
        learning_rate=lr_scheduler,
        weight_decay=cfg.opt.weight_decay,
    )

    # Initialize the loss function
    loss_fn = LpLoss(size_average=True)

    # start training
    for ep in range(cfg.num_epochs):
        model.train()

        tic = default_timer()

        train_loss_meter = AverageMeter()
        for i, data_dict in enumerate(train_loader):
            optimizer.clear_grad()

            # data transform
            input_grid_features, output_points = model.data_dict_to_input(data_dict)

            # model forward
            pred_var = model.forward(input_grid_features, output_points)

            # compute loss
            if "pressure" in data_dict:
                true_var = data_dict["pressure"].unsqueeze(-1)
            elif "velocity" in data_dict:
                true_var = data_dict["velocity"]
            elif "cd" in data_dict:
                pred_var = paddle.mean(pred_var)
                true_var = data_dict["cd"]
            else:
                raise NotImplementedError("only pressure velocity works")
            loss = loss_fn(pred_var, true_var)

            # loss backward
            loss.backward()

            # update parameters
            optimizer.step()

            # log loss value
            train_loss_meter.update(loss.item())

        # adjust learning rate by epoch
        lr_scheduler.step()

        epoch_cost = default_timer() - tic
        logger.info(
            f"Training epoch {ep} took {epoch_cost:.2f} seconds. L2 loss: {train_loss_meter.avg:.4f}"
        )

        # save model weights and optimizer parameters
        if ep % cfg.save_interval == 0 or ep == cfg.num_epochs - 1 and ep > 1:
            save_load.save_checkpoint(
                model,
                optimizer,
                output_dir=cfg.output_dir,
                prefix=f"model-{cfg.model}-{cfg.track}-{ep}",
                print_log=(ep == 0),
            )


@paddle.no_grad()
def eval(
    output_dir: str,
    model: nn.Layer,
    datamodule,
    cfg: DictConfig,
    loss_fn: callable = None,
    track: str = "Dataset_1",
):
    test_loader = datamodule.test_dataloader(
        batch_size=cfg.eval_batch_size, shuffle=False, num_workers=0
    )
    data_list = []
    cd_list = []

    for i, data_dict in enumerate(test_loader):
        out_dict = model.eval_dict(
            data_dict, loss_fn=loss_fn, decode_fn=datamodule.decode
        )
        if "l2 eval loss" in out_dict:
            if i == 0:
                data_list.append(["id", "l2 p"])
            else:
                data_list.append([i, float(out_dict["l2 eval loss"])])

        # TODO : you may write velocity into vtk, and analysis in your report
        if cfg.write_to_vtk is True:
            logger.info(
                f"datamodule.test_mesh_paths = {datamodule.test_mesh_paths[i]}",
            )
            write_to_vtk(
                cfg.output_dir,
                out_dict,
                cfg.point_data_pos,
                datamodule.test_mesh_paths[i],
                track,
            )

        # Your submit your npy to leaderboard here
        if "pressure" in out_dict:
            p = out_dict["pressure"].reshape((-1,)).astype(np.float32)
            test_indice = datamodule.test_indices[i]
            npy_leaderboard = osp.join(
                output_dir, f"{track}/press_{str(test_indice).zfill(3)}.npy"
            )
            logger.info(
                f"saving *.npy file for [{track}] leaderboard : {npy_leaderboard}"
            )
            np.save(npy_leaderboard, p)

        if "velocity" in out_dict:
            v = out_dict["velocity"].reshape((-1, 3)).astype(np.float32)
            test_indice = datamodule.test_indices[i]
            npy_leaderboard = osp.join(
                output_dir, f"){track}/vel_{str(test_indice).zfill(3)}.npy"
            )
            logger.info(
                f"saving *.npy file for [{track}] leaderboard : {npy_leaderboard}"
            )
            np.save(npy_leaderboard, v)

        if "cd" in out_dict:
            cd = out_dict["cd"].item()
            test_indice = datamodule.test_indices[i]
            cd_list.append([i, cd])

        # check csv in ./output
        with open(
            osp.join(output_dir, f"{cfg.project_name}.csv"), "w", newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerows(data_list)

    if "cd" in out_dict:
        titles = ["", "Cd"]
        df = pd.DataFrame(cd_list, columns=titles)
        df.to_csv(osp.join(output_dir, f"{track}/Answer.csv"), index=False)


def leader_board(cfg: DictConfig, track: str):
    os.makedirs(osp.join(cfg.output_dir, track), exist_ok=True)
    model = instantiate_network(cfg)
    save_load.load_pretrain(
        model,
        osp.join(
            cfg.output_dir,
            f"model-{cfg.model}-{cfg.track}-{cfg.num_epochs - 1}.pdparams",
        ),
    )
    logger.info(f"\n-------Starting Evaluation over [{cfg.track}] --------")
    cfg.n_train = 1
    tic = default_timer()

    cfg.mode = "test"
    eval(
        cfg.output_dir,
        model,
        instantiate_datamodule(cfg),
        cfg,
        loss_fn=lambda x, y: 0,
        track=track,
    )
    time_cost = default_timer() - tic
    logger.info(f"Inference over [Dataset_1 pressure] took {time_cost:.2f} seconds.")


if __name__ == "__main__":
    cfg_p = OmegaConf.load("configs/UnetShapeNetCar.yaml")
    os.makedirs(cfg_p.output_dir, exist_ok=True)
    cfg_p.n_test = 50
    train(cfg_p)

    cfg_cd = OmegaConf.load("configs/Unet_Cd.yaml")
    index_list = np.loadtxt(
        "./data/Training/Dataset_2/Label_File/dataset2_train_label.csv",
        delimiter=",",
        dtype=str,
        encoding="utf-8",
    )[:, 1][1:]
    cfg_cd.train_index_list = index_list[:10].tolist()
    cfg_cd.test_index_list = index_list[500:550].tolist()
    train(cfg_cd)

    # test on leader_board, or do evaluation by yourself
    leader_board(cfg_cd, "Gen_Answer")
    leader_board(cfg_p, "Gen_Answer")
    os.system("zip -r -j ./output/Gen_Answer.zip ./output/Gen_Answer")
