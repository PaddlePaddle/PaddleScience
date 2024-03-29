import argparse
import os
import sys

import data
import mesh_utils
import models
import numpy as np
import paddle
import pgl.utils.data.dataloader as pgl_dataloader
import PIL
import su2paddle.su2_function_mpi as su2_function_mpi

import ppsci
from ppsci.utils import logger

os.environ["SU2_RUN"] = "/root/autodl-tmp/SU2_bin"
sys.path.append("/root/autodl-tmp/SU2_bin")


# GCN
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        "-e",
        default="gcn_interp",
        help="Experiment name, defaults to model name.",
    )
    parser.add_argument("--su2-config", "-sc", default="coarse.cfg")
    parser.add_argument(
        "--data-dir",
        "-d",
        default="data/NACA0012_interpolate",
        help="Directory with dataset.",
    )
    parser.add_argument(
        "--coarse-mesh",
        default="meshes/mesh_NACA0012_xcoarse.su2",
        help="Path to coarse mesh (required for CFD-GCN).",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="If specified log version doesnt exist, create it."
        " If it exists, continue from where it stopped.",
    )
    parser.add_argument(
        "--load-model", "-lm", default="", help="Load previously trained model."
    )

    parser.add_argument("--model", "-m", default="gcn", help="Which model to use.")
    parser.add_argument(
        "--max-epochs",
        "-me",
        type=int,
        default=1000,
        help="Max number of epochs to train for.",
    )
    parser.add_argument("--optim", default="adam", help="Optimizer.")
    parser.add_argument("--batch-size", "-bs", type=int, default=4)
    parser.add_argument("--learning-rate", "-lr", dest="lr", type=float, default=5e-4)
    parser.add_argument("--num-layers", "-nl", type=int, default=3)
    parser.add_argument("--num-end-convs", type=int, default=3)
    parser.add_argument("--hidden-size", "-hs", type=int, default=512)
    parser.add_argument(
        "--freeze-mesh", action="store_true", help="Do not do any learning on the mesh."
    )

    parser.add_argument(
        "--eval", action="store_true", help="Skips training, does only eval."
    )
    parser.add_argument("--profile", action="store_true", help="Run profiler.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--gpus", type=int, default=1, help="Number of gpus to use, 0 for none."
    )
    parser.add_argument(
        "--dataloader-workers",
        "-dw",
        type=int,
        default=2,
        help="Number of Paddle Dataloader workers to use.",
    )
    parser.add_argument(
        "--train-val-split",
        "-tvs",
        type=float,
        default=0.9,
        help="Percentage of training set to use for training.",
    )
    parser.add_argument(
        "--val-check-interval",
        "-vci",
        type=int,
        default=None,
        help="Run validation every N batches, " "defaults to once every epoch.",
    )
    parser.add_argument(
        "--early-stop-patience",
        "-esp",
        type=int,
        default=0,
        help="Patience before early stopping. " "Does not early stop by default.",
    )
    parser.add_argument(
        "--train-pct",
        type=float,
        default=1.0,
        help="Run on a reduced percentage of the training set,"
        " defaults to running with full data.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1],
        help="Verbosity level. Defaults to 1, 0 for quiet.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode. Doesnt write logs. Runs "
        "a single iteration of training and validation.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        default=False,
        help="Dont save any logs or checkpoints.",
    )

    args = parser.parse_args()
    args.nodename = os.uname().nodename
    if args.exp_name == "":
        args.exp_name = args.model
    if args.val_check_interval is None:
        args.val_check_interval = 1.0
    args.distributed_backend = "dp"

    return args


def collate_fn(batch_data):
    return batch_data


class Runner:
    def __init__(self, hparams):
        self.hparams = hparams
        self.step = None  # count test step because apparently Trainer doesnt
        self.criterion = paddle.nn.MSELoss()
        self.data = data.MeshAirfoilDataset(hparams.data_dir, mode="train")
        self.val_data = data.MeshAirfoilDataset(hparams.data_dir, mode="test")
        self.test_data = data.MeshAirfoilDataset(hparams.data_dir, mode="test")

        in_channels = self.data[0].node_feat["feature"].shape[-1]
        out_channels = self.data[0].y.shape[-1]
        hidden_channels = hparams.hidden_size

        if hparams.model == "cfd_gcn":
            self.model = models.CFDGCN(
                hparams.su2_config,
                self.hparams.coarse_mesh,
                fine_marker_dict=self.data.marker_dict,
                hidden_channels=hidden_channels,
                num_convs=self.hparams.num_layers,
                num_end_convs=self.hparams.num_end_convs,
                out_channels=out_channels,
                process_sim=self.data.preprocess,
                freeze_mesh=self.hparams.freeze_mesh,
            )
        elif hparams.model == "gcn":
            self.model = models.MeshGCN(
                in_channels,
                hidden_channels,
                out_channels,
                fine_marker_dict=self.data.marker_dict,
                num_layers=hparams.num_layers,
            )
        else:
            raise NotImplementedError

        # config optimizer
        self.parameters = self.model.parameters()
        if self.hparams.optim.lower() == "adam":
            self.optimizer = paddle.optimizer.Adam(
                parameters=self.parameters, learning_rate=self.hparams.lr
            )
        elif self.hparams.optim.lower() == "rmsprop":
            self.optimizer = paddle.optimizer.RMSProp(
                parameters=self.parameters, learning_rate=self.hparams.lr
            )
        elif self.hparams.optim.lower() == "sgd":
            self.optimizer = paddle.optimizer.SGD(
                parameters=self.parameters, learning_rate=self.hparams.lr
            )
        else:
            self.optimizer = paddle.optimizer.SGD(
                parameters=self.parameters, learning_rate=self.hparams.lr
            )
            logger.warning(
                f"Please confirm optimizer type: {self.hparams.optim}, we set SGD as default optimizer."
            )

        # config dataloader
        self.train_loader = self.train_dataloader()
        self.val_loader = self.val_dataloader()
        self.test_loader = self.test_dataloader()

        # config criterion
        self.criterion = paddle.nn.loss.MSELoss()

        self.sum_loss = 0.0
        self.global_step = 0

    def on_epoch_start(self):
        logger.info("------")
        self.sum_loss = 0.0

    def on_epoch_end(self):
        avg_loss = self.sum_loss / max(len(self.train_loader), 1)
        logger.info(f"train_loss:{avg_loss}, step:{self.global_step}")

    def common_step(self, graphs):
        loss = 0.0
        pred_fields = self.model(graphs)
        for idx, pred_field in enumerate(pred_fields):
            true_field = graphs[idx].y
            mse_loss = self.criterion(pred_field, true_field)
            loss += mse_loss

        loss = loss / len(graphs)
        self.global_step += 1

        return loss, pred_fields

    def training_step(self, batch, batch_idx):
        loss, pred = self.common_step(batch)
        self.sum_loss += loss.item()

        logger.info(f"batch_train_loss:{loss.item()}")

        if batch_idx == 0 and not self.hparams.no_log:
            self.log_images(
                batch[0].node_feat["feature"][:, :2],
                pred[0],
                batch[0].y,
                self.data.elems_list,
                "train",
            )

        loss.backward()
        self.optimizer.step()
        self.optimizer.clear_grad()

    def validation_step(self, batch, batch_idx):
        loss, pred = self.common_step(batch)

        if batch_idx == 0 and not self.hparams.no_log:
            self.log_images(
                batch[0].node_feat["feature"][:, :2],
                pred[0],
                batch[0].y,
                self.data.elems_list,
                "val",
            )

        return loss.item()

    def test_step(self, batch, batch_idx):
        loss, pred = self.common_step(batch)
        self.step = 0 if self.step is None else self.step
        self.step += 1

        if not self.hparams.no_log:
            for i in range(len(pred)):
                self.log_images(
                    batch[i].node_feat["feature"][:, :2],
                    pred[i],
                    batch[i].y,
                    self.data.elems_list,
                    "test",
                    i,
                    batch_idx,
                )

        return loss.item()

    def train_dataloader(self):
        train_loader = pgl_dataloader.Dataloader(
            self.data,
            batch_size=self.hparams.batch_size,
            shuffle=(
                self.hparams.train_pct == 1.0
            ),  # don't shuffle if using reduced set
            num_workers=1,
            collate_fn=collate_fn,
        )
        if self.hparams.verbose:
            logger.info(
                f"Train data: {len(self.data)} examples, "
                f"{len(train_loader)} batches."
            )
        return train_loader

    def val_dataloader(self):
        # use test data here to get full training curve for test set
        val_loader = pgl_dataloader.Dataloader(
            self.val_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn,
        )
        if self.hparams.verbose:
            logger.info(
                f"Val data: {len(self.val_data)} examples, "
                f"{len(val_loader)} batches."
            )
        return val_loader

    def test_dataloader(self):
        test_loader = pgl_dataloader.Dataloader(
            self.test_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn,
        )
        if self.hparams.verbose:
            logger.info(
                f"Test data: {len(self.test_data)} examples, "
                f"{len(test_loader)} batches."
            )
        return test_loader

    def log_images(self, nodes, pred, true, elems_list, mode, log_idx=0, epoch_idx=0):
        for field in range(pred.shape[1]):
            true_img = mesh_utils.plot_field(
                nodes, elems_list, true[:, field], title="true"
            )
            min_max = (true[:, field].min().item(), true[:, field].max().item())
            pred_img = mesh_utils.plot_field(
                nodes, elems_list, pred[:, field], title="pred", clim=min_max
            )
            os.makedirs(f"{self.hparams.model}-fig", exist_ok=True)
            img_true_name = f"{self.hparams.model}-fig/{mode}_true_f{field}_idx{log_idx}_{epoch_idx}.png"
            img_pred_name = f"{self.hparams.model}-fig/{mode}_pred_f{field}_idx{log_idx}_{epoch_idx}.png"
            im = PIL.Image.fromarray(true_img)
            im.save(img_true_name)
            im = PIL.Image.fromarray(pred_img)
            im.save(img_pred_name)

    @staticmethod
    def get_cross_prods(meshes, store_elems):
        cross_prods = [
            mesh_utils.is_counter_clock_wise(mesh[e, :2], ret_val=True)
            for mesh, elems in zip(meshes, store_elems)
            for e in elems
        ]
        return cross_prods


if __name__ == "__main__":
    paddle.set_device("gpu")
    su2_function_mpi.activate_su2_mpi(remove_temp_files=True)

    args = parse_args()
    logger.info(args)
    ppsci.utils.misc.set_random_seed(args.seed)

    trainer = Runner(args)

    # test for special epoch
    # epoch = 4
    # trainer.model.set_state_dict(
    #     paddle.load(
    #         os.path.join(f"params_{trainer.hparams.model}", f"model{epoch}.pdparams")
    #     )
    # )
    # trainer.optimizer.set_state_dict(
    #     paddle.load(
    #         os.path.join(f"params_{trainer.hparams.model}", f"opt{epoch}.pdopt")
    #     )
    # )
    # total_test_loss = []
    # for i, x in enumerate(trainer.test_loader):
    #     test_loss = trainer.test_step(x, i)
    #     total_test_loss.append(test_loss)
    # mean_test_loss = np.stack(total_test_loss).mean()
    # logger.info(f"test_loss (mean):{mean_test_loss}")

    # load model from special epoch
    # epoch = 254
    # trainer.model.set_state_dict(
    #     paddle.load(
    #         os.path.join(f"params_{trainer.hparams.model}", f"model{epoch}.pdparams")
    #     )
    # )
    # trainer.optimizer.set_state_dict(
    #     paddle.load(
    #         os.path.join(f"params_{trainer.hparams.model}", f"opt{epoch}.pdopt")
    #     )
    # )

    for epoch in range(args.max_epochs):
        logger.info(f"epoch:{epoch}")
        trainer.on_epoch_start()

        # for train
        for i, graphs in enumerate(trainer.train_loader()):
            trainer.training_step(graphs, i)

        trainer.on_epoch_end()

        # for val
        total_val_loss = []
        for i, x in enumerate(trainer.val_loader):
            val_loss = trainer.validation_step(x, i)
            total_val_loss.append(val_loss)
        mean_val_loss = np.stack(total_val_loss).mean()
        logger.info(f"val_loss (mean):{mean_val_loss}")

        # for test
        total_test_loss = []
        for i, x in enumerate(trainer.test_loader):
            test_loss = trainer.test_step(x, i)
            total_test_loss.append(test_loss)
        mean_test_loss = np.stack(total_test_loss).mean()
        logger.info(f"test_loss (mean):{mean_test_loss}")

        os.makedirs(f"params_{trainer.hparams.model}", exist_ok=True)
        paddle.save(
            trainer.model.state_dict(),
            os.path.join(f"params_{trainer.hparams.model}", f"model{epoch}.pdparams"),
        )
        paddle.save(
            trainer.optimizer.state_dict(),
            os.path.join(f"params_{trainer.hparams.model}", f"opt{epoch}.pdopt"),
        )
