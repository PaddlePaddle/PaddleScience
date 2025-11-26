import os

import core.metrics as Metrics
import data as Data
import hydra
import model as Model
from omegaconf import DictConfig

from ppsci.utils import logger


def train(cfg: DictConfig):

    # initialize datasets
    train_set = Data.create_dataset(cfg.datasets, "train")
    train_loader = Data.create_dataloader(train_set, cfg.datasets, "train")
    val_set = Data.create_dataset(cfg.datasets, "eval")
    val_loader = Data.create_dataloader(val_set, cfg.datasets, "eval")
    lr_min, lr_max = train_set.lr_min, train_set.lr_max
    hr_min, hr_max = train_set.hr_min, train_set.hr_max
    lat, lon = train_set.lat, train_set.lon
    logger.message("Initial Dataset Finished")

    # set model
    diffusion = Model.create_model(cfg)
    logger.message("Initial Model Finished")
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch

    n_iter = cfg.train.n_iter
    if cfg.eval.pretrained_model_path:
        logger.message(
            "Resuming training from epoch: {}, iter: {}.".format(
                current_epoch, current_step
            )
        )
    diffusion.set_new_noise_schedule(
        cfg.model.beta_schedule.train, schedule_phase=cfg.mode
    )

    logger.message("Training...")
    while current_step < n_iter:
        current_epoch += 1
        print("current_epoch", current_epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > n_iter:
                break
            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()
            if current_step % cfg["train"]["print_freq"] == 0:
                logs = diffusion.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}> ".format(
                    current_epoch, current_step
                )
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                logger.message(message)
            if current_step % cfg["train"]["val_freq"] == 0:
                avg_rmse = 0.0
                idx = 0
                result_path = "{}/{}".format(cfg["path"]["results"], current_epoch)
                os.makedirs(result_path, exist_ok=True)
                diffusion.set_new_noise_schedule(
                    cfg["model"]["beta_schedule"]["eval"], schedule_phase="eval"
                )
                for ii, val_data in enumerate(val_loader):
                    diffusion.feed_data(val_data)
                    diffusion.test(continous=False)
                    visuals = diffusion.get_current_visuals()
                    sr_tensor = Metrics.tensor2rawdata(visuals["SR"], hr_min, hr_max)
                    hr_tensor = Metrics.tensor2rawdata(visuals["HR"], hr_min, hr_max)
                    lr_tensor = Metrics.tensor2rawdata(visuals["LR"], lr_min, lr_max)
                    idx += len(sr_tensor)
                    sr_img = sr_tensor[:9:3].astype(dtype="float32").cpu().numpy()
                    hr_img = hr_tensor[:9:3].astype(dtype="float32").cpu().numpy()
                    lr_img = lr_tensor[:9:3].astype(dtype="float32").cpu().numpy()
                    Metrics.save_img(
                        (sr_img, hr_img, lr_img),
                        (lat, lon),
                        "{}/{}_{}.png".format(result_path, current_step, idx),
                    )
                    avg_rmse += Metrics.calculate_rmse_sum(sr_tensor, hr_tensor)
                avg_rmse = avg_rmse / idx
                diffusion.set_new_noise_schedule(
                    cfg["model"]["beta_schedule"]["train"], schedule_phase="train"
                )
                logger.message("# Validation # RMSE: {:.4e}".format(avg_rmse.item()))
                logger.message(
                    "<epoch:{:3d}, iter:{:8,d}> rmse: {:.4e}".format(
                        current_epoch, current_step, avg_rmse.item()
                    )
                )
            if current_step % cfg["train"]["save_checkpoint_freq"] == 0:
                logger.message("Saving models and training states.")
                diffusion.save_network(current_epoch, current_step)
    logger.message("End of training.")


def evaluate(cfg: DictConfig):

    # initialize datasets
    val_set = Data.create_dataset(cfg.datasets, "eval")
    val_loader = Data.create_dataloader(val_set, cfg.datasets, "eval")
    lr_min, lr_max = cfg.eval.lr_min, cfg.eval.lr_max
    hr_min, hr_max = cfg.eval.hr_min, cfg.eval.hr_max
    lat, lon = val_set.lat, val_set.lon
    logger.message("Initial Dataset Finished")

    # set model
    diffusion = Model.create_model(cfg)
    logger.message("Initial Model Finished")
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch

    if cfg.eval.pretrained_model_path:
        logger.message(
            "Resuming training from epoch: {}, iter: {}.".format(
                current_epoch, current_step
            )
        )

    logger.message("Evaling...")
    avg_rmse = 0.0
    idx = 0
    result_path = "{}/{}".format(cfg["path"]["results"], current_epoch)
    os.makedirs(result_path, exist_ok=True)
    diffusion.set_new_noise_schedule(
        cfg.model.beta_schedule.eval, schedule_phase=cfg.mode
    )
    for ii, val_data in enumerate(val_loader):
        diffusion.feed_data(val_data)
        diffusion.test(continous=False)
        visuals = diffusion.get_current_visuals()
        sr_tensor = Metrics.tensor2rawdata(visuals["SR"], hr_min, hr_max)
        hr_tensor = Metrics.tensor2rawdata(visuals["HR"], hr_min, hr_max)
        lr_tensor = Metrics.tensor2rawdata(visuals["LR"], lr_min, lr_max)
        idx += len(sr_tensor)
        sr_img = sr_tensor[:9:3].astype(dtype="float32").cpu().numpy()
        hr_img = hr_tensor[:9:3].astype(dtype="float32").cpu().numpy()
        lr_img = lr_tensor[:9:3].astype(dtype="float32").cpu().numpy()
        Metrics.save_img(
            (sr_img, hr_img, lr_img),
            (lat, lon),
            "{}/{}_{}.png".format(result_path, current_step, idx),
        )
        avg_rmse += Metrics.calculate_rmse_sum(sr_tensor, hr_tensor)
    avg_rmse = avg_rmse / idx
    diffusion.set_new_noise_schedule(
        cfg["model"]["beta_schedule"]["train"], schedule_phase="train"
    )
    logger.message("# Validation # RMSE: {:.4e}".format(avg_rmse.item()))
    logger.message(
        "<epoch:{:3d}, iter:{:8,d}> rmse: {:.4e}".format(
            current_epoch, current_step, avg_rmse.item()
        )
    )
    logger.message("End of evaling.")


@hydra.main(version_base=None, config_path="./conf", config_name="kmcast.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
