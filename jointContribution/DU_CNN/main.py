import os

import paddle
from models.network_model_batch import Networkn
from paddle import nn
from sklearn.model_selection import train_test_split
from utils import util
from utils.cosine_annealing_with_warmup import CosineAnnealingWarmUpRestarts


def main():
    # folder to load config file
    CONFIG_PATH = "./"
    # Function to load yaml configuration file
    config = util.load_config(CONFIG_PATH, "config.yaml")
    # data directory
    DATA_PATH, DATA_NAME, PODDATA_NAME = (
        config["data_directory"],
        config["data_name"],
        config["pod_data_name"],
    )
    # structure and training hyperparameters
    nlayer, nkernel, nchannel, ndownsample, lossnum = (
        config["layers"],
        config["kernels"],
        config["channels"],
        config["downsamples"],
        config["lossnum"],
    )
    print(
        f"nl: {nlayer}, nk: {nkernel}, nc: {nchannel}, nd: {ndownsample}, lossnum: {lossnum}"
    )

    # load data
    data, poddata = util.get_dataset(DATA_PATH, DATA_NAME, PODDATA_NAME)
    # split data case
    trainnum, testnum = train_test_split(
        range(len(data["shortdata"])), test_size=0.1, random_state=2
    )
    # dataset loading
    trainloader, testloader = util.get_dataloader(config, data, trainnum, testnum)
    # make directory for save the results
    FPATH = util.setup_log_directory(config)

    # model initailization
    criterion = nn.MSELoss()
    model = Networkn(
        nlayer, ndownsample, nkernel, nchannel, in_nc=1, out_nc=1, act_mode="BR"
    )
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(),
        learning_rate=float(config["learning_rate"]),
        weight_decay=1e-6,
    )
    if config["Scheduler"] == 1:
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer, T_0=50, T_mult=1, eta_max=0.005, T_up=10, gamma=0.1
        )
    else:
        scheduler = False
    epoch, loss_, evalloss_, Timecal = (
        0,
        [],
        [],
        [],
    )  # train loss, test loss, calculation time

    # load pretrained model
    if config["Load_Data"] == 1 and os.path.isfile(
        os.path.join(FPATH, "checkpoint.ckpt")
    ):
        model, optimizer, epoch, loss_ = util.load_network(
            os.path.join(FPATH, "checkpoint.ckpt")
        )

    # training
    total_epochs = config["num_epochs"] + 1
    while epoch < total_epochs:
        util.train_one_epoch(
            epoch,
            model,
            trainloader,
            optimizer,
            criterion,
            scheduler,
            config,
            poddata,
            loss_,
            evalloss_,
            Timecal,
        )
        if epoch % 10 == 0:
            util.evaluate_testloss(model, testloader, criterion, evalloss_)
            util.save_figures(os.path.join(FPATH, "logs"), epoch, model, testnum, data)
        epoch += 1

    # Save model
    util.save_model(
        epoch, model, optimizer, loss_, evalloss_, testnum, trainnum, Timecal, FPATH
    )
    util.save_figures(FPATH, epoch, model, testnum, data)
    util.save_lossfigure(FPATH, loss_, evalloss_)
    paddle.device.cuda.empty_cache()


if __name__ == "__main__":
    main()
