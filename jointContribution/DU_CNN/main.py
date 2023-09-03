import os

import paddle
from models import network_model_batch
from paddle import nn
from sklearn import model_selection
from utils import lr_scheduler
from utils import util


def main():
    # Function to load yaml configuration file
    config = util.load_config("./config.yaml")
    # data directory
    data_path, data_name, poddata_name = (
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
        f"nlayer: {nlayer}, nkernel: {nkernel}, nchannel: {nchannel}, ndownsample: {ndownsample}, lossnum: {lossnum}"
    )

    # load data
    data, poddata = util.get_dataset(data_path, data_name, poddata_name)
    # split data case
    trainnum, testnum = model_selection.train_test_split(
        range(len(data["shortdata"])), test_size=0.1, random_state=2
    )
    # dataset loading
    trainloader, testloader = util.get_dataloader(config, data, trainnum, testnum)
    # make directory for save the results
    output_dir = util.setup_log_directory(config)

    # model initialization
    criterion = nn.MSELoss()
    model = network_model_batch.Networkn(
        nlayer, ndownsample, nkernel, nchannel, in_nc=1, out_nc=1, act_mode="BR"
    )
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(),
        learning_rate=config["learning_rate"],
        weight_decay=1e-6,
    )
    if config["Scheduler"]:
        scheduler = lr_scheduler.CosineAnnealingWarmUpRestarts(
            optimizer.get_lr(), T_0=50, T_mult=1, eta_max=0.005, T_up=10, gamma=0.1
        )
    else:
        scheduler = False
    epoch, loss_, evalloss_ = (0, [], [])  # train loss, test loss

    # load pretrained model
    if config["Load_Data"] == 1 and os.path.isfile(
        os.path.join(output_dir, "checkpoint.ckpt")
    ):
        model, optimizer, epoch, loss_ = util.load_network(
            os.path.join(output_dir, "checkpoint.ckpt")
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
        )
        if epoch % 10 == 0:
            util.evaluate_testloss(model, testloader, criterion, evalloss_)
            util.save_figures(
                os.path.join(output_dir, "logs"), epoch, model, testnum, data
            )
        epoch += 1

    # Save model
    util.save_model(
        epoch, model, optimizer, loss_, evalloss_, testnum, trainnum, output_dir
    )
    util.save_figures(output_dir, epoch, model, testnum, data)
    util.save_lossfigure(output_dir, loss_, evalloss_)
    paddle.device.cuda.empty_cache()


if __name__ == "__main__":
    main()
