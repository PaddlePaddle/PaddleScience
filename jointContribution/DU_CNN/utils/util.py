import os

import mat73
import matplotlib.pyplot as plt
import numpy as np
import paddle
import yaml
from paddle.io import DataLoader
from tqdm import tqdm

from .Mydataset import MapDataset


def load_config(CONFIG_PATH, config_name):
    with open(os.path.join(CONFIG_PATH + "/" + config_name)) as file:
        config = yaml.safe_load(file)
    return config


def get_dataset(DATA_PATH, datafile_name, podfile_name):

    data = {}
    poddata = {}
    specfile = mat73.loadmat(DATA_PATH + "/" + datafile_name)
    data["wv"] = np.transpose(specfile["xframe"])
    data["shortdata"] = specfile["val"]
    data["longdata"] = specfile["vallong"]
    podfile = mat73.loadmat(DATA_PATH + "/" + podfile_name)
    poddata["snapshot_mean"] = paddle.to_tensor(
        np.array(podfile["Snapshot_mean"])
    ).astype("float32")
    poddata["V"] = paddle.to_tensor(np.array(podfile["V"])).astype("float32")
    poddata["Dvbd"] = paddle.to_tensor(np.array(podfile["DvBoundary"])).astype(
        "float32"
    )

    return data, poddata


def get_dataloader(config, data, trainnum, testnum):

    trainloader = DataLoader(
        MapDataset(data["shortdata"], data["longdata"], trainnum),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=5,
        persistent_workers=True,
    )

    testloader = DataLoader(
        MapDataset(data["shortdata"], data["longdata"], testnum),
        batch_size=8,
        shuffle=False,
    )

    return trainloader, testloader


def setup_log_directory(config):
    nlayer, nkernel, nchannel, ndownsample, lossnum = (
        config["layers"],
        config["kernels"],
        config["channels"],
        config["downsamples"],
        config["lossnum"],
    )
    output_dir = f"./{nlayer}layers_{nkernel}kernel_{nchannel}channel_down{ndownsample}_loss{lossnum}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir2 = f"{output_dir}/logs"
    if not os.path.exists(output_dir2):
        os.mkdir(output_dir2)
    return output_dir


def load_network(PATH, model, optimizer):
    checkpoint = paddle.load(PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    return model, optimizer, epoch, loss


def train_one_epoch(
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
):
    start, end = paddle.device.cuda.Event(enable_timing=True), paddle.device.cuda.Event(
        enable_timing=True
    )
    start.record()
    running_loss = 0.0
    model.train()
    totalepoch = int(config["num_epochs"])
    with tqdm(total=len(trainloader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {epoch}/{totalepoch}")
        for data, data_label in trainloader:
            tq.update(1)
            if config["Randomintensity"] == 1:
                randint = paddle.to_tensor(
                    np.random.uniform(0.8, 1.2, size=(data.shape[0], 1))
                ).astype("float32")
            else:
                randint = 1
            spec = data * randint
            label = data_label * randint
            output = model(spec.unsqueeze(1))

            loss = calloss(
                label, output, randint, criterion, poddata, config["lossnum"]
            )

            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.shape[0]

            # tq.set_postfix_str(s=f"Loss: {loss:.4f}")
            tq.set_postfix_str(s="Loss: ###")
        if config["Scheduler"] == 1:
            scheduler.step()
        # print(scheduler.get_last_lr())
        # ===================log========================
        end.record()  # Waits for everything to finish running
        loss_.append(running_loss / len(trainloader.sampler))
        paddle.device.cuda.synchronize()
        tq.set_postfix_str(s=f"Epoch Loss: {loss_[-1]:.4f}")


def calloss(label, output, randint, criterion, poddata, lossnum):
    if lossnum == 0:
        loss = criterion(output.squeeze(), label)
    elif lossnum == 1:
        ouputpca = paddle.matmul(
            (output.squeeze()).subtract(randint * poddata["snapshot_mean"]),
            poddata["V"],
        )
        labelpca = paddle.matmul(
            (label).subtract(randint * poddata["snapshot_mean"]), poddata["V"]
        )
        loss = 0.9 * criterion(output.squeeze(), label) + 0.1 * criterion(
            ouputpca, labelpca
        )
    elif lossnum == 2:
        ouputpca = paddle.matmul(
            (output.squeeze()).subtract(randint * poddata["snapshot_mean"]),
            poddata["V"],
        )
        labelpca = paddle.matmul(
            (label).subtract(randint * poddata["snapshot_mean"]), poddata["V"]
        )
        normouputpca = (ouputpca - poddata["Dvbd"][:, 1]) / (
            poddata["Dvbd"][:, 0] - poddata["Dvbd"][:, 1]
        ).tile((labelpca.shape[0], 1))
        normlabelpca = (labelpca - poddata["Dvbd"][:, 1]) / (
            poddata["Dvbd"][:, 0] - poddata["Dvbd"][:, 1]
        ).tile((labelpca.shape[0], 1))
        loss = 0.9 * criterion(output.squeeze(), label) + 0.1 * criterion(
            normouputpca, normlabelpca
        )
    elif lossnum == 3:
        ouputpca = paddle.matmul(
            (output.squeeze()).subtract(randint * poddata["snapshot_mean"]),
            poddata["V"],
        )
        labelpca = paddle.matmul(
            (label).subtract(randint * poddata["snapshot_mean"]), poddata["V"]
        )
        loss = criterion(ouputpca, labelpca)
    elif lossnum == 4:
        ouputpca = paddle.matmul(
            (output.squeeze()).subtract(randint * poddata["snapshot_mean"]),
            poddata["V"],
        )
        labelpca = paddle.matmul(
            (label).subtract(randint * poddata["snapshot_mean"]), poddata["V"]
        )
        normouputpca = (ouputpca - poddata["Dvbd"][:, 1]) / (
            poddata["Dvbd"][:, 0] - poddata["Dvbd"][:, 1]
        ).tile((labelpca.shape[0], 1))
        normlabelpca = (labelpca - poddata["Dvbd"][:, 1]) / (
            poddata["Dvbd"][:, 0] - poddata["Dvbd"][:, 1]
        ).tile((labelpca.shape[0], 1))
        loss = criterion(normouputpca, normlabelpca)
    return loss


def save_model(
    epoch, model, optimizer, loss_, evalloss_, testnum, trainnum, Timecal, FPATH
):
    paddle.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss_,
            "testloss": evalloss_,
            "testnum": testnum,
            "trainnum": trainnum,
            "Timecal": Timecal,
        },
        os.path.join(FPATH, "tut.ckpt"),
    )


def evaluate_testloss(model, testloader, criterion, evalloss_):
    model.eval()
    eval_loss = 0.0
    for evaldata, eval_label in testloader:
        spec = evaldata
        label = eval_label
    with paddle.no_grad():
        out = model(spec.unsqueeze(1))
        loss = criterion(out.squeeze(), label)
    eval_loss += loss.item() * evaldata.shape[0]
    evalloss_.append(eval_loss / len(testloader.sampler))


def save_figures(FPATH, epoch, model, testnum, data):
    wv = data["wv"]
    testshow = []
    testshow_label = []

    for i in range(len(testnum)):
        testshow.append(data["shortdata"][testnum[i]][0][:, 1])
        testshow_label.append(data["longdata"][testnum[i]][0][:, 1])
    testshow = paddle.to_tensor(np.array(testshow)).astype("float32")
    testshow_label = paddle.to_tensor(np.array(testshow_label)).astype("float32")

    plt.figure(figsize=(20, 10))
    for i in range(len(testnum)):
        plt.subplot(len(testnum), 4, 4 * i + 1)  # nrows=2, ncols=1, index=1
        plt.plot(wv, testshow[i])

        if i == 0:
            plt.title("input")

        if i == len(testnum) - 1:
            plt.xticks(visible=True)
            plt.xlabel("wavelength (nm)")
        else:
            plt.xticks(visible=False)

        # plt.xlabel('wavelength (nm)')
        plt.subplot(len(testnum), 4, 4 * i + 2)  # nrows=2, ncols=1, index=2
        plt.plot(wv, model(testshow.unsqueeze(1))[i].squeeze())
        if i == 0:
            plt.title("output")
        if i == len(testnum) - 1:
            plt.xlabel("wavelength (nm)")
            plt.xticks(visible=True)
        else:
            plt.xticks(visible=False)

        plt.subplot(len(testnum), 4, 4 * i + 3)  # nrows=2, ncols=1, index=2
        plt.plot(wv, testshow_label[i])

        if i == 0:
            plt.title("real output")
        if i == len(testnum) - 1:
            plt.xlabel("wavelength (nm)")
            plt.xticks(visible=True)
        else:
            plt.xticks(visible=False)

        plt.subplot(len(testnum), 4, 4 * i + 4)  # nrows=2, ncols=1, index=2
        plt.plot(
            wv,
            (model(testshow.unsqueeze(1))[i].squeeze() - testshow_label[i])
            / testshow_label[i],
        )
        plt.ylim([-0.2, 0.2])
        if i == 0:
            plt.title("Normalized difference")
        if i == len(testnum) - 1:
            plt.xlabel("wavelength (nm)")
            plt.xticks(visible=True)
        else:
            plt.xticks(visible=False)

    plt.tight_layout()
    plt.savefig(FPATH + "/image_{}.png".format(epoch), bbox_inches="tight")
    plt.close("all")
    plt.clf()


def save_lossfigure(FPATH, loss_, evalloss_):
    plt.figure(2)
    plt.plot(range(1, len(loss_) + 1), loss_)
    #  plt.plot(loss_)
    plt.title("Loss")
    plt.xlabel("epoch")
    plt.plot(range(1, len(evalloss_) * 10 + 1, 10), evalloss_)
    plt.legend(["train loss", "test loss"])
    plt.yscale("log")
    plt.savefig(FPATH + "/loss.png", bbox_inches="tight")

    plt.close("all")
    plt.clf()
