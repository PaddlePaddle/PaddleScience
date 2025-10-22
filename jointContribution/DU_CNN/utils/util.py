import os

import mat73
import matplotlib.pyplot as plt
import numpy as np
import paddle
import yaml
from paddle import io
from tqdm import tqdm
from utils import mydataset


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def get_dataset(data_path, datafile_name, podfile_name):
    data = {}
    poddata = {}
    specfile = mat73.loadmat(os.path.join(data_path, datafile_name))
    data["wv"] = np.transpose(specfile["xframe"])
    data["shortdata"] = specfile["val"]
    data["longdata"] = specfile["vallong"]
    podfile = mat73.loadmat(os.path.join(data_path, podfile_name))

    poddata["snapshot_mean"] = paddle.to_tensor(
        np.array(podfile["Snapshot_mean"]), dtype=paddle.get_default_dtype()
    )
    poddata["V"] = paddle.to_tensor(
        np.array(podfile["V"]), dtype=paddle.get_default_dtype()
    )
    poddata["Dvbd"] = paddle.to_tensor(
        np.array(podfile["DvBoundary"]), dtype=paddle.get_default_dtype()
    )
    return data, poddata


def get_dataloader(config, data, trainnum, testnum):
    trainloader = io.DataLoader(
        mydataset.MapDataset(data["shortdata"], data["longdata"], trainnum),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )

    testloader = io.DataLoader(
        mydataset.MapDataset(data["shortdata"], data["longdata"], testnum),
        batch_size=2,
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
    output_logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(output_logs_dir, exist_ok=True)
    return output_dir


def load_network(path, model, optimizer):
    checkpoint = paddle.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    return model, optimizer, epoch, loss


def train_one_epoch(
    epoch, model, trainloader, optimizer, criterion, scheduler, config, poddata, loss_
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
        data_count = 0
        for data, data_label in trainloader:
            data_count = data_count + len(data)
            tq.update(1)
            if config["Randomintensity"] == 1:
                randint = paddle.to_tensor(
                    np.random.uniform(0.8, 1.2, size=(data.shape[0], 1)),
                    dtype=paddle.get_default_dtype(),
                )
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

            tq.set_postfix_str(s=f"Loss: {loss.item():.4f}")
        if config["Scheduler"]:
            scheduler.step()

        # ===================log========================
        end.record()  # Waits for everything to finish running
        loss_.append(running_loss / data_count)
        paddle.device.cuda.synchronize()
        tq.set_postfix_str(s=f"Epoch Loss: {loss_[-1]:.4f}")


def calloss(label, output, randint, criterion, poddata, lossnum):
    if lossnum == 0:
        loss = criterion(output.squeeze(), label)
    elif lossnum == 1:
        ouputpca = paddle.matmul(
            output.squeeze().subtract(randint * poddata["snapshot_mean"]),
            poddata["V"],
        )
        labelpca = paddle.matmul(
            label.subtract(randint * poddata["snapshot_mean"]), poddata["V"]
        )
        loss = 0.9 * criterion(output.squeeze(), label) + 0.1 * criterion(
            ouputpca, labelpca
        )
    elif lossnum == 2:
        ouputpca = paddle.matmul(
            output.squeeze().subtract(randint * poddata["snapshot_mean"]),
            poddata["V"],
        )
        labelpca = paddle.matmul(
            label.subtract(randint * poddata["snapshot_mean"]), poddata["V"]
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
            output.squeeze().subtract(randint * poddata["snapshot_mean"]),
            poddata["V"],
        )
        labelpca = paddle.matmul(
            label.subtract(randint * poddata["snapshot_mean"]), poddata["V"]
        )
        loss = criterion(ouputpca, labelpca)
    elif lossnum == 4:
        ouputpca = paddle.matmul(
            output.squeeze().subtract(randint * poddata["snapshot_mean"]),
            poddata["V"],
        )
        labelpca = paddle.matmul(
            label.subtract(randint * poddata["snapshot_mean"]), poddata["V"]
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
    epoch, model, optimizer, loss_, evalloss_, testnum, trainnum, output_dir
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
        },
        os.path.join(output_dir, "tut.ckpt"),
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
    evalloss_.append(eval_loss / len(testloader))


def save_figures(output_dir, epoch, model, testnum, data):
    wv = data["wv"]
    testshow = []
    testshow_label = []

    for i in range(len(testnum)):
        testshow.append(data["shortdata"][testnum[i]][0][:, 1])
        testshow_label.append(data["longdata"][testnum[i]][0][:, 1])
    testshow = paddle.to_tensor(np.array(testshow), dtype=paddle.get_default_dtype())
    testshow_label = paddle.to_tensor(
        np.array(testshow_label), dtype=paddle.get_default_dtype()
    )

    testshow_predict = model(testshow.unsqueeze(1)).numpy()
    plt.figure(figsize=(20, 10))
    for i in range(len(testnum)):
        plt.subplot(len(testnum), 4, 4 * i + 1)  # nrows=2, ncols=1, index=1
        plt.plot(wv, testshow[i].numpy())

        if i == 0:
            plt.title("input")

        if i == len(testnum) - 1:
            plt.xticks(visible=True)
            plt.xlabel("wavelength (nm)")
        else:
            plt.xticks(visible=False)

        plt.subplot(len(testnum), 4, 4 * i + 2)  # nrows=2, ncols=1, index=2
        plt.plot(wv, testshow_predict[i].squeeze())
        if i == 0:
            plt.title("output")
        if i == len(testnum) - 1:
            plt.xlabel("wavelength (nm)")
            plt.xticks(visible=True)
        else:
            plt.xticks(visible=False)

        plt.subplot(len(testnum), 4, 4 * i + 3)  # nrows=2, ncols=1, index=2
        plt.plot(wv, testshow_label[i].numpy())

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
            (testshow_predict[i].squeeze() - testshow_label[i].numpy())
            / testshow_label[i].numpy(),
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
    plt.savefig(os.path.join(output_dir, f"image_{epoch}.png"), bbox_inches="tight")
    plt.close("all")
    plt.clf()


def save_lossfigure(output_dir, loss_, evalloss_):
    plt.figure(2)
    plt.plot(range(1, len(loss_) + 1), loss_)
    plt.title("Loss")
    plt.xlabel("epoch")
    plt.plot(range(1, len(evalloss_) * 10 + 1, 10), evalloss_)
    plt.legend(["train loss", "test loss"])
    plt.yscale("log")
    plt.savefig(os.path.join(output_dir, "loss.png"), bbox_inches="tight")
    plt.close("all")
    plt.clf()
