from math import exp

import matplotlib.pyplot as plt
import numpy as np
import paddle
from matplotlib.colors import ListedColormap
from paddle.vision.transforms import Compose

rainbow_cmap = ListedColormap(np.load("rainbow256.npy"))


def plot_velocity(output, target, path, vmin=None, vmax=None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    if vmin is None or vmax is None:
        vmax, vmin = np.max(target), np.min(target)
    im = ax[0].matshow(output, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
    ax[0].set_title("Prediction", y=1.08)
    ax[1].matshow(target, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
    ax[1].set_title("Ground Truth", y=1.08)

    for axis in ax:
        axis.set_xticks(range(0, 70, 10))
        axis.set_xticklabels(range(0, 700, 100))
        axis.set_yticks(range(0, 70, 10))
        axis.set_yticklabels(range(0, 700, 100))

        axis.set_ylabel("Depth (m)", fontsize=12)
        axis.set_xlabel("Offset (m)", fontsize=12)

    fig.colorbar(im, ax=ax, shrink=0.75, label="Velocity(m/s)")
    plt.savefig(path, format="jpg")
    plt.close("all")


def create_transform(ctx, k):
    log_data_min = log_transform(ctx["data_min"], k)
    log_data_max = log_transform(ctx["data_max"], k)
    transform_data = Compose(
        [LogTransform(k), MinMaxNormalize(log_data_min, log_data_max)]
    )
    transform_label = Compose([MinMaxNormalize(ctx["label_min"], ctx["label_max"])])

    return transform_data, transform_label


def minmax_normalize(vid, vmin, vmax, scale=2):
    vid -= vmin
    vid /= vmax - vmin
    return (vid - 0.5) * 2 if scale == 2 else vid


class MinMaxNormalize(object):
    def __init__(self, datamin, datamax, scale=2):
        self.datamin = datamin
        self.datamax = datamax
        self.scale = scale

    def __call__(self, vid):
        return minmax_normalize(vid, self.datamin, self.datamax, self.scale)


class LogTransform(object):
    def __init__(self, k=1, c=0):
        self.k = k
        self.c = c

    def __call__(self, data):
        return log_transform(data, k=self.k, c=self.c)


def log_transform(data, k=1, c=0):
    return np.log1p(np.abs(k * data) + c) * np.sign(data)


def gaussian(window_size, sigma):
    gauss = paddle.to_tensor(
        data=[
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ],
        dtype="float32",
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = (
        paddle.mm(_1D_window, _1D_window.t())
        .astype("float32")
        .unsqueeze(0)
        .unsqueeze(0)
    )
    window = _2D_window.expand([channel, 1, window_size, window_size])
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = paddle.nn.functional.conv2d(
        x=img1, weight=window, padding=window_size // 2, groups=channel
    )
    mu2 = paddle.nn.functional.conv2d(
        x=img2, weight=window, padding=window_size // 2, groups=channel
    )

    mu1_sq = mu1.pow(y=2)
    mu2_sq = mu2.pow(y=2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        paddle.nn.functional.conv2d(
            x=img1 * img1, weight=window, padding=window_size // 2, groups=channel
        )
        - mu1_sq
    )
    sigma2_sq = (
        paddle.nn.functional.conv2d(
            x=img2 * img2, weight=window, padding=window_size // 2, groups=channel
        )
        - mu2_sq
    )
    sigma12 = (
        paddle.nn.functional.conv2d(
            x=img1 * img2, weight=window, padding=window_size // 2, groups=channel
        )
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = (
        (2 * mu1_mu2 + C1)
        * (2 * sigma12 + C2)
        / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(axis=1).mean(axis=1).mean(axis=1)


class SSIM(paddle.nn.Layer):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.shape

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.place.is_gpu_place():
                window = window.cuda(img1.place.gpu_device_id())
            window = window.astype(img1.dtype)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    _, channel, _, _ = tuple(img1.shape)
    window = create_window(window_size, channel)
    if "gpu" in str(img1.place):
        window = window.cuda(device_id=img1.place.gpu_device_id(), blocking=True)
    window = window.astype(dtype=img1.dtype)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def ssim_metirc(output_dict, label_dict):
    ssim_loss = SSIM(window_size=11)
    metric_dict = {}

    for key in label_dict:
        ssim = ssim_loss(label_dict[key] / 2 + 0.5, output_dict[key] / 2 + 0.5)
        metric_dict[key] = ssim

    return metric_dict


class GenFuncs:
    def __init__(self):
        self.model_dis = None

    def loss_func_gen(self, output_dict, label_dict, weight_dict):
        l1loss = paddle.nn.L1Loss()
        l2loss = paddle.nn.MSELoss()

        pred = output_dict["fake_image"]
        label = label_dict["real_image"]

        loss_g1v = l1loss(pred, label)
        loss_g2v = l2loss(pred, label)

        loss = (
            weight_dict["lambda_g1v"][0] * loss_g1v
            + weight_dict["lambda_g2v"][0] * loss_g2v
        )
        if self.model_dis is not None:
            loss_adv = -paddle.mean(self.model_dis({"image": pred})["score"])
            loss += weight_dict["lambda_adv"][0] * loss_adv

        return {"loss_g": loss}


class DisFuncs:
    def __init__(self):
        self.model_dis = None

    def compute_gradient_penalty(self, real_samples, fake_samples):
        alpha = paddle.rand([real_samples.shape[0], 1, 1, 1], dtype=real_samples.dtype)
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.stop_gradient = False  # Allow gradients to be calculated
        d_interpolates = self.model_dis({"image": interpolates})["score"]

        gradients = paddle.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=paddle.ones(
                [real_samples.shape[0], d_interpolates.shape[1]], dtype="float32"
            ),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.reshape([gradients.shape[0], -1])
        gradient_penalty = paddle.mean((paddle.norm(gradients, p=2, axis=1) - 1) ** 2)
        return gradient_penalty

    def loss_func_dis(self, output_dict, label_dict, weight_dict):

        pred = output_dict["fake_image"]
        label = label_dict["real_image"]

        gradient_penalty = self.compute_gradient_penalty(label, pred)
        loss_real = paddle.mean(self.model_dis({"image": label})["score"])
        loss_fake = paddle.mean(self.model_dis({"image": pred})["score"])
        loss = -loss_real + loss_fake + gradient_penalty * weight_dict["lambda_gp"][0]
        return {"loss_d": loss}
