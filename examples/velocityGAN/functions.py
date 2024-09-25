from math import exp

import matplotlib.pyplot as plt
import numpy as np
import paddle
from matplotlib.colors import ListedColormap
from paddle.vision.transforms import Compose

rainbow_cmap = ListedColormap(np.load("rainbow256.npy"))


class GenFuncs:
    """Defined a class for conveniently calculating the generator's loss.

    Args:
        model_dis: Discriminator model.
        weight: Weight dict of loss value.
    """

    def __init__(self, model_dis, weight):
        self.model_dis = model_dis
        self.weight = weight

    def loss_func_gen(self, output_dict, label_dict, *args):
        """Calculate loss of generator.
            The loss includes L1 loss, L2 loss, and adversarial loss. Each of these losses has a corresponding weight,
            and if the weight of any loss is zero, it means that this loss component is not added during training.

        Args:
            output_dict: Output dict of model.
            label_dict: Label dict.

        Returns:
            Loss of generator.
        """
        l1loss = paddle.nn.L1Loss()
        l2loss = paddle.nn.MSELoss()

        pred = output_dict["fake_image"]
        label = label_dict["real_image"]

        loss_g1v = l1loss(pred, label)
        loss_g2v = l2loss(pred, label)

        loss = (
            self.weight["lambda_g1v"] * loss_g1v + self.weight["lambda_g2v"] * loss_g2v
        )

        loss_adv = -paddle.mean(self.model_dis({"image": pred})["score"])

        loss += self.weight["lambda_adv"] * loss_adv

        return {"loss_g": loss}


class DisFuncs:
    """Defined a class for conveniently calculating the discriminator's loss.

    Args:
        model_dis: Discriminator model.
        weight: Weight dict of loss value.
    """

    def __init__(self, model_dis, weight):
        self.model_dis = model_dis
        self.weight = weight

    def loss_func_dis(self, output_dict, label_dict, *args):
        """Calculate loss of discriminator.
            The discriminator's loss includes Wasserstein loss and gradient penalty, and only the gradient penalty has a weight parameter.

        Args:
            output_dict: Output dict of model.
            label_dict: Label dict.

        Returns:
            Loss of discriminator.
        """
        pred = output_dict["fake_image"]
        pred.stop_gradient = True
        label = label_dict["real_image"]

        gradient_penalty = self.compute_gradient_penalty(label, pred)

        loss_real = paddle.mean(self.model_dis({"image": label})["score"])
        loss_fake = paddle.mean(self.model_dis({"image": pred})["score"])

        loss = -loss_real + loss_fake + gradient_penalty * self.weight["lambda_gp"]

        return {"loss_d": loss}

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculate the gradient penalty.
            Generate a random interpolation factor, create mixed samples, process through the discriminator,
            compute the gradient of the output, apply L2 norm and constrain it to 1, and finally obtain the gradient penalty.

        Args:
            real_samples: Ground truth data from dataset.
            fake_samples: Generated data from generator.

        Returns:
            Gradient penalty.
        """
        alpha = paddle.rand([real_samples.shape[0], 1, 1, 1], dtype=real_samples.dtype)
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.stop_gradient = False  # Allow gradients to be calculated
        d_interpolates = self.model_dis({"image": interpolates})["score"]

        gradients = paddle.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.reshape([gradients.shape[0], -1])
        gradient_penalty = paddle.mean((paddle.norm(gradients, p=2, axis=1) - 1) ** 2)
        return gradient_penalty


def plot_velocity(output, target, path, vmin=None, vmax=None):
    """Results visualization.

    Args:
        output: Generated data.
        target: Ground truth data.
        path: The path to save the result.
        vmin: Minimum speed value.
        vmax: Maximum speed value.
    """
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
    """Create transformations for data and label.

    Args:
        ctx: A dictionary that stores dataset configuration information.
        k: A factor used to scale data.

    Returns:
        transform_data, transform_label
    """
    log_data_min = log_transform(ctx["data_min"], k)
    log_data_max = log_transform(ctx["data_max"], k)
    transform_data = Compose(
        [LogTransform(k), MinMaxNormalize(log_data_min, log_data_max)]
    )
    transform_label = Compose([MinMaxNormalize(ctx["label_min"], ctx["label_max"])])

    return transform_data, transform_label


class MinMaxNormalize:
    def __init__(self, datamin, datamax, scale=2):
        self.datamin = datamin
        self.datamax = datamax
        self.scale = scale

    def __call__(self, vid):
        vid -= self.datamin
        vid /= self.datamax - self.datamin
        return (vid - 0.5) * 2 if self.scale == 2 else vid


class LogTransform:
    def __init__(self, k=1, c=0):
        self.k = k
        self.c = c

    def __call__(self, data):
        return log_transform(data, k=self.k, c=self.c)


def log_transform(data, k=1, c=0):
    return np.log1p(np.abs(k * data) + c) * np.sign(data)


class SSIM(paddle.nn.Layer):
    """
    SSIM is used to measure the similarity between two images.

    Attributes:
        window_size (int): The size of the gaussian window used for computing SSIM. Defaults to 11.
        size_average (bool): If True, the SSIM values across spatial dimensions are averaged. Defaults to True.

    Methods:
        forward(img1, img2): Computes the SSIM score between two images using a gaussian filter defined by `window`.
    """

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


def ssim_metirc(output_dict, label_dict):
    ssim_loss = SSIM(window_size=11)
    metric_dict = {}

    for key in label_dict:
        ssim = ssim_loss(label_dict[key] / 2 + 0.5, output_dict[key] / 2 + 0.5)
        metric_dict[key] = ssim

    return metric_dict
