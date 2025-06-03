import gzip
import pickle
import random
import tarfile
from typing import Dict

import numpy as np
import paddle
from matplotlib import pyplot as plt
from paddle.nn import functional as F
from paddle.vision import transforms
from PIL import Image
from sklearn.datasets import make_swiss_roll


class Cifar10GenFuncs:
    """
    Loss function for cifar10 generator
    Args
        discriminator_model: discriminator model
        acgan_scale_g: scale of acgan loss for generator

    """

    def __init__(
        self,
        discriminator_model,
        acgan_scale_g=0.1,
    ):
        self.crossEntropyLoss = paddle.nn.CrossEntropyLoss()
        self.acgan_scale_g = acgan_scale_g
        self.discriminator_model = discriminator_model

    def loss(self, output_dict: Dict, *args):
        fake_image = output_dict["fake_data"]
        labels = output_dict["labels"]
        outputs = self.discriminator_model({"data": fake_image, "labels": labels})
        disc_fake, disc_fake_acgan = outputs["disc_fake"], outputs["disc_acgan"]
        gen_cost = -paddle.mean(disc_fake)
        if disc_fake_acgan is not None:
            gen_acgan_cost = self.crossEntropyLoss(disc_fake_acgan, labels)
            gen_cost += self.acgan_scale_g * gen_acgan_cost
        return {"loss_g": gen_cost}


class Cifar10DisFuncs:
    """
    Loss function for cifar10 discriminator
    Args
        discriminator_model: discriminator model
        acgan_scale: scale of acgan loss for discriminator

    """

    def __init__(self, discriminator_model, acgan_scale):
        self.crossEntropyLoss = paddle.nn.CrossEntropyLoss()
        self.acgan_scale = acgan_scale
        self.discriminator_model = discriminator_model

    def loss(self, output_dict: Dict, label_dict: Dict, *args):
        fake_image = output_dict["fake_data"]
        real_image = label_dict["real_data"]
        labels = output_dict["labels"]
        disc_fake = self.discriminator_model({"data": fake_image, "labels": labels})[
            "disc_fake"
        ]
        out = self.discriminator_model({"data": real_image, "labels": labels})
        disc_real, disc_real_acgan = out["disc_fake"], out["disc_acgan"]
        gradient_penalty = self.compute_gradient_penalty(real_image, fake_image, labels)
        disc_cost = paddle.mean(disc_fake) - paddle.mean(disc_real)
        disc_wgan = disc_cost + gradient_penalty
        if disc_real_acgan is not None:
            disc_acgan_cost = self.crossEntropyLoss(disc_real_acgan, labels)
            disc_acgan = disc_acgan_cost.sum()
            disc_cost = disc_wgan + (self.acgan_scale * disc_acgan)
        else:
            disc_cost = disc_wgan
        return {"loss_d": disc_cost}

    def compute_gradient_penalty(self, real_data, fake_data, labels):
        differences = fake_data - real_data
        alpha = paddle.rand([fake_data.shape[0], 1])
        interpolates = real_data + (alpha * differences)
        gradients = paddle.grad(
            outputs=self.discriminator_model({"data": interpolates, "labels": labels})[
                "disc_fake"
            ],
            inputs=interpolates,
            create_graph=True,
            retain_graph=False,
        )[0]
        slopes = paddle.sqrt(paddle.sum(paddle.square(gradients), axis=1))
        gradient_penalty = 10 * paddle.mean((slopes - 1.0) ** 2)
        return gradient_penalty


class InceptionScore:
    """
    Inception Score
    Args
        eps: epsilon to avoid log(0)
        splits: number of splits
    """

    def __init__(self, eps=1e-16, splits=10, batch_size=64):
        self.inception_v3 = paddle.vision.inception_v3(pretrained=True)
        self.inception_v3.fc.bias.set_value(
            paddle.to_tensor(np.zeros(self.inception_v3.fc.bias.shape, dtype="float32"))
        )
        self.inception_v3.eval()
        self.eps = eps
        self.splits = splits
        self.softmax = paddle.nn.Softmax(axis=1)
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def inception_score(self, output_dict: Dict, label_dict, *args):
        with paddle.no_grad():
            images = output_dict["fake_data"]
            images = images.reshape((-1, 3, 32, 32))
            images = (images + 1.0) * (255.99 / 2)
            predict = []
            for i in range(images.shape[0] // self.batch_size):
                image = images[i * self.batch_size : (i + 1) * self.batch_size]
                image = F.interpolate(image, size=(299, 299), mode="bilinear")
                image = image / 255
                image = self.transform(image)
                predict.append(self.inception_v3(image))
            else:
                image = images[(images.shape[0] // self.batch_size) * self.batch_size :]
                if image.shape[0] != 0:
                    image = F.interpolate(image, size=(299, 299), mode="bilinear")
                    image = image / 255
                    image = self.transform(image)
                    predict.append(self.inception_v3(image))
            predict = paddle.concat(predict, axis=0)
            predict = self.softmax(predict) + self.eps
            scores = []
            split_size = predict.shape[0] // self.splits
            for i in range(self.splits):
                part = predict[i * split_size : (i + 1) * split_size]
                kl = part * (paddle.log(part) - paddle.log(paddle.mean(part, 0)))
                kl = paddle.mean(paddle.sum(kl, 1))
                scores.append(paddle.exp(kl))
            scores = paddle.to_tensor(scores)
            return {"inception_score": paddle.mean(scores)}


def show_save_image(image_data, path):
    image_data = paddle.reshape(image_data, (3, 32, 32))
    image_data = (image_data + 1.0) * (255.0 / 2)
    image_data = paddle.unsqueeze(image_data, axis=0)
    image_data = paddle.squeeze(image_data, axis=0)
    image_data = paddle.transpose(image_data, [1, 2, 0])
    image_data = image_data.numpy().astype("uint8")
    image = Image.fromarray(image_data)
    image.save(path)


def load_cifar10(input_keys, label_keys, data_path):
    datas, labels = unpickle(data_path)
    datas = datas.astype("float32")
    datas_ = ((datas / 256.0) - 0.5) * 2
    random_uniform = np.random.uniform(size=[50000, 3072], low=0.0, high=1.0 / 128)
    datas_ = (datas_ + random_uniform).astype("float32")
    labels_ = np.array(labels, dtype="int32")
    labels = {label_keys[0]: datas_}
    datas = {input_keys[0]: labels_}
    return datas, labels


def unpickle(data_path):
    label = []
    data = []
    with tarfile.open(data_path, "r:gz") as tar:
        for member in tar.getmembers():
            if "data_batch_" in member.name:
                file = tar.extractfile(member)
                if file is not None:
                    dict_ = pickle.load(file, encoding="bytes")
                    label.extend(dict_[b"labels"])
                    data.append(dict_[b"data"])
    data = np.vstack(data)
    return data, label


def load_toy_data(input_keys, mode):
    data = []
    if mode == "25gaussians":
        for i in range(100000 // 25):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    data.append(point)
        data = np.array(data, dtype="float32")
        np.random.shuffle(data)
        data /= 2.828  # stdev
    elif mode == "swissroll":
        data = make_swiss_roll(n_samples=100000, noise=0.25)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 7.5  # stdev plus a little

    elif mode == "8gaussians":
        scale = 2.0
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        data = []
        for i in range(100000 // 8):
            point = np.random.randn(2) * 0.02
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            data.append(point)
        data = np.array(data, dtype="float32")
        data /= 1.414  # stdev
    data = {input_keys[0]: data}
    return data


class ToyGenFuncs:
    """
    Loss function for toy generator
    Args
        discriminator_model: discriminator model
    """

    def __init__(self, discriminator_model):
        self.discriminator_model = discriminator_model

    def loss(self, output_dict: Dict, *args):
        fake_data = output_dict["fake_data"]
        outputs = self.discriminator_model({"data": fake_data})
        disc_fake = outputs["score"]
        gen_cost = -paddle.mean(disc_fake)
        return {"loss_g": gen_cost}


class ToyDisFuncs:
    """
    Loss function for toy discriminator
    Args
        discriminator_model: discriminator model
        lamda: gradient penalty coefficient
    """

    def __init__(self, discriminator_model, lamda):
        self.discriminator_model = discriminator_model
        self.lamda = lamda

    def loss(self, output_dict: Dict, *args):
        real_data = output_dict["real_data"]
        fake_data = output_dict["fake_data"]
        disc_fake = self.discriminator_model({"data": fake_data})["score"]
        disc_real = self.discriminator_model({"data": real_data})["score"]
        gradient_penalty = self.compute_gradient_penalty(real_data, fake_data)
        disc_cost = paddle.mean(disc_fake) - paddle.mean(disc_real)
        disc_cost = disc_cost + gradient_penalty
        loss = disc_cost
        return {"loss_d": loss}

    def compute_gradient_penalty(self, real_data, fake_data):
        differences = fake_data - real_data
        alpha = paddle.rand([fake_data.shape[0], 1])
        interpolates = real_data + (alpha * differences)
        gradients = paddle.grad(
            outputs=self.discriminator_model({"data": interpolates})["score"],
            inputs=interpolates,
            create_graph=True,
            retain_graph=False,
        )[0]
        slopes = paddle.sqrt(paddle.sum(paddle.square(gradients), axis=1))
        gradient_penalty = self.lamda * paddle.mean((slopes - 1.0) ** 2)
        return gradient_penalty


def generate_toy_image(true_dist, discriminator, path):
    n_points = 128
    range_ = 3
    points = np.zeros((n_points, n_points, 2), dtype="float32")
    points[:, :, 0] = np.linspace(-range_, range_, n_points)[:, None]
    points[:, :, 1] = np.linspace(-range_, range_, n_points)[None, :]
    points = points.reshape((-1, 2))

    disc_map = (
        discriminator({"data": paddle.to_tensor(points)})["score"].numpy().reshape(-1)
    )

    plt.clf()
    x = y = np.linspace(-range_, range_, n_points)
    plt.contour(x, y, disc_map.reshape((n_points, n_points)).T)
    plt.scatter(true_dist[:, 0], true_dist[:, 1], c="orange", marker="+")
    plt.savefig(path)


class MnistGenFuncs:
    """
    Loss function for mnist generator
    Args
        discriminator_model: discriminator model
    """

    def __init__(self, discriminator_model):
        self.discriminator_model = discriminator_model

    def loss(self, output_dict: Dict, *args):
        fake_data = output_dict["fake_data"]
        score = self.discriminator_model({"data": fake_data})["score"]
        gen_cost = -paddle.mean(score)
        return {"loss_g": gen_cost}


class MnistDisFuncs:
    """
    Loss function for mnist discriminator
    Args
        discriminator_model: discriminator model
        lamda: gradient penalty coefficient
    """

    def __init__(self, discriminator_model, lamda):
        self.discriminator_model = discriminator_model
        self.lamda = lamda

    def loss(self, output_dict: Dict, *args):
        real_data = output_dict["real_data"]
        fake_data = output_dict["fake_data"]
        disc_fake = self.discriminator_model({"data": fake_data})["score"]
        disc_real = self.discriminator_model({"data": real_data})["score"]
        gradient_penalty = self.compute_gradient_penalty(real_data, fake_data)
        disc_cost = paddle.mean(disc_fake) - paddle.mean(disc_real)
        disc_cost = disc_cost + gradient_penalty
        loss = disc_cost
        return {"loss_d": loss}

    def compute_gradient_penalty(self, real_data, fake_data):
        differences = fake_data - real_data
        alpha = paddle.rand([fake_data.shape[0], 1])
        interpolates = real_data + (alpha * differences)
        gradients = paddle.grad(
            outputs=self.discriminator_model({"data": interpolates})["score"],
            inputs=interpolates,
            create_graph=True,
            retain_graph=False,
        )[0]
        slopes = paddle.sqrt(paddle.sum(paddle.square(gradients), axis=1))
        gradient_penalty = self.lamda * paddle.mean((slopes - 1.0) ** 2)
        return gradient_penalty


def load_mnist(
    data_path,
    input_keys,
):
    with gzip.open(data_path, "rb") as f:
        train_data, _, _ = pickle.load(f, encoding="latin1")
    data, _ = train_data
    data = {input_keys[0]: data}
    return data


def show_mnist(data, path):
    data = data.reshape([1, 28, 28])
    data = data * 255
    data = data.numpy().squeeze()
    data = data.astype(np.uint8)
    img = Image.fromarray(data, mode="L")  # 'L' 表示灰度图
    img.show()
    img.save(path)


def invalid_metric(*args, **kwargs):
    return {"invalid_metric": 0}
