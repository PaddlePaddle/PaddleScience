import os
import random
from collections import abc

import matplotlib.pyplot as plt
import numpy as np
import paddle
import utils.paddle_aux as paddle_aux  # NOQA
from mpl_toolkits.mplot3d import Axes3D


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.place
    B = tuple(points.shape)[0]
    view_shape = list(tuple(idx.shape))
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(tuple(idx.shape))
    repeat_shape[0] = 1
    batch_indices = (
        paddle.arange(dtype="int64", end=B)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def fps(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.place
    B, N, C = tuple(xyz.shape)
    centroids = paddle.zeros(shape=[B, npoint], dtype="int64").to(device)
    distance = paddle.ones(shape=[B, N]).to(device) * 10000000000.0
    farthest = paddle.randint(low=0, high=N, shape=(B,), dtype="int64").to(device)
    batch_indices = paddle.arange(dtype="int64", end=B).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = paddle.sum(x=(xyz - centroid) ** 2, axis=-1)
        distance = paddle_aux.min(distance, dist)
        farthest = paddle_aux.max(distance, -1)[1]
    return index_points(xyz, centroids)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_lambda_sche(opti, config):
    if config.get("decay_step") is not None:

        def lr_lbmd(e):
            return max(config.lr_decay ** (e / config.decay_step), config.lowest_decay)

        tmp_lr = paddle.optimizer.lr.LambdaDecay(
            lr_lambda=lr_lbmd, learning_rate=opti.get_lr()
        )
        opti.set_lr_scheduler(tmp_lr)
        scheduler = tmp_lr
    else:
        raise NotImplementedError()
    return scheduler


def build_lambda_bnsche(model, config):
    if config.get("decay_step") is not None:

        def bnm_lmbd(e):
            return max(
                config.bn_momentum * config.bn_decay ** (e / config.decay_step),
                config.lowest_decay,
            )

        bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd)
    else:
        raise NotImplementedError()
    return bnm_scheduler


def set_random_seed(seed):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
    """
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed=seed)


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.
    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.
    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(
            m, (paddle.nn.BatchNorm1D, paddle.nn.BatchNorm2D, paddle.nn.BatchNorm3D)
        ):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):
    def __init__(self, model, bn_lambda, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, paddle.nn.Layer):
            raise RuntimeError(
                "Class '{}' is not a Paddle nn Layer".format(type(model).__name__)
            )
        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)


def seprate_point_cloud(xyz, num_points, crop, fixed_points=None, padding_zeros=False):
    """
    seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    """
    _, n, c = tuple(xyz.shape)
    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None
    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop, list):
            num_crop = random.randint(crop[0], crop[1])
        else:
            num_crop = crop
        points = points.unsqueeze(axis=0)
        if fixed_points is None:
            center = paddle.nn.functional.normalize(
                x=paddle.randn(shape=[1, 1, 3]), p=2, axis=-1
            ).cuda(blocking=True)
        else:
            if isinstance(fixed_points, list):
                fixed_point = random.sample(fixed_points, 1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1, 1, 3).cuda(blocking=True)
        distance_matrix = paddle.linalg.norm(
            x=center.unsqueeze(axis=2) - points.unsqueeze(axis=1), p=2, axis=-1
        )
        idx = paddle.argsort(x=distance_matrix, axis=-1, descending=False)[0, 0]
        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] = input_data[0, idx[:num_crop]] * 0
        else:
            input_data = points.clone()[0, idx[num_crop:]].unsqueeze(axis=0)
        crop_data = points.clone()[0, idx[:num_crop]].unsqueeze(axis=0)
        if isinstance(crop, list):
            INPUT.append(fps(input_data, 2048))
            CROP.append(fps(crop_data, 2048))
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)
    input_data = paddle.concat(x=INPUT, axis=0)
    crop_data = paddle.concat(x=CROP, axis=0)
    return input_data, crop_data


def get_ptcloud_img(ptcloud):
    fig = plt.figure(figsize=(8, 8))
    x = ptcloud
    perm_10 = list(range(x.ndim))
    perm_10[1] = 0
    perm_10[0] = 1
    x, z, y = x.transpose(perm=perm_10)
    ax = fig.gca(projection=Axes3D.name, adjustable="box")
    ax.axis("off")
    ax.view_init(30, 45)
    max, min = np.max(ptcloud), np.min(ptcloud)
    ax.set_xbound(min, max)
    ax.set_ybound(min, max)
    ax.set_zbound(min, max)
    ax.scatter(x, y, z, zdir="z", c=x, cmap="jet")
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img


def visualize_KITTI(
    path,
    data_list,
    titles=["input", "pred"],
    cmap=["bwr", "autumn"],
    zdir="y",
    xlim=(-1, 1),
    ylim=(-1, 1),
    zlim=(-1, 1),
):
    fig = plt.figure(figsize=(6 * len(data_list), 6))
    for i in range(len(data_list)):
        ax = fig.add_subplot(1, len(data_list), i + 1, projection="3d")
        ax.view_init(30, -120)
        ax.set_title(titles[i])
        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.2, hspace=0)
    if not os.path.exists(path):
        os.makedirs(path)
    pic_path = path + ".png"
    fig.savefig(pic_path)
    np.save(os.path.join(path, "input.npy"), data_list[0].numpy())
    np.save(os.path.join(path, "pred.npy"), data_list[1].numpy())
    plt.close(fig)


def random_dropping(pc, e):
    up_num = max(64, 768 // (e // 50 + 1))
    pc = pc
    random_num = paddle.randint(low=1, high=up_num, shape=(1, 1))[0, 0]
    pc = fps(pc, random_num)
    padding = paddle.zeros(shape=[pc.shape[0], 2048 - pc.shape[1], 3]).to(pc.place)
    pc = paddle.concat(x=[pc, padding], axis=1)
    return pc


def random_scale(partial, scale_range=[0.8, 1.2]):
    scale = (
        paddle.rand(shape=[1]).cuda(blocking=True) * (scale_range[1] - scale_range[0])
        + scale_range[0]
    )
    return partial * scale
