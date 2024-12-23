import sys
from utils import paddle_aux
import paddle
from tqdm import *


class IdentityTransformer:

    def __init__(self, X):
        self.mean = X.mean(axis=0, keepdim=True)
        self.std = X.std(axis=0, keepdim=True) + 1e-08

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda(blocking=True)
        self.std = self.std.cuda(blocking=True)

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

    def encode(self, x):
        return x

    def decode(self, x):
        return x


class UnitTransformer:

    def __init__(self, X):
        self.mean = X.mean(axis=(0, 1), keepdim=True)
        self.std = X.std(axis=(0, 1), keepdim=True) + 1e-08

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda(blocking=True)
        self.std = self.std.cuda(blocking=True)

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

    def encode(self, x):
        x = (x - self.mean) / self.std
        return x

    def decode(self, x):
        return x * self.std + self.mean

    def transform(self, X, inverse=True, component='all'):
        if component == 'all' or 'all-reduce':
            if inverse:
                orig_shape = tuple(X.shape)
                return (X * (self.std - 1e-08) + self.mean).view(orig_shape)
            else:
                return (X - self.mean) / self.std
        elif inverse:
            orig_shape = tuple(X.shape)
            return (X * (self.std[:, component] - 1e-08) + self.mean[:,
                component]).view(orig_shape)
        else:
            return (X - self.mean[:, component]) / self.std[:, component]


class UnitGaussianNormalizer(object):

    def __init__(self, x, eps=1e-05, time_last=True):
        super(UnitGaussianNormalizer, self).__init__()
        self.mean = paddle.mean(x=x, axis=0)
        self.std = paddle.std(x=x, axis=0)
        self.eps = eps
        self.time_last = time_last

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps
            mean = self.mean
        else:
            if self.mean.ndim == sample_idx.ndim or self.time_last:
                std = self.std[sample_idx] + self.eps
                mean = self.mean[sample_idx]
            if self.mean.ndim > sample_idx.ndim and not self.time_last:
                std = self.std[..., sample_idx] + self.eps
                mean = self.mean[..., sample_idx]
        x = x * std + mean
        return x

    def to(self, device):
        if paddle.is_tensor(x=self.mean):
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        else:
            self.mean = paddle.to_tensor(data=self.mean).to(device)
            self.std = paddle.to_tensor(data=self.std).to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda(blocking=True)
        self.std = self.std.cuda(blocking=True)

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
