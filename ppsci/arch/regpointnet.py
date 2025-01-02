from typing import Tuple

import paddle

"""
@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu mohamed.elrefaie@tum.de

This module is part of the research presented in the paper:
"DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks".

This module is used to define both point-cloud based and graph-based models, including RegDGCNN, PointNet, and several Graph Neural Network (GNN) models
for the task of surrogate modeling of the aerodynamic drag.
"""


def transpose_aux_func(dims, dim0, dim1):
    perm = list(range(dims))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return perm


def view(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], (tuple, list, str)):
            return paddle.view(self, args[0])
        else:
            return paddle.view(self, list(args))
    elif kwargs:
        return paddle.view(self, shape_or_dtype=list(kwargs.values())[0])


setattr(paddle.Tensor, "view", view)


def min_class_func(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.minimum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.minimum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.min(self, *args, **kwargs), paddle.argmin(
                self, *args, **kwargs
            )
        else:
            ret = paddle.min(self, *args, **kwargs)

    return ret


def max_class_func(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.maximum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.max(self, *args, **kwargs), paddle.argmax(
                self, *args, **kwargs
            )
        else:
            ret = paddle.max(self, *args, **kwargs)

    return ret


setattr(paddle.Tensor, "min", min_class_func)
setattr(paddle.Tensor, "max", max_class_func)


def knn(x, k):
    """
    Computes the k-nearest neighbors for each point in x.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, num_dims, num_points).
        k (int): The number of nearest neighbors to find.

    Returns:
        torch.Tensor: Indices of the k-nearest neighbors for each point, shape (batch_size, num_points, k).
    """
    inner = -2 * paddle.matmul(
        x=x.transpose(perm=transpose_aux_func(x.ndim, 2, 1)), y=x
    )
    xx = paddle.sum(x=x**2, axis=1, keepdim=True)
    pairwise_distance = (
        -xx - inner - xx.transpose(perm=transpose_aux_func(xx.ndim, 2, 1))
    )
    idx = pairwise_distance.topk(k=k, axis=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    Constructs local graph features for each point by finding its k-nearest neighbors and
    concatenating the relative position vectors.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, num_dims, num_points).
        k (int): The number of neighbors to consider for graph construction.
        idx (torch.Tensor, optional): Precomputed k-nearest neighbor indices.

    Returns:
        torch.Tensor: The constructed graph features of shape (batch_size, 2*num_dims, num_points, k).
    """
    batch_size = x.shape[0]
    num_points = x.shape[2]
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    idx_base = paddle.arange(start=0, end=batch_size).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = tuple(x.shape)
    x = x.transpose(perm=transpose_aux_func(x.ndim, 2, 1)).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).tile(repeat_times=[1, 1, k, 1])
    feature = (
        paddle.concat(x=(feature - x, x), axis=3)
        .transpose(perm=[0, 3, 1, 2])
        .contiguous()
    )
    return feature


class RegPointNet(paddle.nn.Layer):
    """
    PointNet-based regression model for 3D point cloud data.

    Args:
        args (dict): Configuration parameters including 'emb_dims' for embedding dimensions and 'dropout' rate.

    Methods:
        forward(x): Forward pass through the network.
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        weight_keys: Tuple[str, ...],
        args,
    ):
        """
        Initialize the RegPointNet model for regression tasks with enhanced complexity,
        including additional layers and residual connections.

        Parameters:
            emb_dims (int): Dimensionality of the embedding space.
            dropout (float): Dropout probability.
        """
        super(RegPointNet, self).__init__()
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.weight_keys = weight_keys
        self.args = args
        self.conv1 = paddle.nn.Conv1D(
            in_channels=3, out_channels=512, kernel_size=1, bias_attr=False
        )
        self.conv2 = paddle.nn.Conv1D(
            in_channels=512, out_channels=1024, kernel_size=1, bias_attr=False
        )
        self.conv3 = paddle.nn.Conv1D(
            in_channels=1024, out_channels=1024, kernel_size=1, bias_attr=False
        )
        self.conv4 = paddle.nn.Conv1D(
            in_channels=1024, out_channels=1024, kernel_size=1, bias_attr=False
        )
        self.conv5 = paddle.nn.Conv1D(
            in_channels=1024, out_channels=1024, kernel_size=1, bias_attr=False
        )
        self.conv6 = paddle.nn.Conv1D(
            in_channels=1024,
            out_channels=args["emb_dims"],
            kernel_size=1,
            bias_attr=False,
        )
        self.bn1 = paddle.nn.BatchNorm1D(num_features=512)
        self.bn2 = paddle.nn.BatchNorm1D(num_features=1024)
        self.bn3 = paddle.nn.BatchNorm1D(num_features=1024)
        self.bn4 = paddle.nn.BatchNorm1D(num_features=1024)
        self.bn5 = paddle.nn.BatchNorm1D(num_features=1024)
        self.bn6 = paddle.nn.BatchNorm1D(num_features=args["emb_dims"])
        self.dropout_conv = paddle.nn.Dropout(p=args["dropout"])
        self.dropout_linear = paddle.nn.Dropout(p=args["dropout"])
        self.conv_shortcut = paddle.nn.Conv1D(
            in_channels=3, out_channels=args["emb_dims"], kernel_size=1, bias_attr=False
        )
        self.bn_shortcut = paddle.nn.BatchNorm1D(num_features=args["emb_dims"])
        self.linear1 = paddle.nn.Linear(
            in_features=args["emb_dims"], out_features=512, bias_attr=False
        )
        self.bn7 = paddle.nn.BatchNorm1D(num_features=512)
        self.linear2 = paddle.nn.Linear(
            in_features=512, out_features=256, bias_attr=False
        )
        self.bn8 = paddle.nn.BatchNorm1D(num_features=256)
        self.linear3 = paddle.nn.Linear(in_features=256, out_features=128)
        self.bn9 = paddle.nn.BatchNorm1D(num_features=128)
        self.linear4 = paddle.nn.Linear(in_features=128, out_features=64)
        self.bn10 = paddle.nn.BatchNorm1D(num_features=64)
        self.final_linear = paddle.nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        """
        Forward pass of the network.

        Parameters:
            x (Tensor): Input tensor of shape (batch_size, 3, num_points).

        Returns:
            Tensor: Output tensor of the predicted scalar value.
        """
        x = x[self.input_keys[0]]
        shortcut = self.bn_shortcut(self.conv_shortcut(x))
        x = paddle.nn.functional.relu(x=self.bn1(self.conv1(x)))
        x = self.dropout_conv(x)
        x = paddle.nn.functional.relu(x=self.bn2(self.conv2(x)))
        x = self.dropout_conv(x)
        x = paddle.nn.functional.relu(x=self.bn3(self.conv3(x)))
        x = self.dropout_conv(x)
        x = paddle.nn.functional.relu(x=self.bn4(self.conv4(x)))
        x = self.dropout_conv(x)
        x = paddle.nn.functional.relu(x=self.bn5(self.conv5(x)))
        x = self.dropout_conv(x)
        x = paddle.nn.functional.relu(x=self.bn6(self.conv6(x)))
        x = x + shortcut
        x = paddle.nn.functional.adaptive_max_pool1d(x=x, output_size=1).squeeze(
            axis=-1
        )
        x = paddle.nn.functional.relu(x=self.bn7(self.linear1(x)))
        x = paddle.nn.functional.relu(x=self.bn8(self.linear2(x)))
        x = paddle.nn.functional.relu(x=self.bn9(self.linear3(x)))
        x = paddle.nn.functional.relu(x=self.bn10(self.linear4(x)))
        x = self.final_linear(x)
        return {self.label_keys[0]: x}
