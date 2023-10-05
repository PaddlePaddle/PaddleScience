import numpy as np
import paddle
from paddle.io import Dataset

from ppsci.constraint.base import Constraint

# from ppsci.validate.base import Validator


def augmentation(input, label):
    """Apply random transformation from D4 symmetry group
    # Arguments
        x_batch, y_batch: input tensors of size `(batch_size, any, height, width)`
    """
    X = paddle.to_tensor(input["input"])
    Y = paddle.to_tensor(label["output"])
    n_obj = len(X)
    indices = np.arange(n_obj)
    np.random.shuffle(indices)

    if len(X.shape) == 3:
        # random horizontal flip
        if np.random.random() > 0.5:
            X = paddle.flip(X, axis=2)
            Y = paddle.flip(Y, axis=2)
        # random vertical flip
        if np.random.random() > 0.5:
            X = paddle.flip(X, axis=1)
            Y = paddle.flip(Y, axis=1)
        # random 90* rotation
        if np.random.random() > 0.5:
            new_perm = list(range(len(X.shape)))
            new_perm[1], new_perm[2] = new_perm[2], new_perm[1]
            X = paddle.transpose(X, perm=new_perm)
            Y = paddle.transpose(Y, perm=new_perm)
        X = X.reshape([1] + X.shape)
        Y = Y.reshape([1] + Y.shape)
    else:
        # random horizontal flip
        batch_size = X.shape[0]
        mask = np.random.random(size=batch_size) > 0.5
        X[mask] = paddle.flip(X[mask], axis=3)
        Y[mask] = paddle.flip(Y[mask], axis=3)
        # random vertical flip
        mask = np.random.random(size=batch_size) > 0.5
        X[mask] = paddle.flip(X[mask], axis=2)
        Y[mask] = paddle.flip(Y[mask], axis=2)
        # random 90* rotation
        mask = np.random.random(size=batch_size) > 0.5
        new_perm = list(range(len(X.shape)))
        new_perm[2], new_perm[3] = new_perm[3], new_perm[2]
        X[mask] = paddle.transpose(X[mask], perm=new_perm)
        Y[mask] = paddle.transpose(Y[mask], perm=new_perm)

    return X, Y


def batch_transform_wrapper(sampler):
    def batch_transform_fun(batch):
        batch_input = paddle.to_tensor([])
        batch_label = paddle.to_tensor([])
        k = sampler()
        for i in range(len(batch)):
            x1 = batch[i][0][:, k, :, :]
            x2 = batch[i][0][:, k - 1, :, :]
            x = paddle.stack((x1, x1 - x2), axis=1)
            batch_input = paddle.concat((batch_input, x), axis=0)
            batch_label = paddle.concat((batch_label, batch[i][1]), axis=0)
        return ({"input": batch_input}, {"output": batch_label}, {})

    return batch_transform_fun


class NewNamedArrayDataset(Dataset):
    def __init__(
        self,
        input,
        label,
        weight=None,
        transforms=None,
    ):
        super().__init__()
        self.input = input
        self.label = label
        self.input_keys = tuple(input.keys())
        self.label_keys = tuple(label.keys())
        self.weight = {} if weight is None else weight
        self.transforms = transforms
        self._len = len(next(iter(input.values())))

    def __getitem__(self, idx):
        input_item = {key: value[idx] for key, value in self.input.items()}
        label_item = {key: value[idx] for key, value in self.label.items()}
        weight_item = {key: value[idx] for key, value in self.weight.items()}

        ##### Transforms may be applied on label and weight.
        if self.transforms is not None:
            input_item, label_item = self.transforms(input_item, label_item)

        return (input_item, label_item, weight_item)

    def __len__(self):
        return self._len


class NewSupConstraint(Constraint):
    def __init__(
        self,
        dataset,
        data_loader,
        loss,
        output_expr=None,
        name: str = "sup_constraint",
    ):
        ##### build dataset
        _dataset = dataset

        self.input_keys = _dataset.input_keys
        self.output_keys = (
            tuple(output_expr.keys())
            if output_expr is not None
            else _dataset.label_keys
        )

        self.output_expr = output_expr
        if self.output_expr is None:
            self.output_expr = {
                key: (lambda out, k=key: out[k]) for key in self.output_keys
            }

        ##### construct dataloader with dataset and dataloader_cfg
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)
        self.loss = loss
        self.name = name

    def __str__(self):
        return ", ".join(
            [
                self.__class__.__name__,
                f"name = {self.name}",
                f"input_keys = {self.input_keys}",
                f"output_keys = {self.output_keys}",
                f"output_expr = {self.output_expr}",
                f"loss = {self.loss}",
            ]
        )


# class NewSupValidator(Validator):
#     def __init__(
#         self,
#         dataset,
#         data_loader,
#         loss,
#         output_expr = None,
#         metric = None,
#         name = "sup_validator",
#     ):
#         self.output_expr = output_expr

#         ##### build dataset
#         _dataset = dataset

#         self.input_keys = _dataset.input_keys
#         self.output_keys = (
#             tuple(output_expr.keys())
#             if output_expr is not None
#             else _dataset.label_keys
#         )

#         if self.output_expr is None:
#             self.output_expr = {
#                 key: lambda out, k=key: out[k] for key in self.output_keys
#             }

#         ##### construct dataloader with dataset and dataloader_cfg
#         self.data_loader = data_loader
#         self.data_iter = iter(self.data_loader)
#         self.loss = loss
#         self.metric = metric
#         self.name = name

#     def __str__(self):
#         return ", ".join(
#             [
#                 self.__class__.__name__,
#                 f"name = {self.name}",
#                 f"input_keys = {self.input_keys}",
#                 f"output_keys = {self.output_keys}",
#                 f"output_expr = {self.output_expr}",
#                 f"len(dataloader) = {len(self.data_loader)}",
#                 f"loss = {self.loss}",
#                 f"metric = {list(self.metric.keys())}",
#             ]
#         )
