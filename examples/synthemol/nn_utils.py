import math
from typing import List
from typing import Union

import numpy as np
import paddle


def compute_pnorm(model: paddle.nn.Layer) -> float:
    """
    Computes the norm of the parameters of a model.

    :param model: A model.
    :return: The norm of the parameters of the model.
    """
    return math.sqrt(sum([(p.norm().item() ** 2) for p in model.parameters()]))


def compute_gnorm(model: paddle.nn.Layer) -> float:
    """
    Computes the norm of the gradients of a model.

    :param model: A model.
    :return: The norm of the gradients of the model.
    """
    return math.sqrt(
        sum(
            [
                (p.grad.norm().item() ** 2)
                for p in model.parameters()
                if p.grad is not None
            ]
        )
    )


def param_count(model: paddle.nn.Layer) -> int:
    """
    Determines number of trainable parameters.

    :param model: A model.
    :return: The number of trainable parameters in the model.
    """
    return sum(param.size for param in model.parameters() if not param.stop_gradient)


def param_count_all(model: paddle.nn.Layer) -> int:
    """
    Determines number of trainable parameters.

    :param model: A model.
    :return: The number of trainable parameters in the model.
    """
    return sum(param.size for param in model.parameters())


def index_select_ND(source: paddle.Tensor, index: paddle.Tensor) -> paddle.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in :code:`index`.

    :param source: A tensor of shape :code:`(num_bonds, hidden_size)` containing message features.
    :param index: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds)` containing the atom or bond
                  indices to select from :code:`source`.
    :return: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds, hidden_size)` containing the message
             features corresponding to the atoms/bonds specified in index.
    """
    index_size = tuple(index.shape)
    suffix_dim = tuple(source.shape)[1:]
    final_size = index_size + suffix_dim
    # print("index", index)
    target = source.index_select(axis=0, index=index.reshape(-1))
    target = target.reshape(final_size)
    return target


def get_activation_function(activation: str) -> paddle.nn.Layer:
    """
    Gets an activation function module given the name of the activation.

    Supports:

    * :code:`ReLU`
    * :code:`LeakyReLU`
    * :code:`PReLU`
    * :code:`tanh`
    * :code:`SELU`
    * :code:`ELU`

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == "ReLU":
        return paddle.nn.ReLU()
    elif activation == "LeakyReLU":
        return paddle.nn.LeakyReLU(negative_slope=0.1)
    elif activation == "PReLU":
        return paddle.nn.PReLU()
    elif activation == "tanh":
        return paddle.nn.Tanh()
    elif activation == "SELU":
        return paddle.nn.SELU()
    elif activation == "ELU":
        return paddle.nn.ELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')


def initialize_weights(model: paddle.nn.Layer) -> None:
    """
    Initializes the weights of a model in place.

    :param model: A model.
    """
    for param in model.parameters():
        if param.dim() == 1:
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(param)
        else:
            init_XavierNormal = paddle.nn.initializer.XavierNormal()
            init_XavierNormal(param)


class NoamLR(paddle.optimizer.lr.LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.

    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where :code:`warmup_steps = warmup_epochs * steps_per_epoch`).
    Then the learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr` over the
    course of the remaining :code:`total_steps - warmup_steps` (where :code:`total_steps =
    total_epochs * steps_per_epoch`). This is roughly based on the learning rate
    schedule from `Attention is All You Need <https://arxiv.org/abs/1706.03762>`_, section 5.3.
    """

    def __init__(
        self,
        optimizer: paddle.optimizer.Optimizer,
        warmup_epochs: List[Union[float, int]],
        total_epochs: List[int],
        steps_per_epoch: int,
        init_lr: List[float],
        max_lr: List[float],
        final_lr: List[float],
    ):
        """
        :param optimizer: A optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after :code:`warmup_epochs`).
        :param final_lr: The final learning rate (achieved after :code:`total_epochs`).
        """
        if (
            not len(optimizer._param_groups)
            == len(warmup_epochs)
            == len(total_epochs)
            == len(init_lr)
            == len(max_lr)
            == len(final_lr)
        ):
            raise ValueError(
                f"Number of param groups must match the number of epochs and learning rates! got: len(optimizer.param_groups)= {len(optimizer._param_groups)}, len(warmup_epochs)= {len(warmup_epochs)}, len(total_epochs)= {len(total_epochs)}, len(init_lr)= {len(init_lr)}, len(max_lr)= {len(max_lr)}, len(final_lr)= {len(final_lr)}"
            )
        self.num_lrs = len(optimizer._param_groups)
        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)
        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps
        self.exponential_gamma = (self.final_lr / self.max_lr) ** (
            1 / (self.total_steps - self.warmup_steps)
        )
        super(NoamLR, self).__init__(optimizer.get_lr())

    def get_lr(self) -> List[float]:
        """
        Gets a list of the current learning rates.

        :return: A list of the current learning rates.
        """
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.

        :param current_step: Optionally specify what step to set the learning rate to.
                             If None, :code:`current_step = self.current_step + 1`.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1
        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = (
                    self.init_lr[i] + self.current_step * self.linear_increment[i]
                )
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * self.exponential_gamma[i] ** (
                    self.current_step - self.warmup_steps[i]
                )
            else:
                self.lr[i] = self.final_lr[i]
            self.optimizer._param_groups[i]["learning_rate"] = self.lr[i]


def activate_dropout(module: paddle.nn.Layer, dropout_prob: float):
    """
    Set p of dropout layers and set to train mode during inference for uncertainty estimation.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param dropout_prob: A float on (0,1) indicating the dropout probability.
    """
    if isinstance(module, paddle.nn.Dropout):
        module.p = dropout_prob
        module.train()
