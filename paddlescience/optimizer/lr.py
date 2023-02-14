from abc import abstractmethod
from typing import Union

from paddle.optimizer import lr


class LRBase(object):
    """Base class for custom learning rates

    Args:
        epochs (int): total epoch(s)
        step_each_epoch (int): number of iterations within an epoch
        learning_rate (float): learning rate
        warmup_epoch (int): number of warmup epoch(s)
        warmup_start_lr (float): start learning rate within warmup
        last_epoch (int): last epoch
        by_epoch (bool): learning rate decays by epoch when by_epoch is True, else by iter
        verbose (bool): If True, prints a message to stdout for each update. Defaults to False
    """

    def __init__(self,
                 epochs: int,
                 step_each_epoch: int,
                 learning_rate: float,
                 warmup_epoch: int,
                 warmup_start_lr: float,
                 last_epoch: int,
                 by_epoch: bool,
                 verbose: bool=False) -> None:
        """Initialize and record the necessary parameters
        """
        super(LRBase, self).__init__()
        if warmup_epoch >= epochs:
            msg = f"When using warm up, the value of \"Global.epochs\" must be greater than value of \"Optimizer.lr.warmup_epoch\". The value of \"Optimizer.lr.warmup_epoch\" has been set to {epochs}."
            print(msg)
            warmup_epoch = epochs
        self.epochs = epochs
        self.step_each_epoch = step_each_epoch
        self.learning_rate = learning_rate
        self.warmup_epoch = warmup_epoch
        self.warmup_steps = self.warmup_epoch if by_epoch else round(
            self.warmup_epoch * self.step_each_epoch)
        self.warmup_start_lr = warmup_start_lr
        self.last_epoch = last_epoch
        self.by_epoch = by_epoch
        self.verbose = verbose

    @abstractmethod
    def __call__(self, *kargs, **kwargs) -> lr.LRScheduler:
        """generate an learning rate scheduler

        Returns:
            lr.LinearWarmup: learning rate scheduler
        """
        pass

    def linear_warmup(
            self,
            learning_rate: Union[float, lr.LRScheduler]) -> lr.LinearWarmup:
        """Add an Linear Warmup before learning_rate

        Args:
            learning_rate (Union[float, lr.LRScheduler]): original learning rate without warmup

        Returns:
            lr.LinearWarmup: learning rate scheduler with warmup
        """
        warmup_lr = lr.LinearWarmup(
            learning_rate=learning_rate,
            warmup_steps=self.warmup_steps,
            start_lr=self.warmup_start_lr,
            end_lr=self.learning_rate,
            last_epoch=self.last_epoch,
            verbose=self.verbose)
        return warmup_lr


class Constant(lr.LRScheduler):
    """Constant learning rate Class implementation

    Args:
        learning_rate (float): The initial learning rate
        last_epoch (int, optional): The index of last epoch. Default: -1.
    """

    def __init__(self, learning_rate, last_epoch=-1, **kwargs):
        self.learning_rate = learning_rate
        self.last_epoch = last_epoch
        super(Constant, self).__init__()

    def get_lr(self) -> float:
        """always return the same learning rate
        """
        return self.learning_rate


class Cosine(LRBase):
    """Cosine learning rate decay

    ``lr = 0.05 * (math.cos(epoch * (math.pi / epochs)) + 1)``

    Args:
        epochs (int): total epoch(s)
        step_each_epoch (int): number of iterations within an epoch
        learning_rate (float): learning rate
        eta_min (float, optional): Minimum learning rate. Defaults to 0.0.
        warmup_epoch (int, optional): The epoch numbers for LinearWarmup. Defaults to 0.
        warmup_start_lr (float, optional): start learning rate within warmup. Defaults to 0.0.
        last_epoch (int, optional): last epoch. Defaults to -1.
        by_epoch (bool, optional): learning rate decays by epoch when by_epoch is True, else by iter. Defaults to False.
    """

    def __init__(self,
                 epochs,
                 step_each_epoch,
                 learning_rate,
                 eta_min=0.0,
                 warmup_epoch=0,
                 warmup_start_lr=0.0,
                 last_epoch=-1,
                 by_epoch=False,
                 **kwargs):
        super(Cosine, self).__init__(epochs, step_each_epoch, learning_rate,
                                     warmup_epoch, warmup_start_lr, last_epoch,
                                     by_epoch)
        self.T_max = (self.epochs - self.warmup_epoch) * self.step_each_epoch
        self.eta_min = eta_min
        if self.by_epoch:
            self.T_max = self.epochs - self.warmup_epoch

    def __call__(self):
        learning_rate = lr.CosineAnnealingDecay(
            learning_rate=self.learning_rate,
            T_max=self.T_max,
            eta_min=self.eta_min,
            last_epoch=self.last_epoch) if self.T_max > 0 else Constant(
                self.learning_rate)

        if self.warmup_steps > 0:
            learning_rate = self.linear_warmup(learning_rate)

        setattr(learning_rate, "by_epoch", self.by_epoch)
        return learning_rate
