from .data_losses import H1Loss
from .data_losses import LpLoss
from .equation_losses import BurgersEqnLoss
from .equation_losses import ICLoss
from .meta_losses import WeightedSumLoss

__all__ = ["H1Loss", "LpLoss", "BurgersEqnLoss", "ICLoss", "WeightedSumLoss"]
