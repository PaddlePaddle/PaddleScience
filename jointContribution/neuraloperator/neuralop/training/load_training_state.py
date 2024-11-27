"""
Snippet to load all artifacts of training state as Modules
without constraining to use inside a default Trainer
"""
from pathlib import Path
from typing import Union

import paddle
from paddle import nn


def load_training_state(
    save_dir: Union[str, Path],
    save_name: str,
    model: nn.Layer,
    optimizer: nn.Layer = None,
    scheduler: nn.Layer = None,
    regularizer: nn.Layer = None,
) -> dict:
    """load_training_state returns model and optional other training modules
    saved from prior training for downstream use

    Parameters
    ----------
    save_dir : Union[str, Path]
        directory from which to load training state (model, optional optimizer, scheduler, regularizer)
    save_name : str
        name of model to load
    """
    training_state = {}

    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    training_state["model"] = model.from_checkpoint(save_dir, save_name)

    # load optimizer if state exists
    if optimizer is not None:
        optimizer_pth = save_dir / "optimizer.pt"
        if optimizer_pth.exists():
            training_state["optimizer"] = optimizer.load_state_dict(
                paddle.load(optimizer_pth)
            )
        else:
            print(
                f"Warning: requested to load optimizer state, but no saved optimizer state exists in {save_dir}."
            )

    if scheduler is not None:
        scheduler_pth = save_dir / "scheduler.pt"
        if scheduler_pth.exists():
            training_state["scheduler"] = scheduler.load_state_dict(
                paddle.load(scheduler_pth)
            )
        else:
            print(
                f"Warning: requested to load scheduler state, but no saved scheduler state exists in {save_dir}."
            )

    if regularizer is not None:
        regularizer_pth = save_dir / "regularizer.pt"
        if regularizer_pth.exists():
            training_state["regularizer"] = scheduler.load_state_dict(
                paddle.load(regularizer_pth)
            )
        else:
            print(
                f"Warning: requested to load regularizer state, but no saved regularizer state exists in {save_dir}."
            )

    return training_state
