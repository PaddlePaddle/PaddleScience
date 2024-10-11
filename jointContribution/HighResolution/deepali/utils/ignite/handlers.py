from logging import Logger
from typing import Any
from typing import Callable
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Union

import paddle
from ignite.engine import Engine
from ignite.engine import Events
from ignite.engine import State
from paddle.io import DataLoader

from ...core import RE_OUTPUT_KEY_INDEX
from ..tensorboard import add_summary_images
from ..tensorboard import escape_channel_index_format_string


def clamp_learning_rate(
    engine: Engine,
    optimizer: paddle.optim.optimizer.Optimizer,
    min_learning_rate: Optional[float] = None,
    max_learning_rate: Optional[float] = None,
):
    if min_learning_rate is None:
        min_learning_rate = 1e-12
    if max_learning_rate is None:
        max_learning_rate = float("inf")
    for param_group in optimizer.param_groups:
        param_group["lr"] = max(
            min_learning_rate, min(param_group["lr"], max_learning_rate)
        )


def set_distributed_sampler_epoch(engine: Engine, epoch: Optional[int] = None) -> None:
    data = engine.state.dataloader
    if epoch is None:
        epoch = engine.state.epoch - 1
    if isinstance(data, paddle.io.DataLoader):
        set_sampler_epoch = getattr(data.sampler, "set_epoch", None)
        if callable(set_sampler_epoch):
            set_sampler_epoch(epoch)
        set_sampler_epoch = getattr(data.batch_sampler, "set_epoch", None)
        if callable(set_sampler_epoch):
            set_sampler_epoch(epoch)
        if isinstance(data.batch_sampler, paddle.io.BatchSampler):
            set_sampler_epoch = getattr(data.batch_sampler.sampler, "set_epoch", None)
            if callable(set_sampler_epoch):
                set_sampler_epoch(epoch)
        data = data.dataset
    set_epoch = getattr(data, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(epoch)


def print_metrics(
    engine: Engine,
    prefix: Optional[str] = None,
    names: Optional[Union[Mapping[str, str], Sequence[str], Set[str]]] = None,
    logger: Optional[Logger] = None,
    global_step_transform: Optional[Callable] = None,
) -> None:
    """Log evaluated performance metrics stored in ``engine.state.metrics``."""
    metrics = get_scalar_metrics(engine.state)
    if not metrics:
        return
    if names is not None:
        metrics = {name: metrics[name] for name in names if name in metrics}
        if isinstance(names, Mapping):
            metrics = {names[k]: v for k, v in metrics.items()}
    global_step = get_global_step(engine, global_step_transform)
    if prefix is None:
        prefix = ""
        if global_step_transform is None:
            prefix = "epoch={e:03d}, "
        prefix += "iter={i:06d}, "
    msg = prefix.format(e=engine.state.epoch, i=global_step)
    msg += ", ".join(
        f"{key}=" + (f"{value:.5f}" if isinstance(value, float) else f"{value}")
        for key, value in metrics.items()
    )
    print(msg) if logger is None else logger.info(msg)


def reset_iterable_dataset(engine: Engine) -> None:
    """Reset iterator over engine.state.dataloader.

    This handler also calls ``set_distributed_sampler_epoch(engine)`` such that the
    epoch number is set prior to calling ``engine.set_data(engine.state.dataloader)``,
    which invokes ``iter(engine.state.dataloader)``. This is to ensure that if the
    latter shuffles the data using the epoch number as seed, that the dataset is
    then shuffled with the new epoch number.

    """
    set_distributed_sampler_epoch(engine, engine.state.epoch)
    engine.set_data(engine.state.dataloader)


def reset_model_grads(engine: Engine, model: paddle.nn.Layer) -> None:
    """Set ``grad`` attributes of model parameters to ``None``."""
    for p in model.parameters():
        p.grad = None


def reset_optimizer_grads(
    engine: Engine, optimizer: paddle.optim.optimizer.Optimizer
) -> None:
    """Set ``grad`` attributes of optimizer parameters to ``None``."""
    for group in optimizer.param_groups:
        for p in group["params"]:
            p.grad = None


def set_engine_state(engine: Engine, **kwargs) -> None:
    """Add dictionary values as attributes to ``engine.state`` object."""
    for name, value in kwargs.items():
        setattr(engine.state, name, value)


def terminate_on_max_iteration(engine: Engine, max_iterations: int):
    """Terminate training when maximum number of global iterations reached."""
    if max_iterations > 0 and engine.state.iteration >= max_iterations:
        engine.terminate()


def write_summary_hists(
    engine: Engine,
    writer: paddle.utils.tensorboard.SummaryWriter,
    model: paddle.nn.Layer,
    prefix: Optional[str] = None,
    weights: bool = True,
    grads: bool = True,
    global_step_transform: Optional[Callable] = None,
) -> None:
    """Add histograms of model weights to TensorBoard summary."""
    global_step = get_global_step(engine, global_step_transform)
    prefix = (prefix or "").format(e=engine.state.epoch, i=engine.state.iteration)
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        name = name.replace(".", "/")
        if weights:
            writer.add_histogram(
                tag=f"{prefix}weights/{name}",
                values=p.data.detach().cpu().numpy(),
                global_step=global_step,
            )
        if grads:
            writer.add_histogram(
                tag=f"{prefix}grads/{name}",
                values=p.grad.detach().cpu().numpy(),
                global_step=global_step,
            )


def write_summary_images(
    engine: Engine,
    writer: paddle.utils.tensorboard.SummaryWriter,
    names: Optional[Union[Mapping[str, str], Sequence[str], Set[str]]] = None,
    prefix: Optional[str] = None,
    image_transform: Optional[Callable[[str, paddle.Tensor], paddle.Tensor]] = True,
    rescale_transform: Union[
        bool, Callable[[str, paddle.Tensor], paddle.Tensor]
    ] = False,
    global_step_transform: Optional[Callable[[Engine, Events], int]] = None,
    channel_offset: Union[bool, Callable[[Engine], int]] = False,
) -> None:
    """Add images stored in ``engine.state.batch`` and ``engine.state.output`` to TensorBoard summary.

    Args:
        engine: Ignite engine.
        writer: TensorBoard writer with open event file.
        names: Possibly hierarchical keys of input batch or output image entries to write.
            When a dictionary is given, replace image name by the map value.
        prefix: Prefix string for TensorBoard tags. Can be format string with
            placeholders ``{e}`` for epoch and ``{i}`` global iteration.
        image_transform: Callable used to extract a 2D image tensor of shape
            ``(C, Y, X)`` from each image. When a multi-channel tensor is returnd (C > 1),
            each channel is saved as separate image to the TensorBoard event file.
            By default, the central slice of the first image in the batch is extracted.
            The first argument is the name of the image tensor if applicable, or an empty
            string otherwise. This can be used to differently process different images.
        rescale_transform: Image intensities must be in the closed interval ``[0, 1]``.
            Set to ``False``, if this is already the case or when a custom
            ``image_transform`` is used which normalizes the image intensities.
        global_step_transform: Callable used to obtain global iteration.
            Called with arguments ``engine`` and ``Events.ITERATION_COMPLETED``.
            If not specified, use ``engine.state.iteration``.
        channel_offset: Offset to use for channel index in image tag format string.
            If ``True``, use batch index times batch size of data loader plus one. If ``False``,
            use zero. Otherwise, a callable must be given which receives the engine on which
            the event is triggered as argument.

    """

    def filter_images(
        arg: Union[Mapping[str, Any], Sequence[Any]], prefix: str = ""
    ) -> Dict[str, paddle.Tensor]:
        images = {}
        if not isinstance(arg, dict):
            arg = {str(i): value for i, value in enumerate(arg)}
        for name, value in arg.items():
            if name == "loss":
                continue
            if isinstance(value, (dict, tuple, list)):
                images.update(filter_images(value, prefix + name + "."))
            elif not isinstance(value, paddle.Tensor) or value.ndim < 4:
                continue
            else:
                name = prefix + name
                if isinstance(names, dict):
                    if name not in names:
                        continue
                    name = names[name]
                elif names is not None and name not in names:
                    continue
                images[name] = value
        return images

    def format_tag(arg: str) -> str:
        arg = escape_channel_index_format_string(arg)
        arg = arg.format(e=engine.state.epoch, i=engine.state.iteration, b=batch_index)
        return arg

    batch_index = (engine.state.iteration - 1) % engine.state.epoch_length + 1
    global_step = get_global_step(engine, global_step_transform)
    channel_offset_value = 0
    if channel_offset is True:
        dataloader: DataLoader = engine.state.dataloader
        if not dataloader.batch_size:
            raise RuntimeError(
                "write_summary_images() 'channel_offset' is True, but 'engine.state.dataloader.batch_size' is None"
            )
        channel_offset_value = (batch_index - 1) * dataloader.batch_size + 1
    elif callable(channel_offset):
        channel_offset_value = channel_offset(engine)
    elif channel_offset is not False:
        raise TypeError(
            "write_summary_images() 'channel_offset' must be bool or callable"
        )
    prefix = format_tag(prefix or "")
    if isinstance(names, dict):
        names = {
            RE_OUTPUT_KEY_INDEX.sub(".\\1", name): tag for name, tag in names.items()
        }
        names = {name: format_tag(tag) for name, tag in names.items()}
    elif names is not None:
        names = {RE_OUTPUT_KEY_INDEX.sub(".\\1", name) for name in names}
        names = {format_tag(name) for name in names}
    batch = engine.state.batch
    if isinstance(batch, paddle.Tensor):
        batch = {"batch": batch}
    batch = filter_images(batch)
    add_summary_images(
        writer,
        prefix,
        batch,
        global_step=global_step,
        image_transform=image_transform,
        rescale_transform=rescale_transform,
        channel_offset=channel_offset_value,
    )
    output = engine.state.output
    if isinstance(output, paddle.Tensor):
        output = {"output": output}
    output = filter_images(output)
    add_summary_images(
        writer,
        prefix,
        output,
        global_step=global_step,
        image_transform=image_transform,
        rescale_transform=rescale_transform,
        channel_offset=channel_offset_value,
    )
    writer.flush()


def write_summary_metrics(
    engine: Engine,
    writer: paddle.utils.tensorboard.SummaryWriter,
    prefix: Optional[str] = None,
    global_step_transform: Optional[Callable] = None,
):
    """Add computed values stored in ``engine.state.metrics`` to Tensorboard summary."""
    global_step = get_global_step(engine, global_step_transform)
    prefix = (prefix or "").format(e=engine.state.epoch, i=engine.state.iteration)
    metrics = get_scalar_metrics(engine.state)
    for name, value in metrics.items():
        writer.add_scalar(prefix + name, value, global_step=global_step)
    writer.flush()


def write_summary_optim_params(
    engine: Engine,
    writer: paddle.utils.tensorboard.SummaryWriter,
    optimizer: paddle.optim.optimizer.Optimizer,
    params: Optional[Union[str, Sequence[str]]] = None,
    prefix: Optional[str] = None,
    global_step_transform: Optional[Callable] = None,
) -> None:
    """Add optimization parameters to TensorBoard summary."""
    global_step = get_global_step(engine, global_step_transform)
    prefix = (prefix or "").format(e=engine.state.epoch, i=engine.state.iteration)
    if isinstance(params, str):
        params = [params]
    for group_id, param_group in enumerate(optimizer.param_groups):
        for param_name in params or param_group.keys():
            try:
                param_value = float(param_group[param_name])
            except (KeyError, TypeError):
                continue
            tag = prefix + param_name
            if len(optimizer.param_groups) > 1:
                tag += f"/group_{group_id}"
            writer.add_scalar(tag, param_value, global_step)


def get_global_step(
    engine: Engine, global_step_transform: Optional[Callable] = None
) -> int:
    """Get global step for summary event."""
    if global_step_transform is None:
        return engine.state.iteration
    global_step = global_step_transform(engine, Events.ITERATION_COMPLETED)
    if not isinstance(global_step, int):
        raise TypeError(
            f"'global_step_transform' must return an int, got {type(global_step)}"
        )
    return global_step


def get_scalar_metrics(state: State) -> Dict[str, Union[float, int]]:
    """Get scalar metrics stored in ``state.metrics``."""
    metrics = {}
    for key, value in state.metrics.items():
        if isinstance(value, paddle.Tensor):
            if value.squeeze().dim() != 0:
                continue
            value = value.item()
            value = int(value) if isinstance(value, int) else float(value)
        if isinstance(value, (float, int)):
            metrics[key] = value
    return metrics
