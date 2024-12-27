from pathlib import Path
from timeit import default_timer as timer
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Union

import paddle
from deepali.core import Axes
from deepali.core import Device
from deepali.core import Grid
from deepali.core import PathStr
from deepali.core import functional as U
from deepali.core.config import join_kwargs_in_sequence
from deepali.data import FlowField
from deepali.data import Image
from deepali.losses import RegistrationResult
from deepali.losses import new_loss
from deepali.spatial import DisplacementFieldTransform
from deepali.spatial import HomogeneousTransform
from deepali.spatial import NonRigidTransform
from deepali.spatial import QuaternionRotation
from deepali.spatial import RigidQuaternionTransform
from deepali.spatial import SequentialTransform
from deepali.spatial import SpatialTransform
from deepali.spatial import Translation
from deepali.spatial import new_spatial_transform

from .engine import RegistrationEngine
from .hooks import RegistrationEvalHook
from .hooks import RegistrationStepHook
from .hooks import normalize_grad_hook
from .hooks import smooth_grad_hook
from .losses import PairwiseImageRegistrationLoss
from .losses import weight_channel_names
from .optim import new_optimizer


def register_pairwise(
    target: Union[PathStr, Dict[str, PathStr]],
    source: Union[PathStr, Dict[str, PathStr]],
    config: Optional[Dict[str, Any]] = None,
    outdir: Optional[PathStr] = None,
    verbose: Union[bool, int] = False,
    debug: Union[bool, int] = False,
    device: Optional[Device] = None,
) -> SpatialTransform:
    """Register pair of images."""
    if config is None:
        config = {}
    if outdir is not None:
        outdir = Path(outdir).absolute()
        outdir.mkdir(parents=True, exist_ok=True)
    loss_config, loss_weights = get_loss_config(config)
    model_name, model_args, model_init = get_model_config(config)
    optim_name, optim_args, optim_loop = get_optim_config(config)
    levels, coarsest_level, finest_level = get_levels_config(config)
    finest_spacing, min_size, pyramid_dims = get_pyramid_config(config)
    device = get_device_config(config, device)
    verbose = int(verbose)
    debug = int(debug)
    if verbose > 0:
        print()
    start = timer()
    target_keys = set(loss_config.keys()) | set(
        weight_channel_names(loss_weights).values()
    )
    target_image, target_chns = read_images(target, names=target_keys, device=device)
    source_image, source_chns = read_images(
        source, names=loss_config.keys(), device=device
    )
    if verbose > 3:
        print(f"Read images from files in {timer() - start:.3f}s")
    start_reg = timer()
    target_image = append_mask(target_image, target_chns, config)
    source_image = append_mask(source_image, source_chns, config)
    norm_params = get_normalize_config(config, target_image, target_chns)
    target_image = normalize_data_(target_image, target_chns, **norm_params)
    source_image = normalize_data_(source_image, source_chns, **norm_params)
    start = timer()
    target_pyramid = target_image.pyramid(
        levels,
        start=finest_level,
        end=coarsest_level,
        dims=pyramid_dims,
        spacing=finest_spacing,
        min_size=min_size,
    )
    source_pyramid = source_image.pyramid(
        levels,
        start=finest_level,
        end=coarsest_level,
        dims=pyramid_dims,
        spacing=finest_spacing,
        min_size=min_size,
    )
    if verbose > 3:
        print(f"Constructed Gaussian resolution pyramids in {timer() - start:.3f}s\n")
    if verbose > 2:
        print("Target image pyramid:")
        print_pyramid_info(target_pyramid)
        print("Source image pyramid:")
        print_pyramid_info(source_pyramid)
    del target_image
    del source_image
    source_grid = source_pyramid[finest_level].grid()
    finest_grid = target_pyramid[finest_level].grid()
    coarsest_grid = target_pyramid[coarsest_level].grid()
    post_transform = get_post_transform(config, finest_grid, source_grid)
    transform_downsample = model_args.pop("downsample", 0)
    transform_grid = coarsest_grid.downsample(transform_downsample)
    # here is ok
    transform = new_spatial_transform(
        model_name, grid=transform_grid, groups=1, **model_args
    )
    if model_init:
        if verbose > 1:
            print(f"Fitting '{model_init}'...")
        disp_field = FlowField.read(model_init).to(device=device)
        assert isinstance(disp_field, FlowField)
        start = timer()
        transform = transform.to(device=device).fit(disp_field.batch())
        if verbose > 0:
            print(f"Fitted initial displacement field in {timer() - start:.3f}s")
        del disp_field
    grid_transform = SequentialTransform(transform, post_transform)
    grid_transform = grid_transform.to(device=device)
    for level in range(coarsest_level, finest_level - 1, -1):
        target_image = target_pyramid[level]
        source_image = source_pyramid[level]
        # here is ok
        if outdir and debug > 0:
            write_channels(
                data=target_image.tensor(),
                grid=target_image.grid(),
                channels=target_chns,
                outdir=outdir,
                prefix=f"level_{level}_target_",
            )
            write_channels(
                data=source_image.tensor(),
                grid=source_image.grid(),
                channels=source_chns,
                outdir=outdir,
                prefix=f"level_{level}_source_",
            )
        if level != coarsest_level:
            start = timer()
            transform_grid = target_image.grid().downsample(transform_downsample)
            transform.grid_(transform_grid)
            if verbose > 3:
                print(f"Subdivided control point grid in {timer() - start:.3f}s")
        grid_transform.grid_(target_image.grid())
        loss_terms = new_loss_terms(loss_config)
        loss = PairwiseImageRegistrationLoss(
            losses=loss_terms,
            source_data=source_image.tensor().unsqueeze(axis=0),
            target_data=target_image.tensor().unsqueeze(axis=0),
            source_grid=source_image.grid(),
            target_grid=target_image.grid(),
            source_chns=source_chns,
            target_chns=target_chns,
            transform=grid_transform,
            weights=loss_weights,
        )
        loss = loss.to(device=device)
        if outdir and debug > 1:
            start = timer()
            result = loss.eval()
            if verbose > 3:
                print(f"Evaluated initial loss in {timer() - start:.3f}s")
            write_result(
                result,
                grid=target_image.grid(),
                channels=source_chns,
                outdir=outdir,
                prefix=f"level_{level}_initial_",
            )
            flow = grid_transform.flow(target_image.grid(), device=device)
            flow[0].write(outdir / f"level_{level}_initial_def.mha")
        optimizer = new_optimizer(optim_name, model=grid_transform, **optim_args)
        engine = RegistrationEngine(
            model=grid_transform,
            loss=loss,
            optimizer=optimizer,
            max_steps=optim_loop.get("max_steps", 250),
            min_delta=float(optim_loop.get("min_delta", "nan")),
        )
        grad_sigma = float(optim_loop.get("smooth_grad", 0))
        if isinstance(transform, NonRigidTransform) and grad_sigma > 0:
            engine.register_eval_hook(smooth_grad_hook(transform, sigma=grad_sigma))
        engine.register_eval_hook(normalize_grad_hook(transform))
        if verbose > 2:
            engine.register_eval_hook(print_eval_loss_hook(level))
        elif verbose > 1:
            engine.register_step_hook(print_step_loss_hook(level))
        if outdir and debug > 2:
            engine.register_eval_hook(
                write_result_hook(
                    level=level,
                    grid=target_image.grid(),
                    channels=source_chns,
                    outdir=outdir,
                )
            )
        engine.run()
        if verbose > 0 or outdir and debug > 0:
            start = timer()
            result = loss.eval()
            if verbose > 3:
                print(f"Evaluated final loss in {timer() - start:.3f}s")
            if verbose > 0:
                loss_value = float(result["loss"])
                print(
                    f"level={level:d}: loss={loss_value:.5f} ({engine.num_steps:d} steps)",
                    flush=True,
                )
            if outdir and debug > 0:
                write_result(
                    result,
                    grid=target_image.grid(),
                    channels=source_chns,
                    outdir=outdir,
                    prefix=f"level_{level}_final_",
                )
                flow = grid_transform.flow(device=device)
                flow[0].write(outdir / f"level_{level}_final_def.mha")
    if verbose > 3:
        print(f"Registered images in {timer() - start_reg:.3f}s")
    if verbose > 0:
        print()
    return grid_transform


def append_mask(
    image: Image, channels: Dict[str, Tuple[int, int]], config: Dict[str, Any]
) -> Image:
    """Append foreground mask to data tensor."""
    data = image.tensor()
    if "img" in channels:
        lower_threshold, upper_threshold = get_clamp_config(config, "img")
        mask = U.threshold(
            data[slice(*channels["img"])], lower_threshold, upper_threshold
        )
    else:
        mask = paddle.ones(shape=(1,) + tuple(data.shape)[1:], dtype=data.dtype)
    data = paddle.concat(x=[data, mask.astype(data.dtype)], axis=0)
    channels["msk"] = tuple(data.shape)[0] - 1, tuple(data.shape)[0]
    return Image(data, image.grid())


def append_data(
    data: Optional[paddle.Tensor],
    channels: Dict[str, Tuple[int, int]],
    name: str,
    other: paddle.Tensor,
) -> paddle.Tensor:
    """Append image data."""
    if data is None:
        data = other
    else:
        data = paddle.concat(x=[data, other], axis=0)
    channels[name] = tuple(data.shape)[0] - tuple(other.shape)[0], tuple(data.shape)[0]
    return data


def read_images(
    sample: Union[PathStr, Dict[str, PathStr]], names: Set[str], device: str
) -> Tuple[Image, Dict[str, Tuple[int, int]]]:
    """Read image data from input files."""
    data = None
    grid = None
    if isinstance(sample, (Path, str)):
        sample = {"img": sample}
    img_path = sample.get("img")
    seg_path = sample.get("seg")
    sdf_path = sample.get("sdf")
    for path in (img_path, seg_path, sdf_path):
        if not path:
            continue
        grid = Grid.from_file(path).align_corners_(True)
        break
    else:
        raise ValueError(
            "One of 'img', 'seg', or 'sdf' input image file paths is required"
        )
    assert grid is not None
    dtype = "float32"
    channels = {}
    if "img" in names:
        temp = Image.read(img_path, dtype=dtype, device=device)
        data = append_data(data, channels, "img", temp.tensor())
    if "seg" in names:
        if seg_path is None:
            raise ValueError("Missing segmentation label image file path")
        temp = Image.read(seg_path, dtype="int64", device=device)
        temp_grid = temp.grid()
        num_classes = int(temp.max()) + 1
        temp = temp.tensor().unsqueeze(axis=0)
        temp = U.as_one_hot_tensor(temp, num_classes).to(dtype=dtype)
        temp = temp.squeeze(axis=0)
        temp = Image(temp, grid=temp_grid).sample(grid)
        data = append_data(data, channels, "seg", temp.tensor())
    if "sdf" in names:
        if sdf_path is None:
            raise ValueError(
                "Missing segmentation boundary signed distance field file path"
            )
        temp = Image.read(sdf_path, dtype=dtype, device=device)
        temp = temp.sample(shape=grid)
        data = append_data(data, channels, "sdf", temp.tensor())
    if data is None:
        if img_path is None:
            raise ValueError("Missing intensity image file path")
        data = Image.read(img_path, dtype=dtype, device=device)
        channels = {"img": (0, 1)}
    image = Image(data, grid=grid)
    return image, channels


def get_device_config(
    config: Dict[str, Any], device: Optional[Union[str, str]] = None
) -> str:
    """Get configured PyTorch device."""
    if device is None:
        device = config.get("device", "cpu")
    if isinstance(device, int):
        device = f"cuda:{device}"
    elif device == "cuda":
        device = "cuda:0"
    return str(device).replace("cuda", "gpu")


def load_transform(path: PathStr, grid: Grid) -> SpatialTransform:
    """Load transformation from file.

    Args:
        path: File path from which to load spatial transformation.
        grid: Target domain grid with respect to which transformation is defined.

    Returns:
        Loaded spatial transformation.

    """
    target_grid = grid

    def convert_matrix(
        matrix: paddle.Tensor, grid: Optional[Grid] = None
    ) -> paddle.Tensor:
        if grid is None:
            pre = target_grid.transform(Axes.CUBE_CORNERS, Axes.WORLD)
            post = target_grid.transform(Axes.WORLD, Axes.CUBE_CORNERS)
            matrix = U.homogeneous_matmul(post, matrix, pre)
        elif grid != target_grid:
            pre = target_grid.transform(Axes.CUBE_CORNERS, grid=grid)
            post = grid.transform(Axes.CUBE_CORNERS, grid=target_grid)
            matrix = U.homogeneous_matmul(post, matrix, pre)
        return matrix

    path = Path(path)
    if path.suffix == ".pt":
        value = paddle.load(path=path)
        if isinstance(value, dict):
            matrix = value.get("matrix")
            if matrix is None:
                raise KeyError(
                    "load_transform() .pt file dict must contain key 'matrix'"
                )
            grid = value.get("grid")
        elif isinstance(value, paddle.Tensor):
            matrix = value
            grid = None
        else:
            raise RuntimeError("load_transform() .pt file must contain tensor or dict")
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(axis=0)
        if matrix.ndim != 3 or tuple(matrix.shape)[1:] != (3, 4):
            raise RuntimeError(
                "load_transform() .pt file tensor must have shape (N, 3, 4)"
            )
        params = convert_matrix(matrix, grid)
        return HomogeneousTransform(target_grid, params=params)
    flow = FlowField.read(path, axes=Axes.WORLD)
    flow = flow.axes(Axes.from_grid(target_grid))
    flow = flow.sample(shape=target_grid)
    return DisplacementFieldTransform(
        target_grid, params=flow.tensor().unsqueeze(axis=0)
    )


def get_post_transform(
    config: Dict[str, Any], target_grid: Grid, source_grid: Grid
) -> Optional[SpatialTransform]:
    """Get constant rigid transformation between image grid domains."""
    align = config.get("align", False)
    if align is False or align is None:
        return None
    if isinstance(align, (Path, str)):
        return load_transform(align, target_grid)
    if align is True:
        align_centers = True
        align_directions = True
    elif isinstance(align, dict):
        align_centers = bool(align.get("centers", True))
        align_directions = bool(align.get("directions", True))
    else:
        raise ValueError(
            "get_post_transform() 'config' has invalid 'align' value: {align}"
        )
    center_offset = (
        target_grid.world_to_cube(source_grid.center()).unsqueeze(axis=0)
        if align_centers
        else None
    )
    rotation_matrix = (
        source_grid.direction() @ target_grid.direction().t().unsqueeze(axis=0)
        if align_directions
        else None
    )
    transform = None
    if center_offset is not None and rotation_matrix is not None:
        transform = RigidQuaternionTransform(
            target_grid, translation=center_offset, rotation=False
        )
        transform.rotation.matrix_(rotation_matrix)
    elif center_offset is not None:
        transform = Translation(target_grid, params=center_offset)
    elif rotation_matrix is not None:
        transform = QuaternionRotation(target_grid, params=False)
        transform.matrix_(rotation_matrix)
    return transform


def get_clamp_config(
    config: Dict[str, Any], channel: str
) -> Tuple[Optional[float], Optional[float]]:
    """Get thresholds for named image channel.

    Args:
        config: Configuration.
        channel: Name of image channel.

    Returns:
        lower_threshold: Lower threshold.
        upper_threshold: Upper threshold.

    """
    input_config = config.get("input", {})
    if not isinstance(input_config, dict):
        raise ValueError("get_clamp_config() 'input' value must be dict")
    channel_config = input_config.get(channel)
    if not isinstance(channel_config, dict):
        channel_config = {"clamp": channel_config}
    thresholds = channel_config.get("clamp", input_config.get("clamp"))
    if thresholds is None:
        thresholds = None, None
    elif isinstance(thresholds, (int, float)):
        thresholds = float(thresholds), None
    if not isinstance(thresholds, (list, tuple)):
        raise ValueError("get_clamp_config() value must be scalar or sequence")
    if len(thresholds) != 2:
        raise ValueError("get_clamp_config() value must be scalar or [min, max]")
    thresholds = tuple(None if v is None else float(v) for v in thresholds)
    lower_threshold, upper_threshold = thresholds
    return lower_threshold, upper_threshold


def get_scale_config(config: Dict[str, Any], channel: str) -> Optional[float]:
    """Get channel scaling factor."""
    input_config = config.get("input", {})
    if not isinstance(input_config, dict):
        return None
    channel_config = input_config.get(channel)
    if not isinstance(channel_config, dict):
        return None
    value = channel_config.get("scale", input_config.get("scale"))
    if value is None:
        return None
    return float(value)


def get_normalize_config(
    config: Dict[str, Any], image: Image, channels: Dict[str, Tuple[int, int]]
) -> Dict[str, Dict[str, paddle.Tensor]]:
    """Calculate data normalization parameters.

    Args:
        config: Configuration.
        image: Image data.
        channels: Map of image channel slices.

    Returns:
        Dictionary of normalization parameters.

    """
    scale = {}
    shift = {}
    for channel, (start, stop) in channels.items():
        start_1 = image.tensor().shape[0] + start if start < 0 else start
        data = paddle.slice(image.tensor(), [0], [start_1], [start_1 + (stop - start)])
        lower_threshold, upper_threshold = get_clamp_config(config, channel)
        scale_factor = get_scale_config(config, channel)
        if channel in ("msk", "seg"):
            if lower_threshold is None:
                lower_threshold = 0
            if upper_threshold is None:
                upper_threshold = 1
        else:
            if lower_threshold is None:
                lower_threshold = data.min()
            if upper_threshold is None:
                upper_threshold = data.max()
        if scale_factor is None:
            if upper_threshold > lower_threshold:
                scale_factor = upper_threshold - lower_threshold
            else:
                scale_factor = 1
        else:
            scale_factor = 1 / scale_factor
        shift[channel] = lower_threshold
        scale[channel] = scale_factor
    return dict(shift=shift, scale=scale)


def normalize_data_(
    image: Image,
    channels: Dict[str, Tuple[int, int]],
    shift: Optional[Dict[str, paddle.Tensor]] = None,
    scale: Optional[Dict[str, paddle.Tensor]] = None,
) -> Image:
    """Normalize image data."""
    if shift is None:
        shift = {}
    if scale is None:
        scale = {}
    for channel, (start, stop) in channels.items():
        start_2 = image.tensor().shape[0] + start if start < 0 else start
        data = paddle.slice(image.tensor(), [0], [start_2], [start_2 + (stop - start)])
        offset = shift.get(channel)
        if offset is not None:
            data -= offset
        norm = scale.get(channel)
        if norm is not None:
            data /= norm
        if channel in ("msk", "seg"):
            data.clip_(min=0, max=1)
    return image


def get_levels_config(config: Dict[str, Any]) -> Tuple[int, int, int]:
    """Get indices of coarsest and finest level from configuration."""
    cfg = config.get("pyramid", {})
    levels = cfg.get("levels", 4)
    if isinstance(levels, int):
        levels = levels - 1, 0
    if not isinstance(levels, (list, tuple)):
        raise TypeError(
            "register_pairwise() 'config' key 'pyramid.levels': value must be int, tuple, or list"
        )
    coarsest_level, finest_level = levels
    if finest_level > coarsest_level:
        raise ValueError(
            "register_pairwise() 'config' key 'pyramid.levels':"
            + " finest level must be less or equal than coarsest level"
        )
    levels = coarsest_level + 1
    if "max_level" in cfg:
        levels = max(levels, cfg["max_level"])
    return levels, coarsest_level, finest_level


def get_pyramid_config(
    config: Dict[str, Any]
) -> Tuple[Optional[Union[float, Sequence[float]]], int, Optional[Union[str, int]]]:
    """Get settings of Gaussian resolution pyramid from configuration."""
    cfg = config.get("pyramid", {})
    min_size = cfg.get("min_size", 16)
    finest_spacing = cfg.get("spacing")
    dims = cfg.get("dims")
    return finest_spacing, min_size, dims


def get_loss_config(
    config: Dict[str, Any]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    """Instantiate terms of registration loss given configuration object.

    Args:
        config: Configuration.

    Returns:
        losses: Preparsed configuration of loss terms (cf. ``new_loss_terms()``).
        weights: Weights of loss terms.

    """
    cfg = None
    losses = {}
    weights = {}
    sections = "loss", "losses", "energy"
    for name in sections:
        if name in config:
            if cfg is not None:
                raise ValueError(
                    "get_loss_config() keys {sections} are mutually exclusive"
                )
            cfg = config[name]
    if not cfg:
        cfg = "SSD"
    if isinstance(cfg, str):
        cfg = (cfg,)
    if isinstance(cfg, Sequence):
        names, cfg = cfg, {}
        for i, name in enumerate(names):
            cfg[f"loss_{i}"] = str(name)
    if isinstance(cfg, dict):
        for key, value in cfg.items():
            name = None
            weight = 1
            kwargs = {}
            if isinstance(value, str):
                name = value
            elif isinstance(value, Sequence):
                if not value:
                    raise ValueError(f"get_loss_config() '{key}' loss entry is empty")
                if len(value) == 1:
                    if isinstance(value[0], str):
                        value = {"name": value[0]}
                elif len(value) > 1:
                    if isinstance(value[0], (int, float)):
                        value[0] = {"weight": value[0]}
                    if isinstance(value[1], str):
                        value[1] = {"name": value[1]}
                value = join_kwargs_in_sequence(value)
                if isinstance(value, dict):
                    kwargs = dict(value)
                    name = kwargs.pop("name", None)
                    weight = kwargs.pop("weight", 1)
                elif len(value) == 2:
                    name = value[0]
                    kwargs = dict(value[1])
                elif len(value) == 3:
                    weight = float(value[0])
                    name = value[1]
                    kwargs = dict(value[2])
                else:
                    raise ValueError(
                        f"get_loss_config() '{key}' invalid loss configuration"
                    )
            elif isinstance(value, dict):
                kwargs = dict(value)
                name = kwargs.pop("name", None)
                weight = kwargs.pop("weight", 1)
            else:
                weight, name = value
            if name is None:
                raise ValueError(f"get_loss_config() missing 'name' for loss '{key}'")
            if not isinstance(name, str):
                raise TypeError(f"get_loss_config() 'name' of loss '{key}' must be str")
            kwargs["name"] = name
            losses[key] = kwargs
            weights[key] = weight
    else:
        raise TypeError(
            "get_loss_config() 'config' \"losses\" must be str, tuple, list, or dict"
        )
    weights_config = config.get("weights", {})
    if isinstance(weights_config, (int, float)):
        weights_config = (weights_config,)
    if isinstance(weights_config, (list, tuple)):
        names, weights_config = weights_config, {}
        for i, weight in enumerate(names):
            weights_config[f"loss_{i}"] = weight
    if not isinstance(weights_config, dict):
        raise TypeError(
            "get_loss_config() 'weights' must be scalar, tuple, list, or dict"
        )
    weights.update(weights_config)
    losses = {k: v for k, v in losses.items() if weights.get(k, 0)}
    weights = {k: v for k, v in weights.items() if k in losses}
    return losses, weights


def new_loss_terms(config: Dict[str, Any]) -> Dict[str, paddle.nn.Layer]:
    """Instantiate terms of registration loss.

    Args:
        config: Preparsed configuration of loss terms.
        target_tree: Target vessel centerline tree.

    Returns:
        Mapping from channel or loss name to loss module instance.

    """
    losses = {}
    for key, value in config.items():
        kwargs = dict(value)
        name = kwargs.pop("name", None)
        _ = kwargs.pop("weight", None)
        if name is None:
            raise ValueError(f"new_loss_terms() missing 'name' for loss '{key}'")
        if not isinstance(name, str):
            raise TypeError(f"new_loss_terms() 'name' of loss '{key}' must be str")
        loss = new_loss(name, **kwargs)
        losses[key] = loss
    return losses


def get_model_config(
    config: Dict[str, Any]
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """Get configuration of transformation model to use."""
    cfg = config.get("model", {})
    cfg = dict(name=cfg) if isinstance(cfg, str) else dict(cfg)
    model_name = cfg.pop("name")
    assert isinstance(model_name, str)
    assert model_name != ""
    model_init = cfg.pop("init", None)
    if model_init is not None:
        model_init = Path(model_init).as_posix()
    model_args = dict(cfg.get(model_name, cfg))
    return model_name, model_args, model_init


def get_optim_config(
    config: Dict[str, Any]
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Get configuration of optimizer to use."""
    cfg = config.get("optim", {})
    cfg = dict(name=cfg) if isinstance(cfg, str) else dict(cfg)
    if "optimizer" in cfg:
        if "name" in cfg:
            raise ValueError(
                "get_optim_config() keys ('name', 'optimizer') are mutually exclusive"
            )
        cfg["name"] = cfg.pop("optimizer")
    optim_name = str(cfg.pop("name", "LBFGS"))
    optim_loop = {}
    for key in ("max_steps", "min_delta", "smooth_grad"):
        if key in cfg:
            optim_loop[key] = cfg.pop(key)
    optim_args = {k: v for k, v in cfg.items() if isinstance(k, str) and k[0].islower()}
    optim_args.update(cfg.get(optim_name, {}))
    lr_keys = "step_size", "learning_rate"
    for lr_key in ("step_size", "learning_rate"):
        if lr_key in optim_args:
            if "lr" in optim_args:
                raise ValueError(
                    f"get_optim_config() keys {lr_keys + ('lr',)} are mutually exclusive"
                )
            optim_args["lr"] = optim_args.pop(lr_key)
    return optim_name, optim_args, optim_loop


@paddle.no_grad()
def write_channels(
    data: paddle.Tensor,
    grid: Grid,
    channels: Mapping[str, Tuple[int, int]],
    outdir: PathStr,
    prefix: str = "",
) -> None:
    """Write image channels."""
    for name, (start, stop) in channels.items():
        image = data[slice(start, stop, 1)]
        if name == "seg":
            image = image.argmax(axis=0, keepdim=True).astype("uint8")
        elif name == "msk":
            image = image.mul(255).clip_(min=0, max=255).astype("uint8")
        if not isinstance(image, Image):
            image = Image(image, grid=grid)
        image.write(outdir / f"{prefix}{name}.mha")


@paddle.no_grad()
def write_result(
    result: RegistrationResult,
    grid: Grid,
    channels: Mapping[str, Tuple[int, int]],
    outdir: PathStr,
    prefix: str = "",
) -> None:
    """Write registration result to output directory."""
    data = result["source"]
    assert isinstance(data, paddle.Tensor)
    write_channels(data[0], grid=grid, channels=channels, outdir=outdir, prefix=prefix)
    data = result["mask"]
    assert isinstance(data, paddle.Tensor)
    if data.dtype == "bool":
        data = data.astype("uint8").multiply_(y=paddle.to_tensor(255))
    mask = Image(data[0], grid=grid)
    mask.write(outdir / f"{prefix}olm.mha")


def write_result_hook(
    level: int, grid: Grid, channels: Mapping[str, Tuple[int, int]], outdir: Path
) -> RegistrationEvalHook:
    """Get callback function for writing results after each evaluation."""

    def fn(
        _: RegistrationEngine,
        num_steps: int,
        num_evals: int,
        result: RegistrationResult,
    ) -> None:
        prefix = f"level_{level}_step_{num_steps:03d}_eval_{num_evals}_"
        write_result(result, grid=grid, channels=channels, outdir=outdir, prefix=prefix)

    return fn


def print_eval_loss_hook(level: int) -> RegistrationEvalHook:
    """Get callback function for printing loss after each evaluation."""

    def fn(
        _: RegistrationEngine, num_steps: int, num_eval: int, result: RegistrationResult
    ) -> None:
        loss = float(result["loss"])
        message = f"  {num_steps:>4d}:"
        message += f" {loss:>12.05f} (loss)"
        weights: Dict[str, Union[str, float]] = result.get("weights", {})
        losses: Dict[str, paddle.Tensor] = result["losses"]
        for name, value in losses.items():
            value = float(value)
            weight = weights.get(name, 1.0)
            if not isinstance(weight, str):
                value *= weight
            elif "+" in weight:
                weight = f"({weight})"
            message += f", {value:>12.05f} [{weight} * {name}]"
        if num_eval > 1:
            message += " [evals={num_eval:d}]"
        print(message, flush=True)

    return fn


def print_step_loss_hook(level: int) -> RegistrationStepHook:
    """Get callback function for printing loss after each step."""

    def fn(_: RegistrationEngine, num_steps: int, num_eval: int, loss: float) -> None:
        message = f"  {num_steps:>4d}: {loss:>12.05f}"
        if num_eval > 1:
            message += " [evals={num_eval:d}]"
        print(message, flush=True)

    return fn


def print_pyramid_info(pyramid: Dict[str, Image]) -> None:
    """Print information of image resolution pyramid."""
    levels = sorted(pyramid.keys())
    for level in reversed(levels):
        grid = pyramid[level].grid()
        size = ", ".join([f"{n:>3d}" for n in tuple(grid.shape)])
        origin = ", ".join([f"{n:.2f}" for n in grid.origin()])
        extent = ", ".join([f"{n:.2f}" for n in grid.extent()])
        domain = ", ".join([f"{n:.2f}" for n in grid.cube_extent()])
        print(
            f"- Level {level}:"
            + f" size=({size})"
            + f", origin=({origin})"
            + f", extent=({extent})"
            + f", domain=({domain})"
        )
    print()
