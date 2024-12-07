import json
import logging
import sys
from pathlib import Path
from timeit import default_timer as timer
from typing import Any
from typing import Dict

import deepali
import paddle
import yaml
from deepali.core import Grid
from deepali.core import PathStr

# from deepali.utils.cli import filter_warning_of_experimental_named_tensors_feature
from deepali.core.argparse import Args
from deepali.core.argparse import ArgumentParser
from deepali.core.argparse import main_func
from deepali.core.environ import cuda_visible_devices
from deepali.core.logging import configure_logging
from deepali.core.pathlib import unlink_or_mkdir
from deepali.data import Image
from deepali.modules import TransformImage
from paddle import Tensor

from .pairwise import register_pairwise

log = logging.getLogger()


def parser(**kwargs) -> ArgumentParser:
    """Construct argument parser."""
    if "description" not in kwargs:
        kwargs["description"] = globals()["__doc__"]
    parser = ArgumentParser(**kwargs)
    parser.add_argument(
        "-c",
        "--config",
        help="Configuration file",
        default=Path(__file__).parent / "params_seg.yaml",
    )
    parser.add_argument(
        "-t", "--target", "--target-img", dest="target_img", help="Fixed target image"
    )
    parser.add_argument(
        "-s", "--source", "--source-img", dest="source_img", help="Moving source image"
    )
    parser.add_argument("--target-seg", help="Fixed target segmentation label image")
    parser.add_argument("--source-seg", help="Moving source segmentation label image")
    parser.add_argument(
        "-o",
        "--output",
        "--output-transform",
        dest="output_transform",
        help="Output transformation parameters",
    )
    parser.add_argument(
        "-w",
        "--warped",
        "--warped-img",
        "--output-img",
        dest="warped_img",
        help="Deformed source image",
    )
    parser.add_argument(
        "--warped-seg",
        "--output-seg",
        dest="warped_seg",
        help="Deformed source segmentation label image",
    )
    parser.add_argument(
        "--device",
        help="Device on which to execute registration",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument("--debug-dir", help="Output directory for intermediate files")
    parser.add_argument(
        "--debug", "--debug-level", help="Debug level", type=int, default=0
    )
    parser.add_argument(
        "-v", "--verbose", help="Verbosity of output messages", type=int, default=0
    )
    parser.add_argument(
        "--log-level",
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser


def init(args: Args) -> int:
    """Initialize registration."""
    configure_logging(log, args)
    if args.device == "cuda":
        if not paddle.device.cuda.device_count() >= 1:
            log.error("Cannot use --device 'cuda' ")
            return 1
        gpu_ids = cuda_visible_devices()
        if len(gpu_ids) != 1:
            log.error("CUDA_VISIBLE_DEVICES must be set to one GPU")
            return 1
    # filter_warning_of_experimental_named_tensors_feature()
    return 0


def register_func(args: Args) -> deepali.spatial.SpatialTransform:
    """Execute registration given parsed arguments."""
    config = load_config_para(args.config)
    device = str("cuda:0" if args.device == "cuda" else "cpu").replace("cuda", "gpu")
    start = timer()
    transform = register_pairwise(
        target={"img": args.target_img, "seg": args.target_seg},
        source={"img": args.source_img, "seg": args.source_seg},
        config=config,
        outdir=args.debug_dir,
        device=args.device,
        verbose=args.verbose,
        debug=args.debug,
    )
    log.info(f"Elapsed time: {timer() - start:.3f}s")
    if args.warped_img:
        target_grid = Grid.from_file(args.target_img)
        source_image = Image.read(args.source_img, device=device)
        warp_image = TransformImage(
            target=target_grid,
            source=source_image.grid(),
            sampling="linear",
            padding=source_image.min(),
        ).to(device)
        # here is ok
        data: Tensor = warp_image(transform.tensor(), source_image)
        warped_image = Image(data, target_grid)
        warped_image.write(unlink_or_mkdir(args.warped_img))
    if args.warped_seg:
        target_grid = Grid.from_file(args.target_seg)
        source_image = Image.read(args.source_seg, device=device)
        warp_labels = TransformImage(
            target=target_grid,
            source=source_image.grid(),
            sampling="nearest",
            padding=0,
        ).to(device)
        data: Tensor = warp_labels(transform.tensor(), source_image)
        warped_image = Image(data, target_grid)
        warped_image.write(unlink_or_mkdir(args.warped_seg))
    if args.output_transform:
        path = unlink_or_mkdir(args.output_transform)
        if path.suffix == ".pt":
            transform.clear_buffers()
            paddle.save(obj=transform, path=path)
        else:
            transform.flow()[0].write(path)
    return transform


main = main_func(parser, register_func, init=init)


def load_config_para(path: PathStr) -> Dict[str, Any]:
    """Load registration parameters from configuration file."""
    config_path = Path(path).absolute()
    log.info(f"Load configuration from {config_path}")
    config_text = config_path.read_text()
    if config_path.suffix == ".json":
        return json.loads(config_text)
    return yaml.safe_load(config_text)


if __name__ == "__main__":
    sys.exit(main())
