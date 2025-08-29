from dataclasses import dataclass

import ffd.register as ffd_register
import paddle
import pyvista as pv
from deepali.core import PathStr


@dataclass
class DeepaliFFDRuntimeArgs:
    """Dataclass packing registration arguments"""

    target_img: PathStr
    source_img: PathStr
    target_seg: PathStr = None
    source_seg: PathStr = None
    config: PathStr = None
    output_transform: PathStr = None
    warped_img: PathStr = None
    warped_seg: PathStr = None
    device: str = "cuda"
    debug_dir: PathStr = None
    debug: int = 0
    verbose: int = 0
    log_level: str = "WARNING"


def register_with_deepali(
    target_img_file: PathStr = None,
    source_img_file: PathStr = None,
    target_seg_file: PathStr = None,
    source_seg_file: PathStr = None,
    target_mesh_file: PathStr = None,
    ffd_params_file: PathStr = None,
    output_transform_path: PathStr = None,
    warped_img_path: PathStr = None,
    warped_mesh_path: PathStr = None,
    warped_seg_path: PathStr = None,
):
    """Register two images using FFD with GPU-enabled Deepali and transform the mesh."""
    args = DeepaliFFDRuntimeArgs(
        target_img=target_img_file,
        source_img=source_img_file,
        target_seg=target_seg_file,
        source_seg=source_seg_file,
        config=ffd_params_file,
        output_transform=output_transform_path,
        warped_img=warped_img_path,
        warped_seg=warped_seg_path,
    )
    ffd_register.init(args)
    transform = ffd_register.register_func(args)
    if target_mesh_file is not None:
        warp_transform_on_mesh(transform, target_mesh_file, warped_mesh_path)
    return transform


def warp_transform_on_mesh(transform, target_mesh_file, warped_mesh_path):
    target_mesh = pv.read(target_mesh_file)
    target_points = paddle.to_tensor(data=target_mesh.points).unsqueeze(axis=0)
    target_points = target_points.to(device=transform.place)
    warped_target_points = transform.points(target_points, axes="grid")
    target_mesh.points = warped_target_points.squeeze(axis=0).detach().cpu().numpy()
    target_mesh.save(warped_mesh_path)
