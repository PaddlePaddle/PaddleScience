"""Auxiliary functions for conversion between SimpleITK and P."""
from typing import Optional
from typing import Sequence

import paddle
import SimpleITK as sitk


def image_from_tensor(
    data: paddle.Tensor,
    origin: Optional[Sequence[float]] = None,
    spacing: Optional[Sequence[float]] = None,
    direction: Optional[Sequence[float]] = None,
) -> sitk.Image:
    """Create ``SimpleITK.Image`` from image data tensor.

    Args:
        data: Image tensor of shape ``(C, ..., X)``.
        origin: World coordinates of center of voxel with zero indices.
        spacing: Voxel size in world units in each dimension.
        direction: Flattened image orientation cosine matrix.

    Returns:
        SimpleITK image.

    """
    data = data.detach().cpu()
    nchannels = tuple(data.shape)[0]
    if nchannels == 1:
        data = data[0]
    else:
        x = data.unsqueeze(axis=-1)
        perm_0 = list(range(x.ndim))
        perm_0[0] = -1
        perm_0[-1] = 0
        data = x.transpose(perm=perm_0).squeeze(axis=0)
    image = sitk.GetImageFromArray(data.numpy(), isVector=nchannels > 1)
    if origin:
        image.SetOrigin(origin)
    if spacing:
        image.SetSpacing(spacing)
    if direction:
        image.SetDirection(direction)
    return image


# def tensor_from_image(image: sitk.Image, dtype: Optional[paddle.dtype]=None,
#     device: Optional=None) ->paddle.Tensor:
#     """Create image data tensor from ``SimpleITK.Image``."""
#     if image.GetPixelID() == sitk.sitkUInt16:
#         image = sitk.Cast(image, sitk.sitkInt32)
#     elif image.GetPixelID() == sitk.sitkUInt32:
#         image = sitk.Cast(image, sitk.sitkInt64)
#     data = paddle.to_tensor(sitk.GetArrayFromImage(image))
#     data = data.unsqueeze(axis=0)
#     if image.GetNumberOfComponentsPerPixel() > 1:
#         x = data
#         perm_1 = list(range(x.ndim))
#         perm_1[0] = -1
#         perm_1[-1] = 0
#         data = x.transpose(perm=perm_1).squeeze(axis=-1)
#     return data.astype(dtype)


def tensor_from_image(
    image: sitk.Image,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[str] = None,
) -> paddle.Tensor:
    """Create image data tensor from ``SimpleITK.Image``."""
    if image.GetPixelID() == sitk.sitkUInt16:
        image = sitk.Cast(image, sitk.sitkInt32)
    elif image.GetPixelID() == sitk.sitkUInt32:
        image = sitk.Cast(image, sitk.sitkInt64)
    data = paddle.to_tensor(sitk.GetArrayFromImage(image), place=device)
    data = data.unsqueeze(axis=0)
    if image.GetNumberOfComponentsPerPixel() > 1:
        x = data
        perm_1 = list(range(x.ndim))
        perm_1[0] = x.ndim - 1  # 将第0维度移动到最后
        perm_1[-1] = 0  # 将最后一维移动到第0维
        data = x.transpose(perm=perm_1).squeeze(axis=-1)
    if dtype is not None:
        data = data.astype(dtype)
    else:
        data = data.astype(paddle.float32)  # 默认使用float32类型

    return data
