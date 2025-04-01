from typing import List

import paddle
import paddle.tensor as Tensor


def _cast_squeeze_in(img: Tensor, req_dtypes: List[paddle.dtype]):
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(axis=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype


def _get_gaussian_kernel1d(
    kernel_size: int, sigma: float, dtype: paddle.dtype
) -> Tensor:
    ksize_half = (kernel_size - 1) * 0.5

    x = paddle.linspace(-ksize_half, ksize_half, num=kernel_size, dtype=dtype)
    pdf = paddle.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d


def _get_gaussian_kernel2d(
    kernel_size: List[int], sigma: List[float], dtype: paddle.dtype
) -> Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0], dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1], dtype)
    kernel2d = paddle.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d


def _cast_squeeze_out(
    img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: paddle.dtype
) -> Tensor:
    if need_squeeze:
        img = img.squeeze(axis=0)

    if need_cast:
        if out_dtype in (
            paddle.uint8,
            paddle.int8,
            paddle.int16,
            paddle.int32,
            paddle.int64,
        ):
            # it is better to round before cast
            img = paddle.round(img)
        img = img.to(out_dtype)

    return img


def gaussian_blur(img: Tensor, kernel_size: List[int], sigma: List[float]) -> Tensor:
    if sigma is None:
        sigma = [ksize * 0.15 + 0.35 for ksize in kernel_size]

    if sigma is not None and not isinstance(sigma, (int, float, list, tuple)):
        raise TypeError(
            f"sigma should be either float or sequence of floats. Got {type(sigma)}"
        )
    if isinstance(sigma, (int, float)):
        sigma = [float(sigma), float(sigma)]
    if isinstance(sigma, (list, tuple)) and len(sigma) == 1:
        sigma = [sigma[0], sigma[0]]
    if len(sigma) != 2:
        raise ValueError(
            f"If sigma is a sequence, its length should be 2. Got {len(sigma)}"
        )
    for s in sigma:
        if s <= 0.0:
            raise ValueError(f"sigma should have positive values. Got {sigma}")
    # print(f"img: {img}")
    # if not (isinstance(img, Tensor)):
    #     raise TypeError(f"img should be Tensor. Got {type(img)}")

    dtype = img.dtype if paddle.is_floating_point(img) else paddle.float32
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype)
    kernel = kernel.expand([img.shape[-3], 1, kernel.shape[0], kernel.shape[1]])

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [kernel.dtype])

    # padding = (left, right, top, bottom)
    padding = [
        kernel_size[0] // 2,
        kernel_size[0] // 2,
        kernel_size[1] // 2,
        kernel_size[1] // 2,
    ]
    img = paddle.nn.functional.pad(img, padding, mode="reflect")
    img = paddle.nn.functional.conv2d(img, kernel, groups=img.shape[-3])

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img
