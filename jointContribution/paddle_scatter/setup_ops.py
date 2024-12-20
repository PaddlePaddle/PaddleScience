import os

import paddle
from paddle.utils.cpp_extension import CppExtension
from paddle.utils.cpp_extension import CUDAExtension
from paddle.utils.cpp_extension import setup


def get_sources():
    csrc_dir_path = os.path.join(os.path.dirname(__file__), "csrc")
    cpp_files = []
    for item in os.listdir(csrc_dir_path):
        if paddle.core.is_compiled_with_cuda():
            if item.endswith(".cc") or item.endswith(".cu"):
                cpp_files.append(os.path.join(csrc_dir_path, item))
        else:
            if item.endswith(".cc"):
                cpp_files.append(os.path.join(csrc_dir_path, item))
    return csrc_dir_path, cpp_files


def get_extensions():
    src = get_sources()
    Extension = CUDAExtension if paddle.core.is_compiled_with_cuda() else CppExtension
    ext_modules = [
        Extension(
            sources=src[1],
            include_dirs=src[0],
        )
    ]
    return ext_modules


setup(
    name="paddle_scatter_ops",
    version="1.0",
    author="NKNaN",
    url="https://github.com/PaddlePaddle/PaddleScience/jointContribution/paddle_scatter",
    description="Paddle extension of scatter and segment operators with min and max reduction methods, originally from https://github.com/rusty1s/pytorch_scatter",
    ext_modules=get_extensions(),
)
