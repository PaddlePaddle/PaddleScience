"""External Development Packages"""

import importlib.util

EXTERNAL_PACKAGES_LIST = [
    "deepali",
    "neuraloperator",
    "open3d",
    "paddle_harmonics",
    "tensorly",
    "warp",
]

__all__ = []
for package_name in EXTERNAL_PACKAGES_LIST:
    if importlib.util.find_spec(package_name):
        globals()[package_name] = __import__(package_name)
        __all__.append(package_name)
