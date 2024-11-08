"""Utility functions for `SimpleITK <https://simpleitk.org/>`_ data objects."""
from .imageio import read_image
from .imageio import write_image

__all__ = "read_image", "write_image"
