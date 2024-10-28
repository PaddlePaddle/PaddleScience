r"""Auxiliary functions for Jupyter notebooks."""

from IPython import get_ipython


def is_jupyter_notebook() -> bool:
    r"""Check if code is running in Jupyter notebook."""
    try:
        ipython = get_ipython()
        if ipython is None or "IPKernelApp" not in ipython.config:
            return False
    except ImportError:
        return False
    return True
