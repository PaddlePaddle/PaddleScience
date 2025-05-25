import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import paddle

# TODO: Manually replace PaddlePaddle utility functions as required
from paddle_geometric.utils import *  # noqa

# Experimental feature flags
__experimental_flag__: Dict[str, bool] = {
    'disable_dynamic_shapes': False,
}

Options = Optional[Union[str, List[str]]]


def get_options(options: Options) -> List[str]:
    """
    Converts the provided options to a list of strings.

    Args:
        options (str, list, or None): The experimental options.

    Returns:
        List[str]: A list of experimental feature flags.
    """
    if options is None:
        options = list(__experimental_flag__.keys())
    if isinstance(options, str):
        options = [options]
    return options


def is_experimental_mode_enabled(options: Options = None) -> bool:
    """
    Checks if experimental mode is enabled.

    Args:
        options (str, list, or None): Optional experimental feature flags.

    Returns:
        bool: True if experimental mode is enabled, False otherwise.
    """
    options = get_options(options)
    return all([__experimental_flag__[option] for option in options])


def set_experimental_mode_enabled(mode: bool, options: Options = None) -> None:
    """
    Enables or disables experimental mode for specific options.

    Args:
        mode (bool): True to enable, False to disable.
        options (str, list, or None): Experimental feature flags to set.
    """
    for option in get_options(options):
        __experimental_flag__[option] = mode


class experimental_mode:
    """
    Context manager to enable experimental mode for testing unstable features.

    Example:
        with paddle_geometric.experimental_mode():
            out = model(data.x, data.edge_index)

    Args:
        options (str or list, optional): List of experimental features to enable.
    """
    def __init__(self, options: Options = None) -> None:
        self.options = get_options(options)
        self.previous_state = {
            option: __experimental_flag__[option]
            for option in self.options
        }

    def __enter__(self) -> None:
        set_experimental_mode_enabled(True, self.options)

    def __exit__(self, *args: Any) -> None:
        for option, value in self.previous_state.items():
            __experimental_flag__[option] = value


class set_experimental_mode:
    """
    Context manager to explicitly set experimental mode on or off.

    This can be used both as a function or as a context manager.

    Example:
        with set_experimental_mode(True):
            # Enable experimental mode here
    """
    def __init__(self, mode: bool, options: Options = None) -> None:
        self.options = get_options(options)
        self.previous_state = {
            option: __experimental_flag__[option]
            for option in self.options
        }
        set_experimental_mode_enabled(mode, self.options)

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: Any) -> None:
        for option, value in self.previous_state.items():
            __experimental_flag__[option] = value


def disable_dynamic_shapes(required_args: List[str]) -> Callable:
    """
    A decorator to disable dynamic shape inference for the specified arguments.

    If any of the `required_args` is missing, an error will be raised.

    Args:
        required_args (List[str]): List of argument names that must be explicitly set.

    Returns:
        Callable: Decorated function with dynamic shape validation.
    """
    def decorator(func: Callable) -> Callable:
        spec = inspect.getfullargspec(func)

        required_args_pos: Dict[str, int] = {}
        for arg_name in required_args:
            if arg_name not in spec.args:
                raise ValueError(f"The function '{func}' does not have a "
                                 f"'{arg_name}' argument")
            required_args_pos[arg_name] = spec.args.index(arg_name)

        num_args = len(spec.args)
        num_default_args = 0 if spec.defaults is None else len(spec.defaults)
        num_positional_args = num_args - num_default_args

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Skip validation if experimental mode is disabled
            if not is_experimental_mode_enabled('disable_dynamic_shapes'):
                return func(*args, **kwargs)

            for required_arg in required_args:
                index = required_args_pos[required_arg]

                value: Optional[Any] = None
                if index < len(args):  # Check positional arguments
                    value = args[index]
                elif required_arg in kwargs:  # Check keyword arguments
                    value = kwargs[required_arg]
                elif num_default_args > 0:  # Check defaults
                    assert spec.defaults is not None
                    value = spec.defaults[index - num_positional_args]

                if value is None:
                    raise ValueError(f"Dynamic shapes disabled. Argument "
                                     f"'{required_arg}' needs to be set")

            return func(*args, **kwargs)

        return wrapper

    return decorator
