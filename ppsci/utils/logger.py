# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import functools
import logging
import os
import sys
from typing import Callable
from typing import Optional

import colorlog
import paddle.distributed as dist

from ppsci.utils import misc

_logger: logging.Logger = None

# INFO(20) is white(no color)
# use custom log level `MESSAGE` for printing message in color
MESSAGE = 25

COLORLOG_CONFIG = {
    "DEBUG": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "MESSAGE": "cyan",
}


def init_logger(
    name: str = "ppsci",
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
) -> None:
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the logger by
    adding one or two handlers, otherwise the initialized logger will be directly
    returned. During initialization, a StreamHandler will always be added. If `log_file`
    is specified a FileHandler will also be added.

    Args:
        name (str, optional): Logger name. Defaults to "ppsci".
        log_file (Optional[str]): The log filename. If specified, a FileHandler
            will be added to the logger. Defaults to None.
        log_level (int, optional): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time. Defaults to logging.INFO.
    """
    # Add custom log level MESSAGE(25), between WARNING(30) and INFO(20)
    logging.addLevelName(MESSAGE, "MESSAGE")

    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())

    global _logger

    # get a clean logger
    _logger = logging.getLogger(name)
    _logger.handlers.clear()

    # add stream_handler, output to stdout such as terminal
    stream_formatter = colorlog.ColoredFormatter(
        "%(log_color)s[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        log_colors=COLORLOG_CONFIG,
    )
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(stream_formatter)
    stream_handler._name = "stream_handler"
    _logger.addHandler(stream_handler)

    # add file_handler, output to log_file(if specified)
    if log_file is not None and dist.get_rank() == 0:
        log_file_folder = os.path.split(log_file)[0]
        os.makedirs(log_file_folder, exist_ok=True)
        file_formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(log_file, "a")  # append mode
        file_handler.setFormatter(file_formatter)
        file_handler._name = "file_handler"
        _logger.addHandler(file_handler)

    if dist.get_rank() == 0:
        _logger.setLevel(log_level)
    else:
        _logger.setLevel(logging.ERROR)

    _logger.propagate = False


def set_log_level(log_level: int):
    """Set logger level, only msg of level >= `log_level` will be printed.

    Args:
        log_level (int): Log level.
    """
    if dist.get_rank() == 0:
        _logger.setLevel(log_level)
    else:
        _logger.setLevel(logging.ERROR)


def ensure_logger(log_func: Callable):
    """
    Automatically initialize `logger` by default arguments
    when init_logger() is not called manually.
    """

    @functools.wraps(log_func)
    def wrapped_log_func(msg, *args):
        if _logger is None:
            init_logger()
            _logger.warning(
                "Before you call functions within the logger, the logger has already "
                "been automatically initialized. Since `log_file` is not specified by "
                "default, information will not be written to any file except being "
                "output to the terminal."
            )

        log_func(msg, *args)

    return wrapped_log_func


@ensure_logger
@misc.run_at_rank0
def info(msg, *args):
    _logger.info(msg, *args)


@ensure_logger
@misc.run_at_rank0
def message(msg, *args):
    _logger.log(MESSAGE, msg, *args)


@ensure_logger
@misc.run_at_rank0
def debug(msg, *args):
    _logger.debug(msg, *args)


@ensure_logger
@misc.run_at_rank0
def warning(msg, *args):
    _logger.warning(msg, *args)


@ensure_logger
@misc.run_at_rank0
def error(msg, *args):
    _logger.error(msg, *args)


def scaler(name: str, value: float, step: int, vdl_writer=None, wandb_writer=None):
    """
    This function will draw a scalar curve generated by the visualdl.
    Usage: Install visualdl: pip3 install visualdl==2.0.0b4
           and then:
           visualdl --logdir ./scalar --host 0.0.0.0 --port 8830
           to preview loss corve in real time.
    """
    if vdl_writer is not None:
        vdl_writer.add_scalar(tag=name, step=step, value=value)
    if wandb_writer is not None:
        wandb_writer.log({"step": step, name: value})


def advertise():
    """
    Show the advertising message like the following:

    ===========================================================
    ==      PaddleScience is powered by PaddlePaddle !       ==
    ===========================================================
    ==                                                       ==
    ==   For more info please go to the following website.   ==
    ==                                                       ==
    ==     https://github.com/PaddlePaddle/PaddleScience     ==
    ===========================================================

    """
    _copyright = "PaddleScience is powered by PaddlePaddle !"
    ad = "Please refer to the following website for more info."
    website = "https://github.com/PaddlePaddle/PaddleScience"
    AD_LEN = 6 + len(max([_copyright, ad, website], key=len))

    info(
        "\n{0}\n{1}\n{2}\n{3}\n{4}\n{5}\n{6}\n{7}\n".format(
            "=" * (AD_LEN + 4),
            "=={}==".format(_copyright.center(AD_LEN)),
            "=" * (AD_LEN + 4),
            "=={}==".format(" " * AD_LEN),
            "=={}==".format(ad.center(AD_LEN)),
            "=={}==".format(" " * AD_LEN),
            "=={}==".format(website.center(AD_LEN)),
            "=" * (AD_LEN + 4),
        )
    )
