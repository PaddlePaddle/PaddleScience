r"""Utilities for setting up logging in a main scripts."""

from __future__ import annotations  # noqa

import logging
from argparse import Namespace
from enum import Enum
from logging import Logger
from typing import Optional
from typing import Union


class LogLevel(str, Enum):
    r"""Enumeration of logging levels.

    In particular, this enumeration can be used for type annotation of log level argument when using Typer.

    """

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    @classmethod
    def from_arg(cls, value: Union[int, str, LogLevel, None]) -> LogLevel:
        if value is None:
            return LogLevel.NOTSET
        if isinstance(value, LogLevel):
            return value
        if isinstance(value, int):
            return cls.from_int(value)
        if isinstance(value, str):
            return cls.from_str(value)
        raise TypeError(f"{cls.__name__}.from_arg() 'value' must be int, str, LogLevel, or None")

    @classmethod
    def from_int(cls, value: int) -> LogLevel:
        if not isinstance(value, int):
            raise TypeError(f"{cls.__name__}.from_int() 'value' must be int")
        log_level = LogLevel.NOTSET
        for level in cls:
            if int(level) > value:
                break
            log_level = level
        return log_level

    @classmethod
    def from_str(cls, value: str) -> LogLevel:
        if not isinstance(value, str):
            raise TypeError(f"{cls.__name__}.from_str() 'value' must be str")
        return cls(value.upper())

    @classmethod
    def from_logger(cls, logger: Logger) -> LogLevel:
        return cls.from_int(logger.level)

    def __str__(self) -> str:
        return self.value

    def __int__(self) -> int:
        r"""Cast enumeration to int logging level."""
        return int(getattr(logging, self.value))

    def __eq__(self, other: Union[LogLevel, int]) -> bool:
        return int(self) == int(other)

    def __lt__(self, other: Union[LogLevel, int]) -> bool:
        return int(self) < int(other)

    def __le__(self, other: Union[LogLevel, int]) -> bool:
        return int(self) <= int(other)

    def __gt__(self, other: Union[LogLevel, int]) -> bool:
        return int(self) > int(other)

    def __ge__(self, other: Union[LogLevel, int]) -> bool:
        return int(self) <= int(other)


LOG_FORMAT = "%(asctime)-15s [%(levelname)s] %(message)s"
LOG_LEVELS = tuple(log_level.value for log_level in LogLevel)


def configure_logging(
    logger: Logger,
    args: Optional[Namespace] = None,
    log_level: Optional[Union[int, str, LogLevel]] = None,
    format: Optional[str] = None,
):
    r"""Initialize logging."""
    logging.basicConfig(format=format or LOG_FORMAT)
    if args is not None:
        log_level = getattr(args, "log_level", log_level)
    log_level = LogLevel.from_arg(log_level)
    if log_level is LogLevel.NOTSET:
        log_level = LogLevel.INFO
    logger.setLevel(log_level.value)
    logging.getLogger("botocore").setLevel(logging.ERROR)
    logging.getLogger("s3transfer").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
