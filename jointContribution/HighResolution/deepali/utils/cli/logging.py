"""Auxiliary functions to set up logging in main scripts."""
import logging
from enum import Enum
from typing import Optional

LOG_FORMAT = "%(asctime)-15s [%(levelname)s] %(message)s"


class LogLevel(str, Enum):
    """Enumeration of logging levels for use as type annotation when using Typer."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

    def __int__(self) -> int:
        """Cast enumeration to int logging level."""
        return int(getattr(logging, self.value))


def configure_logging(log, args, format: Optional[str] = None):
    """Initialize logging."""
    logging.basicConfig(format=format or LOG_FORMAT)
    if hasattr(args, "log_level"):
        log.setLevel(args.log_level)
    else:
        log.setLevel(logging.INFO)
