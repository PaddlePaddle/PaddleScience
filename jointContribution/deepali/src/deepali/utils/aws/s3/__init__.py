r"""Interfaces and utilities for AWS Simple Storage Service (S3)."""
from .client import S3Client
from .config import S3Config
from .object import S3Object

__all__ = "S3Config", "S3Client", "S3Object"
