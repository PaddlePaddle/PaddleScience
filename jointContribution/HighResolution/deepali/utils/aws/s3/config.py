from typing import NamedTuple
from typing import Optional


class S3Config(NamedTuple):
    """Configuration of AWS Simple Storage Service."""

    mode: str = "r"
    verify: bool = True
    region: Optional[str] = None
