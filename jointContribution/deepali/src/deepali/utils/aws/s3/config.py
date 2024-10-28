from typing import NamedTuple
from typing import Optional

from boto3.s3.transfer import TransferConfig

KiB = 1024
MiB = 1024**2


class S3Config(NamedTuple):
    r"""Configuration of AWS Simple Storage Service."""

    mode: str = "rwd"
    verify: bool = True
    region: Optional[str] = None
    multipart_threshold: int = 8 * MiB
    max_concurrency: int = 10
    multipart_chunksize: int = 8 * MiB
    num_download_attempts: int = 5
    max_io_queue: int = 100
    io_chunksize: int = 256 * KiB
    use_threads: bool = True
    max_bandwidth: Optional[int] = None

    def transfer_config(self) -> TransferConfig:
        return TransferConfig(
            multipart_threshold=self.multipart_threshold,
            max_concurrency=self.max_concurrency,
            multipart_chunksize=self.multipart_chunksize,
            num_download_attempts=self.num_download_attempts,
            max_io_queue=self.max_io_queue,
            io_chunksize=self.io_chunksize,
            use_threads=self.use_threads,
            max_bandwidth=self.max_bandwidth,
        )
