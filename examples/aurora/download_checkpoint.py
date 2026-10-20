"""Fetch the pinned official full checkpoint and verify its LFS SHA256."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

REVISION = "0be7e57c685dac86b78c4a19a3ab149d13c6a3dd"
NAME = "aurora-0.25-pretrained.ckpt"
SHA256 = "64e6c62a7043498292a6462e8c3a08171300bc166cba6b2b557021c4b684a4c9"
SIZE = 5025579446


def verify(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    if path.stat().st_size != SIZE or digest.hexdigest() != SHA256:
        raise ValueError("Official checkpoint size or SHA256 mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("checkpoints"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    path = args.directory / NAME
    if not args.verify_only and not path.exists():
        path = Path(
            hf_hub_download(
                repo_id="microsoft/aurora",
                filename=NAME,
                revision=REVISION,
                local_dir=str(args.directory),
            )
        )
    verify(path)
    path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "source": "https://huggingface.co/microsoft/aurora",
                "revision": REVISION,
                "file": NAME,
                "size_bytes": SIZE,
                "sha256": SHA256,
                "verified": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Verified official checkpoint: {path}")


if __name__ == "__main__":
    main()
