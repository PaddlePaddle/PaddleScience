"""Download pinned public Aurora regression fixtures and verify LFS hashes."""

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

DEFAULT_DIRECTORY = Path("data/aurora-official")
REVISION = "a96afd7ee6d65e3bd2d476f3be798a25a56f2296"
NAMES = (
    "aurora-0.25-small-pretrained-test-input.pickle",
    "aurora-0.25-small-pretrained-test-output.pickle",
    "aurora-0.25-static.pickle",
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    directory = parser.parse_args().directory
    directory.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        f"https://huggingface.co/api/models/microsoft/aurora/revision/{REVISION}?blobs=true",
        timeout=60,
    )
    response.raise_for_status()
    files = {x["rfilename"]: x for x in response.json()["siblings"]}

    def fetch(name):
        info = files[name]
        path = directory / name
        expected = info["lfs"]["sha256"]
        if not (path.exists() and sha256(path) == expected):
            partial = path.with_suffix(path.suffix + ".partial")
            url = f"https://huggingface.co/microsoft/aurora/resolve/{REVISION}/{name}"
            for attempt in range(12):
                offset = partial.stat().st_size if partial.exists() else 0
                if offset == info["size"]:
                    break
                try:
                    headers = {"Range": f"bytes={offset}-"} if offset else {}
                    with requests.get(
                        url + f"?resume_offset={offset}",
                        headers=headers,
                        stream=True,
                        timeout=(60, 180),
                    ) as r:
                        r.raise_for_status()
                        if offset and r.status_code == 206:
                            assert r.headers["Content-Range"].startswith(
                                f"bytes {offset}-"
                            )
                            mode = "ab"
                        else:
                            mode = "wb"
                        with partial.open(mode) as f:
                            for chunk in r.iter_content(1024 * 1024):
                                f.write(chunk)
                except requests.RequestException as exc:
                    print(f"Retry {name}: {type(exc).__name__}", flush=True)
                    time.sleep(min(2**attempt, 20))
            assert partial.stat().st_size == info["size"], name
            assert sha256(partial) == expected, name
            partial.replace(path)
        result = {"file": name, "bytes": path.stat().st_size, "sha256": expected}
        print(json.dumps(result), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(fetch, NAMES))
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "source": "https://huggingface.co/microsoft/aurora",
                "revision": REVISION,
                "purpose": "Official regression fixtures; output is a model reference, not ERA5 truth.",
                "files": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
