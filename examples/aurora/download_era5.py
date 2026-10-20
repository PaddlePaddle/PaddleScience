"""Fetch an ERA5 verification interval and Aurora variables from WeatherBench2.

Reads documented Zarr v2 chunks directly to avoid installing a cloud SDK.
Each downloaded object has a local SHA256 and the source GCS ETag recorded.
"""

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import netCDF4
import numpy as np
import requests
from numcodecs import get_codec
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path("data/aurora")
SOURCE = (
    "https://storage.googleapis.com/weatherbench2/datasets/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)
SURFACE = (
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
)
ATMOS = (
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
)
STATIC = ("land_sea_mask", "soil_type", "geopotential_at_surface")
COORDS = ("latitude", "longitude", "level", "time")
START = int((datetime(2023, 1, 1) - datetime(1959, 1, 1)).total_seconds() / 21600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--directory", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--request-bytes",
        type=int,
        default=0,
        help="Bound HTTP range responses; 0 streams a complete object",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Use direct HTTPS instead of environment proxy settings",
    )
    parser.add_argument("--seed-cache", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.workers < 1:
        parser.error("steps and workers must be positive")
    root = args.directory
    start_time = datetime.fromisoformat(args.start)
    if start_time.tzinfo is not None:
        parser.error("start must be a timezone-naive date/time interpreted as UTC")
    if start_time.hour % 6 or start_time.minute or start_time.second:
        parser.error("start must be on an exact 6-hour UTC boundary")
    start = int((start_time - datetime(1959, 1, 1)).total_seconds() / 21600)
    count = args.steps + 2
    run_download(
        root,
        start,
        count,
        start_time,
        args.workers,
        args.seed_cache or args.directory / "source-chunks",
        args.direct,
        args.request_bytes,
    )


def run_download(
    ROOT, START, count, start_time, workers, seed_cache, direct=False, request_bytes=0
):
    ROOT.mkdir(parents=True, exist_ok=True)
    retry = Retry(total=6, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504))
    session = requests.Session()
    session.trust_env = not direct
    session.mount(
        "https://", HTTPAdapter(max_retries=retry, pool_maxsize=max(8, workers))
    )
    metadata_file = ROOT / "source.zmetadata.json"
    if not metadata_file.exists():
        r = session.get(SOURCE + "/.zmetadata", timeout=120)
        r.raise_for_status()
        metadata_file.write_text(r.text)
    meta = json.loads(metadata_file.read_text())["metadata"]
    if START < 0 or START + count > meta["time/.zarray"]["shape"][0]:
        raise ValueError("Requested UTC interval lies outside source time coordinates")

    tasks = []
    for name in SURFACE + ATMOS:
        shape = meta[name + "/.zarray"]
        assert shape["chunks"][0] == 1
        assert shape["chunks"][1:] == shape["shape"][1:]
        for index in range(START, START + count):
            key = ".".join(map(str, [index] + [0] * (len(shape["shape"]) - 1)))
            tasks.append((name, key))
    for name in STATIC + COORDS:
        config = meta[name + "/.zarray"]
        key = ".".join(["0"] * len(config["shape"]))
        if name == "time":
            for chunk in range(
                START // config["chunks"][0],
                (START + count - 1) // config["chunks"][0] + 1,
            ):
                tasks.append((name, str(chunk)))
            continue
        tasks.append((name, key))

    def fetch(task):
        name, key = task
        path = ROOT / "source-chunks" / name / key
        sidecar = path.with_name(key + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        cached = seed_cache / name / key
        if (
            not path.exists()
            and cached.exists()
            and cached.with_name(key + ".json").exists()
        ):
            record = json.loads(cached.with_name(key + ".json").read_text())
            if hashlib.sha256(cached.read_bytes()).hexdigest() == record["sha256"]:
                path.hardlink_to(cached)
                sidecar.write_text(json.dumps(record, indent=2) + "\n")
        if path.exists() and sidecar.exists():
            record = json.loads(sidecar.read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]:
                print("Cached", name, key, flush=True)
                return record
        partial = path.with_name(key + ".partial")
        completed = False
        for attempt in range(12) if not request_bytes else ():
            offset = partial.stat().st_size if partial.exists() else 0
            try:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with session.get(
                    f"{SOURCE}/{name}/{key}",
                    headers=headers,
                    stream=True,
                    timeout=(60, 120),
                ) as r:
                    r.raise_for_status()
                    if offset and r.status_code == 206:
                        assert r.headers["Content-Range"].startswith(f"bytes {offset}-")
                        total = int(r.headers["Content-Range"].split("/")[-1])
                        mode = "ab"
                    else:
                        total = int(r.headers["Content-Length"])
                        mode = "wb"
                    with partial.open(mode) as f:
                        for chunk in r.iter_content(1024 * 1024):
                            f.write(chunk)
                    assert partial.stat().st_size == total
                    etag = r.headers.get("ETag")
                    completed = True
                    break
            except requests.RequestException as exc:
                print(f"Retry {name}/{key}: {type(exc).__name__}", flush=True)
                time.sleep(min(2**attempt, 20))
        if request_bytes:
            etag = fetch_bounded(
                session, f"{SOURCE}/{name}/{key}", partial, request_bytes
            )
            completed = True
        if not completed:
            raise RuntimeError(f"Could not finish {name}/{key}; rerun to resume")
        record = {
            "variable": name,
            "chunk": key,
            "bytes": partial.stat().st_size,
            "sha256": hashlib.sha256(partial.read_bytes()).hexdigest(),
            "etag": etag,
        }
        partial.replace(path)
        sidecar.write_text(json.dumps(record, indent=2) + "\n")
        print("Downloaded", name, key, record["bytes"], flush=True)
        return record

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(fetch, tasks))

    def decode(name, key):
        config = meta[name + "/.zarray"]
        data = (ROOT / "source-chunks" / name / key).read_bytes()
        if config["compressor"]:
            data = get_codec(config["compressor"]).decode(data)
        return np.frombuffer(data, dtype=config["dtype"]).reshape(config["chunks"])

    coords = {
        name: decode(name, "0")[: meta[name + "/.zarray"]["shape"][0]]
        for name in COORDS
        if name != "time"
    }
    tc = meta["time/.zarray"]["chunks"][0]
    time_chunks = {
        index // tc: decode("time", str(index // tc))
        for index in range(START, START + count)
    }
    times = np.asarray(
        [time_chunks[index // tc][index % tc] for index in range(START, START + count)]
    )
    actual_times = netCDF4.num2date(
        times, meta["time/.zattrs"]["units"], only_use_cftime_datetimes=False
    )
    assert [str(t) for t in actual_times] == [
        str(start_time + timedelta(hours=6 * step)) for step in range(count)
    ]
    assert np.array_equal(
        coords["level"],
        [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
    )
    assert np.all(np.diff(coords["latitude"]) < 0)
    assert np.all(np.diff(coords["longitude"]) > 0)

    outputs = []
    for filename, names in (
        ("surface.nc", SURFACE),
        ("atmospheric.nc", ATMOS),
        ("static.nc", STATIC),
    ):
        path = ROOT / filename
        temporary = path.with_suffix(".nc.partial")
        with netCDF4.Dataset(temporary, "w") as nc:
            for name, values in coords.items():
                nc.createDimension(name, len(values))
                var = nc.createVariable(name, values.dtype, (name,))
                var[:] = values
                for key, value in meta[name + "/.zattrs"].items():
                    if key != "_ARRAY_DIMENSIONS":
                        var.setncattr(key, value)
            if names != STATIC:
                nc.createDimension("time", count)
                var = nc.createVariable("time", "i8", ("time",))
                var[:] = times
                var.units = meta["time/.zattrs"]["units"]
                var.calendar = meta["time/.zattrs"]["calendar"]
            for name in names:
                attrs = meta[name + "/.zattrs"]
                dims = tuple(attrs["_ARRAY_DIMENSIONS"])
                var = nc.createVariable(
                    name,
                    "f4",
                    dims,
                    zlib=True,
                    complevel=2,
                    chunksizes=tuple(
                        1 if d in ("time", "level") else len(coords[d]) for d in dims
                    ),
                )
                for key, value in attrs.items():
                    if key != "_ARRAY_DIMENSIONS":
                        var.setncattr(key, value)
                if names == STATIC:
                    values = decode(name, "0.0")
                    assert np.isfinite(values).all(), name
                    var[:] = values
                else:
                    for step, index in enumerate(range(START, START + count)):
                        key = ".".join(map(str, [index] + [0] * (len(dims) - 1)))
                        values = decode(name, key)[0]
                        assert np.isfinite(values).all(), (name, index)
                        var[step] = values
            nc.source = SOURCE
            nc.history = f"Subset only: {actual_times[0]} through {actual_times[-1]} UTC; no regridding or normalization."
        temporary.replace(path)
        outputs.append(
            {
                "file": filename,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        print("Saved", filename, path.stat().st_size, flush=True)
    report = {
        "status": "downloaded_and_verified",
        "source": SOURCE,
        "source_documentation": "https://weatherbench2.readthedocs.io/en/latest/data-guide.html",
        "times_utc": [str(t) for t in actual_times],
        "grid": [721, 1440],
        "pressure_levels_hpa": coords["level"].tolist(),
        "variables": {"surface": SURFACE, "atmospheric": ATMOS, "static": STATIC},
        "source_objects": records,
        "outputs": outputs,
    }
    (ROOT / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print("ERA5 subset complete.", flush=True)


def fetch_bounded(session, url, partial, block_bytes):
    """Resume short byte ranges; interrupted responses never append partial data."""
    etag = None
    while True:
        offset = partial.stat().st_size if partial.exists() else 0
        for attempt in range(12):
            try:
                range_url = f"{url}?aurora_range={offset}-{offset + block_bytes - 1}"
                with session.get(
                    range_url,
                    headers={"Range": f"bytes={offset}-{offset + block_bytes - 1}"},
                    timeout=(20, 30),
                ) as response:
                    content_range = response.headers.get("Content-Range", "")
                    if (
                        response.status_code == 416
                        and content_range == f"bytes */{offset}"
                        and offset > 0
                    ):
                        return etag or response.headers.get("ETag")
                    response.raise_for_status()
                    if response.status_code != 206 or not content_range.startswith(
                        f"bytes {offset}-"
                    ):
                        raise ValueError("Server did not honor exact ERA5 byte range")
                    end_text, total_text = content_range.split("/")
                    end = int(end_text.rsplit("-", 1)[1])
                    total = int(total_text)
                    if (
                        end != min(offset + block_bytes, total) - 1
                        or len(response.content) != end - offset + 1
                    ):
                        raise ValueError("Incomplete or mismatched ERA5 byte range")
                    current_etag = response.headers.get("ETag")
                    if etag and current_etag != etag:
                        raise ValueError("ERA5 object ETag changed during transfer")
                    etag = current_etag
                    with partial.open("ab") as stream:
                        stream.write(response.content)
                    if end + 1 == total:
                        return etag
                    break
            except requests.RequestException as exc:
                print(
                    f"Range retry {partial.name} at {offset}: {type(exc).__name__}",
                    flush=True,
                )
                time.sleep(min(2**attempt, 20))
        else:
            raise RuntimeError(
                f"ERA5 byte range could not finish: {url}; rerun to resume"
            )


if __name__ == "__main__":
    main()
