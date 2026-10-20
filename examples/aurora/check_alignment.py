"""Run separate Torch/Paddle processes against identical weights and inputs.

Torch writes weights.npz and reference.npz. Paddle loads them and reports
per-stage errors for two autoregressive forecast steps. No Torch import is
needed in the Paddle process.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from sample import make_batch
from sample import small_config


def main():
    # Upstream interprets naive datetimes in the process timezone. ERA5 times
    # are UTC; set only this validation process to UTC for a reproducible base.
    os.environ["TZ"] = "UTC"
    time.tzset()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("torch", "paddle"), required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input", type=Path, help="Saved real-data Batch NetCDF")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--official-output",
        type=Path,
        help="Verified official reference output NPZ, for first-step regression",
    )
    parser.add_argument(
        "--paddle-checkpoint",
        type=Path,
        help="Optional original-layout NPZ or native pdparams",
    )
    parser.add_argument("--save-paddle-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--debug", action="store_true", help="Capture encoder sublayers too"
    )
    parser.add_argument("--activation-atol", type=float, default=1e-3)
    parser.add_argument("--prediction-atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=2e-4)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)

    if args.framework == "torch":
        import torch as framework
        from aurora import AuroraSmallPretrained
        from aurora import Batch
        from aurora import Metadata
        from aurora import rollout

        framework.set_num_threads(4)
        framework.manual_seed(13)
        model = AuroraSmallPretrained(**({} if args.checkpoint else small_config()))
        if args.checkpoint:
            model.load_checkpoint_local(str(args.checkpoint))
        else:
            # Exercise attention branches that zero-initialized adaptive norms
            # would otherwise silence in a randomly initialized model.
            with framework.no_grad():
                for name, param in model.named_parameters():
                    if "ln_modulation.1" in name:
                        framework.nn.init.normal_(param, std=0.02)
        with (args.directory / "weights.npz").open("wb") as stream:
            np.savez(
                stream,
                **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()},
            )
    else:
        import paddle as framework

        from ppsci.arch.aurora import AuroraSmallPretrained
        from ppsci.arch.aurora import Batch
        from ppsci.arch.aurora import Metadata
        from ppsci.arch.aurora import rollout

        framework.set_device(args.device)
        model = AuroraSmallPretrained(**({} if args.checkpoint else small_config()))
        model.load_checkpoint_local(
            str(args.paddle_checkpoint or args.directory / "weights.npz")
        )
        if args.save_paddle_checkpoint:
            args.save_paddle_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            framework.save(model.state_dict(), str(args.save_paddle_checkpoint))

    model = model.to(device=args.device, dtype=getattr(framework, args.dtype))
    model.eval()
    batch = (
        Batch.from_netcdf(args.input)
        if args.input
        else make_batch(framework, Batch, Metadata)
    ).to(args.device)
    trace = {}
    counts = {}

    def array(value):
        return value.detach().cpu().numpy()

    def hook(name):
        def capture(layer, inputs, output):
            count = counts.get(name, 0)
            counts[name] = count + 1
            prefix = f"{name}.{count}"
            if hasattr(output, "surf_vars"):
                for kind in ("surf_vars", "atmos_vars"):
                    for key, value in getattr(output, kind).items():
                        trace[f"{prefix}.{kind}.{key}"] = array(value)
            elif isinstance(output, framework.Tensor):
                trace[prefix] = array(output)

        return capture

    layers = {name: getattr(model, name) for name in ("encoder", "backbone", "decoder")}
    if args.debug:
        sublayers = (
            model.encoder.named_modules()
            if args.framework == "torch"
            else model.encoder.named_sublayers()
        )
        layers.update({f"encoder.{name}": layer for name, layer in sublayers if name})
    for name, layer in layers.items():
        if args.framework == "torch":
            layer.register_forward_hook(hook(name))
        else:
            layer.register_forward_post_hook(hook(name))
    with framework.no_grad():
        predictions = list(rollout(model, batch, steps=args.steps))
    fixture_results = {}
    if args.official_output:
        # Match the upstream regression criterion: mean absolute error divided
        # by mean reference magnitude, in physical units. These reference
        # outputs are model predictions, not meteorological ground truth.
        tolerances = {
            "2t": 1e-4,
            "msl": 1e-4,
            "t": 1e-4,
            "10u": 5e-3,
            "10v": 5e-3,
            "u": 5e-3,
            "v": 5e-3,
            "q": 5e-3,
        }
        with np.load(args.official_output) as official:
            pred = predictions[0]
            for kind in ("surf_vars", "atmos_vars"):
                for key, value in getattr(pred, kind).items():
                    actual = array(value).astype("float64")
                    expected = official[f"{kind}.{key}"].astype("float64")
                    assert actual.shape == expected.shape
                    relative_mae = float(
                        np.abs(actual - expected).mean()
                        / max(np.abs(expected).mean(), 1e-12)
                    )
                    fixture_results[f"{kind}.{key}"] = {
                        "relative_mae": relative_mae,
                        "tolerance": tolerances[key],
                        "passed": relative_mae <= tolerances[key],
                    }
            np.testing.assert_allclose(array(pred.metadata.lat), official["lat"])
            np.testing.assert_allclose(array(pred.metadata.lon), official["lon"])
            assert tuple(pred.metadata.atmos_levels) == tuple(official["levels"])
            assert str(pred.metadata.time[0]) == str(official["time"].item())
        fixture_report = {
            "status": "passed"
            if all(v["passed"] for v in fixture_results.values())
            else "failed",
            "framework": args.framework,
            "dtype": args.dtype,
            "purpose": "Upstream model-output regression; not forecast skill evaluation",
            "results": fixture_results,
        }
        (args.directory / f"official-regression-{args.framework}.json").write_text(
            json.dumps(fixture_report, indent=2) + "\n"
        )
        assert fixture_report["status"] == "passed", fixture_report
    if args.save_predictions:
        for step, pred in enumerate(predictions, 1):
            pred.to("cpu").to_netcdf(
                args.directory / f"{args.framework}-prediction-{step:03}.nc"
            )
    for step, pred in enumerate(predictions):
        for kind in ("surf_vars", "atmos_vars"):
            for key, value in getattr(pred.normalise(model.surf_stats), kind).items():
                trace[f"normalized.{step}.{kind}.{key}"] = array(value)

    if args.framework == "torch":
        np.savez(args.directory / "reference.npz", **trace)
        print(f"Saved {len(trace)} reference tensors and canonical Torch weights.")
        return

    results = {}
    failures = []
    with np.load(args.directory / "reference.npz") as reference:
        assert set(reference.files) == set(trace)
        for key, value in trace.items():
            expected = reference[key]
            error = np.abs(value.astype("float64") - expected.astype("float64"))
            relative_l2 = float(
                np.linalg.norm(error) / max(np.linalg.norm(expected), 1e-12)
            )
            results[key] = {"max_abs": float(error.max()), "relative_l2": relative_l2}
            atol = (
                args.activation_atol
                if key.startswith(("encoder.", "backbone."))
                else args.prediction_atol
            )
            if not np.allclose(value, expected, rtol=args.rtol, atol=atol):
                failures.append(key)
    report = {
        "status": "failed" if failures else "passed",
        "framework": framework.__version__,
        "device": args.device,
        "dtype": args.dtype,
        "timezone": "UTC",
        "checkpoint": str(args.checkpoint) if args.checkpoint else "synthetic",
        "input": str(args.input) if args.input else "synthetic",
        "rollout_steps": args.steps,
        "official_reference_regression": fixture_results,
        "tensors_compared": len(results),
        "failed_tensors": failures,
        "prediction_alignment_status": (
            "failed"
            if any(k.startswith(("decoder.", "normalized.")) for k in failures)
            else "passed"
        ),
        "activation_alignment_status": (
            "failed"
            if any(k.startswith(("encoder.", "backbone.")) for k in failures)
            else "passed"
        ),
        "tolerances": {
            "rtol": args.rtol,
            "activation_atol": args.activation_atol,
            "prediction_atol": args.prediction_atol,
        },
        "results": results,
    }
    output = args.directory / f"alignment-{args.device.replace(':', '-')}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    assert not failures, f"Alignment failed for {failures}"


if __name__ == "__main__":
    main()
