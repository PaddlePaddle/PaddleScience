#!/usr/bin/env python3
"""Check that the CINN compiler keeps PaddleScience cases accurate and makes them faster.

For every case in test/cinn/cases the parent process launches two workers, one
in plain dynamic graph mode and one with to_static + CINN, because the compiler
FLAGS must be set before paddle is imported. Both workers start from the same
initial weights and the same synthetic batch (the dynamic worker dumps them,
the CINN worker loads them) and measure:

    infer  model forward under no_grad: output diff and per-iteration time
    train  forward + loss (incl. PDE residuals) + backward + Adam step:
           per-step loss diff and per-iteration time

Usage:
    python test/cinn/run_cinn_test.py --list
    python test/cinn/run_cinn_test.py                          # all cases, GPU 0
    python test/cinn/run_cinn_test.py --cases ldc,laplace --devices 0,1,2,3
    python test/cinn/run_cinn_test.py --iters 50 --warmup 10 --stages infer
    python test/cinn/run_cinn_test.py --cases ntopo --output-dir copied --reuse-inputs
"""
import argparse
import concurrent.futures
import importlib
import json
import os
import queue
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent.parent  # PaddleScience root, so `import ppsci` works
for path in (str(REPO_ROOT), str(TEST_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

RESULT_PREFIX = "[CINN_TEST_RESULT] "
SEED = 2024
COMPILER_ENV_KEYS = (
    "FLAGS_prim_enable_dynamic",
    "FLAGS_prim_all",
    "FLAGS_use_cinn",
    "FLAGS_print_ir",
)
CINN_ENV = {
    "FLAGS_prim_enable_dynamic": "true",
    "FLAGS_prim_all": "true",
    "FLAGS_use_cinn": "true",
    "FLAGS_print_ir": "False",
}
STAGES = ("infer", "train")


def list_cases():
    return sorted(
        p.stem for p in (TEST_DIR / "cases").glob("*.py") if not p.stem.startswith("_")
    )


# ---------------------------------------------------------------------- worker
def flatten(obj, prefix="out"):
    """Tensor / dict / list output -> {name: Tensor}."""
    import paddle

    if isinstance(obj, paddle.Tensor):
        return {prefix: obj}
    flat = {}
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, (list, tuple)):
        items = ((str(i), v) for i, v in enumerate(obj))
    else:
        return flat
    for key, value in items:
        flat.update(flatten(value, f"{prefix}.{key}"))
    return flat


def save_batch(path, inputs, labels):
    arrays, grads = {}, {}
    for group, tensors in (("inputs", inputs), ("labels", labels)):
        for key, t in tensors.items():
            arrays[f"{group}/{key}"] = t.numpy()
            grads[f"{group}/{key}"] = not t.stop_gradient
    np.savez(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(grads))


def load_batch(path):
    import paddle

    grads = json.loads(path.with_suffix(".json").read_text())
    inputs, labels = {}, {}
    with np.load(path) as data:
        for name in data.files:
            group, key = name.split("/", 1)
            t = paddle.to_tensor(data[name])
            t.stop_gradient = not grads[name]
            (inputs if group == "inputs" else labels)[key] = t
    return inputs, labels


def disable_dropout(model):
    """Dropout draws differ between dynamic and static mode, which would mask
    real numeric differences, so every nn.Dropout* layer is set to p=0."""
    import paddle

    dropout_types = (
        paddle.nn.Dropout,
        paddle.nn.Dropout2D,
        paddle.nn.Dropout3D,
        paddle.nn.AlphaDropout,
    )
    count = 0
    for layer in model.sublayers(include_self=True):
        if isinstance(layer, dropout_types):
            layer.p = 0.0
            count += 1
    return count


def timed_loop(step_fn, total_iters, mode, stage):
    import paddle

    times, values = [], []
    for step in range(total_iters):
        start = time.perf_counter()
        value = step_fn()
        paddle.device.synchronize()
        times.append(time.perf_counter() - start)
        values.append(value)
        print(
            f"[{mode}/{stage}] iter {step + 1}/{total_iters} time {times[-1]:.5f}s",
            flush=True,
        )
    return times, values


def time_stats(times, warmup):
    steady = times[warmup:]
    return {
        "first_iter_seconds": times[0],
        "steady_mean_seconds": statistics.mean(steady),
        "steady_median_seconds": statistics.median(steady),
        "steady_stdev_seconds": statistics.stdev(steady) if len(steady) > 1 else 0.0,
    }


def worker(case_name, mode, stages, iters, warmup, work_dir, reuse_inputs=False):
    """Run one case in one mode and print its result as a JSON line.

    With reuse_inputs the dynamic worker also loads init.pdparams / batch.npz from
    work_dir instead of regenerating them, e.g. files copied from another machine."""
    import paddle

    case = importlib.import_module(f"cases.{case_name}")

    class InferNet(paddle.nn.Layer):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inputs):
            if hasattr(case, "forward"):
                return case.forward(self.model, inputs)
            return self.model(inputs)

    class LossNet(paddle.nn.Layer):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inputs, labels):
            return case.compute_loss(self.model, inputs, labels)

    def compile_net(net):
        if mode != "cinn":
            return net
        build_strategy = paddle.static.BuildStrategy()
        build_strategy.build_cinn_pass = True
        return paddle.jit.to_static(net, build_strategy=build_strategy, full_graph=True)

    paddle.seed(SEED)
    np.random.seed(SEED)
    build_start = time.perf_counter()
    model = case.build_model()
    dropout_layers = disable_dropout(model)
    init_path = work_dir / "init.pdparams"
    batch_path = work_dir / "batch.npz"
    if mode == "dynamic" and not reuse_inputs:
        inputs, labels = case.build_batch()
        paddle.save(model.state_dict(), str(init_path))
        save_batch(batch_path, inputs, labels)
    else:
        model.set_state_dict(paddle.load(str(init_path)))
        inputs, labels = load_batch(batch_path)
    if hasattr(case, "prepare"):
        case.prepare(model)  # e.g. lambdify sympy equations, outside of to_static
    result = {
        "case": case_name,
        "mode": mode,
        "batch_size": case.BATCH_SIZE,
        "num_params": int(sum(int(np.prod(p.shape)) for p in model.parameters())),
        "dropout_layers_disabled": dropout_layers,
        "build_seconds": time.perf_counter() - build_start,
    }
    total_iters = iters + warmup

    if "infer" in stages:
        model.eval()
        net = compile_net(InferNet(model))
        last = {}

        def infer_step():
            with paddle.no_grad():
                last["out"] = net(inputs)

        times, _ = timed_loop(infer_step, total_iters, mode, "infer")
        flat = flatten(last.pop("out"))
        np.savez(
            work_dir / f"infer_{mode}.npz",
            **{k: v.astype("float32").numpy() for k, v in flat.items()},
        )
        result["infer"] = time_stats(times, warmup)

    if "train" in stages:
        model.set_state_dict(paddle.load(str(init_path)))
        model.train()
        net = compile_net(LossNet(model))
        optimizer = paddle.optimizer.Adam(
            getattr(case, "LEARNING_RATE", 1e-3), parameters=model.parameters()
        )
        grad_inputs = [t for t in inputs.values() if not t.stop_gradient]

        def train_step():
            loss = net(inputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            for t in grad_inputs:
                t.clear_gradient()
            return float(loss)

        times, losses = timed_loop(train_step, total_iters, mode, "train")
        result["train"] = {**time_stats(times, warmup), "losses": losses}

    print(RESULT_PREFIX + json.dumps(result), flush=True)


# ---------------------------------------------------------------------- parent
def make_env(mode, device):
    env = os.environ.copy()
    for key in COMPILER_ENV_KEYS:
        env.pop(key, None)
    # NVIDIA / Iluvatar read CUDA_VISIBLE_DEVICES, Hygon DCU reads HIP_VISIBLE_DEVICES
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    env["HIP_VISIBLE_DEVICES"] = str(device)
    env["PYTHONUNBUFFERED"] = "1"
    if mode == "cinn":
        env.update(CINN_ENV)
    return env


def run_worker(case_name, mode, args, device, work_dir):
    command = [
        args.python,
        str(Path(__file__).resolve()),
        "--worker",
        mode,
        "--cases",
        case_name,
        "--stages",
        ",".join(args.stages),
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--work-dir",
        str(work_dir),
    ]
    if args.reuse_inputs:
        command.append("--reuse-inputs")
    log_path = work_dir / f"{mode}.log"
    result = None
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(" ".join(command) + "\n")
        log_file.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=make_env(mode, device),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
            )
            return_code = completed.returncode
        except subprocess.TimeoutExpired:
            return_code = "timeout"
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(RESULT_PREFIX):
            result = json.loads(line[len(RESULT_PREFIX) :])
    if return_code != 0 or result is None:
        return None, f"exit={return_code}: {last_error_line(log_path)}"
    result["wall_seconds"] = time.perf_counter() - started
    return result, None


def last_error_line(log_path):
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        if "Error" in line or "error" in line:
            return line.strip()[:300]
    return lines[-1].strip()[:300] if lines else "no output"


def rel_diff(a, b):
    return abs(a - b) / max(abs(a), 1e-12)


def compare_infer(work_dir):
    with np.load(work_dir / "infer_dynamic.npz") as ref, np.load(
        work_dir / "infer_cinn.npz"
    ) as out:
        if set(ref.files) != set(out.files):
            return {"error": f"output keys differ: {ref.files} vs {out.files}"}
        worst_abs, worst_rel = 0.0, 0.0
        for key in ref.files:
            a, b = ref[key].astype("float64"), out[key].astype("float64")
            if a.shape != b.shape:
                return {"error": f"{key} shape {a.shape} vs {b.shape}"}
            if not np.isfinite(b).all() and np.isfinite(a).all():
                return {"error": f"{key} has non-finite values under CINN"}
            diff = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
            scale = float(np.nanmax(np.abs(a))) if a.size else 0.0
            worst_abs = max(worst_abs, diff)
            worst_rel = max(worst_rel, diff / max(scale, 1e-12))
    return {"max_abs_diff": worst_abs, "max_rel_diff": worst_rel}


def compare_case(dynamic, cinn, work_dir, args):
    """Accuracy and speed verdict per stage."""
    report = {}
    if "infer" in dynamic:
        acc = compare_infer(work_dir)
        acc["pass"] = "error" not in acc and acc["max_rel_diff"] <= args.infer_rtol
        report["infer"] = acc
    if "train" in dynamic:
        ref, out = dynamic["train"]["losses"], cinn["train"]["losses"]
        diffs = [rel_diff(a, b) for a, b in zip(ref, out)]
        finite = all(np.isfinite(out)) or not all(np.isfinite(ref))
        report["train"] = {
            "first_loss_rel_diff": diffs[0],
            "final_loss_rel_diff": diffs[-1],
            "max_loss_rel_diff": max(diffs),
            "pass": bool(
                finite
                and diffs[0] <= args.train_first_rtol
                and diffs[-1] <= args.train_final_rtol
            ),
        }
    for stage in report:
        speedup = (
            dynamic[stage]["steady_mean_seconds"] / cinn[stage]["steady_mean_seconds"]
        )
        report[stage].update(
            {
                "dynamic_seconds": dynamic[stage]["steady_mean_seconds"],
                "cinn_seconds": cinn[stage]["steady_mean_seconds"],
                "cinn_first_iter_seconds": cinn[stage]["first_iter_seconds"],
                "speedup": speedup,
            }
        )
    return report


def run_case(case_name, args, devices, out_dir):
    device = devices.get()
    try:
        work_dir = out_dir / case_name
        work_dir.mkdir(parents=True, exist_ok=True)
        summary = {"case": case_name, "device": device}
        dynamic, error = run_worker(case_name, "dynamic", args, device, work_dir)
        if error:
            summary.update(status="CASE_ERROR", error=error)
            return summary
        cinn, error = run_worker(case_name, "cinn", args, device, work_dir)
        if error:
            summary.update(status="CINN_ERROR", error=error)
            return summary
        summary.update(compare_case(dynamic, cinn, work_dir, args))
        stages = [summary[s] for s in args.stages]
        if not all(s["pass"] for s in stages):
            summary["status"] = "ACC_FAIL"
        elif all(s["speedup"] >= 1.0 for s in stages):
            summary["status"] = "PASS"
        else:
            summary["status"] = "SLOWER"
        summary["num_params"] = dynamic["num_params"]
        return summary
    finally:
        devices.put(device)


def fmt_stage(summary, stage):
    s = summary.get(stage)
    if not s:
        return ["-", "-", "-"]
    if stage == "infer":
        acc = s.get("error") or f"{s['max_rel_diff']:.1e}"
    else:
        acc = f"{s['first_loss_rel_diff']:.1e}/{s['final_loss_rel_diff']:.1e}"
    return [
        acc,
        f"{s['dynamic_seconds'] * 1e3:.2f}/{s['cinn_seconds'] * 1e3:.2f}",
        f"{s['speedup']:.2f}x",
    ]


def write_summary(summaries, out_dir, args):
    header = ["case", "status"]
    for stage in args.stages:
        acc = "infer rel diff" if stage == "infer" else "train loss diff first/final"
        header += [acc, f"{stage} ms dyn/cinn", f"{stage} speedup"]
    rows = []
    for s in sorted(summaries, key=lambda x: x["case"]):
        row = [s["case"], s["status"]]
        for stage in args.stages:
            row += fmt_stage(s, stage)
        rows.append(row)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    errors = [s for s in summaries if "error" in s]
    if errors:
        lines += ["", "Errors:"]
        lines += [f"- {s['case']} ({s['status']}): {s['error']}" for s in errors]
    counts = {}
    for s in summaries:
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    lines += ["", "Totals: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))]
    text = "\n".join(lines)
    (out_dir / "summary.md").write_text(text + "\n", encoding="utf-8")
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "cases": summaries}, f, indent=2)
    print("\n" + text)
    print(f"\nresults: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cases", default="all", help="comma separated, or 'all'")
    parser.add_argument("--list", action="store_true", help="list cases and exit")
    parser.add_argument("--stages", default="infer,train", help="infer,train")
    parser.add_argument("--iters", type=int, default=20, help="timed iters per stage")
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="leading iters excluded from timing (compilation lands here)",
    )
    parser.add_argument("--devices", default="0", help="GPU ids, one case per GPU")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per worker")
    parser.add_argument("--infer-rtol", type=float, default=1e-3)
    parser.add_argument("--train-first-rtol", type=float, default=1e-3)
    parser.add_argument("--train-final-rtol", type=float, default=1e-2)
    parser.add_argument("--output-dir", help="defaults to test/cinn/outputs/<stamp>")
    parser.add_argument(
        "--reuse-inputs",
        action="store_true",
        help="load <output-dir>/<case>/init.pdparams + batch.npz instead of regenerating",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--worker", choices=["dynamic", "cinn"], help=argparse.SUPPRESS)
    parser.add_argument("--work-dir", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.stages = [s for s in args.stages.split(",") if s]
    if not set(args.stages) <= set(STAGES):
        parser.error(f"--stages must be a subset of {STAGES}")
    return args


def main():
    args = parse_args()
    if args.worker:
        worker(
            args.cases,
            args.worker,
            args.stages,
            args.iters,
            args.warmup,
            Path(args.work_dir),
            args.reuse_inputs,
        )
        return
    if args.list:
        print("\n".join(list_cases()))
        return
    if args.warmup < 1 or args.iters < 2:
        raise ValueError("need warmup >= 1 and iters >= 2")

    available = list_cases()
    cases = available if args.cases == "all" else args.cases.split(",")
    unknown = sorted(set(cases) - set(available))
    if unknown:
        raise ValueError(f"unknown cases: {unknown}")
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (TEST_DIR / "outputs" / time.strftime("%Y%m%d_%H%M%S"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    devices = queue.Queue()
    device_ids = [d for d in args.devices.split(",") if d]
    for d in device_ids:
        devices.put(d)
    summaries = []
    with concurrent.futures.ThreadPoolExecutor(len(device_ids)) as pool:
        futures = {pool.submit(run_case, c, args, devices, out_dir): c for c in cases}
        for future in concurrent.futures.as_completed(futures):
            s = future.result()
            summaries.append(s)
            print(
                f"[{len(summaries)}/{len(cases)}] {s['case']}: {s['status']}",
                flush=True,
            )
    write_summary(summaries, out_dir, args)


if __name__ == "__main__":
    main()
