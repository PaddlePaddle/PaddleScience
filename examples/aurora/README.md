# Aurora

Native Paddle implementation of [Microsoft Aurora](https://github.com/microsoft/aurora),
pinned to `85dd3880683f0ac19aa8fe4386c5dcacfec99892` and retaining the upstream MIT license.
`AuroraArch` connects variable patch embeddings, Perceiver encoding/decoding and a
3D Swin U-Net to PaddleScience's dictionary interface and `Solver`.

Full-checkpoint validation covers `AuroraPretrained` (1,256,300,176 parameters).
Other variants retain their architectures but have not been validated with their
own complete pretrained weights. Run commands from the PaddleScience repository root.

## Installation and offline demonstration

The verified environment is Python 3.10, PaddlePaddle GPU 3.4.0, CUDA 12.9 and RTX 3090.
Install PaddlePaddle using its official instructions, then:

```sh
python -m pip install -e .
python -m pip install -r examples/aurora/requirements.txt
python examples/aurora/prepare_demo.py
python examples/aurora/finetune.py --config-name tiny
python examples/aurora/finetune.py --config-name tiny mode=eval \
  MODEL.checkpoint=outputs/aurora-tiny/model.pdparams
python examples/aurora/predict.py --tiny --output outputs/aurora-demo
```

The reduced Tiny architecture and synthetic data demonstrate forward/backward
execution and checkpoint reloading. They do not measure weather forecast skill.
Tiny weights are incompatible with the complete pretrained model.

## Native weights

Use `aurora-pretrained.pdparams` (5,025,272,922 bytes, approximately 4.68 GiB).
Its SHA256 is `eb586812d66fea058522700165b2858540bb8b15144390f9255fae434cd44d81`.
The checkpoint is hosted on [AI Studio](https://aistudio.baidu.com/modelsdetail/55897).
Download the [native Paddle weights](https://git.aistudio.baidu.com/api/v1/repos/yubb/aurora-paddle-pretrained/media/aurora-pretrained.pdparams)
to `checkpoints/aurora-pretrained.pdparams` before running the commands below.
The implementation loads native Paddle parameters directly.

## ERA5 data

Download the public WeatherBench2 ERA5 subset at 0.25-degree resolution with 13
pressure levels. The source ends at 2023-01-10 18:00 UTC. `--start` is UTC;
the download includes two historical frames and `--steps` future frames, six hours apart.

```sh
python examples/aurora/download_era5.py --start 2023-01-01 --steps 2 \
  --directory data/aurora
python examples/aurora/prepare_era5.py --directory data/aurora
```

The downloader verifies cached data and records SHA256 and source ETags. Reserve
space for source chunks and prepared NetCDF files. Proxy environment settings
are honored; `--direct` selects direct HTTPS and `--request-bytes 4194304` enables
bounded range requests.

Required variables are surface `2t, 10u, 10v, msl`, atmosphere `z, u, v, t, q`,
and static `lsm, z, slt`. Pressure levels in hPa are
`50,100,150,200,250,300,400,500,600,700,850,925,1000`.
Surface shape is `(B,T,H,W)`, atmosphere `(B,T,C,H,W)`, and static `(H,W)`.
Inputs have `T=2`, targets `T=1`; latitude decreases and longitude is in `[0,360)`.
Metadata time is the last historical frame's UTC time. The model crops incomplete
latitude patches (`721 -> 720`); evaluation strictly aligns coordinates and times.

## Inference and evaluation

```sh
CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 python examples/aurora/predict.py \
  --device gpu:0 --steps 2 --checkpoint checkpoints/aurora-pretrained.pdparams \
  --input data/aurora/input.nc --output outputs/aurora-era5
python examples/aurora/evaluate_era5.py --data data/aurora \
  --prediction outputs/aurora-era5 --steps 2 --output outputs/aurora-era5/metrics
python examples/aurora/plot_era5.py --data data/aurora \
  --prediction outputs/aurora-era5 --steps 2 --output outputs/aurora-era5/figures
```

FP32 attention and decoder locations are processed in independent chunks. The
decoder batch defaults to 2048 locations. Full-grid inference previously measured
17.83 GiB peak allocated GPU memory. Full-resolution global training is not validated.

Metrics use cosine-latitude-weighted RMSE in physical units. Persistence holds the
last input state constant. Results below use initialization 2023-01-01 06:00 UTC,
720x1440 forecast points and all 13 pressure levels.

| Variable | Unit | +6h RMSE | +6h persistence | +12h RMSE | +12h persistence |
| --- | --- | --- | --- | --- | --- |
| 2 m temperature | K | 0.491621 | 2.82432 | 0.612136 | 3.25386 |
| 10 m u-wind | m/s | 0.523374 | 1.95561 | 0.636562 | 2.77033 |
| 10 m v-wind | m/s | 0.533163 | 2.31386 | 0.655754 | 3.38663 |
| Mean sea level pressure | Pa | 31.9165 | 264.613 | 38.4941 | 358.009 |
| Temperature at 850 hPa | K | 0.432024 | 1.49239 | 0.485647 | 2.07166 |
| Geopotential at 500 hPa | m²/s² | 23.9960 | 222.807 | 29.2050 | 338.332 |

[![2 m temperature: ERA5, Paddle, and forecast error](https://paddle-org.bj.bcebos.com/paddlescience/docs/aurora/temperature.png)](https://paddle-org.bj.bcebos.com/paddlescience/docs/aurora/temperature.png)

[![500 hPa geopotential height: ERA5, Paddle, and forecast error](https://paddle-org.bj.bcebos.com/paddlescience/docs/aurora/z500.png)](https://paddle-org.bj.bcebos.com/paddlescience/docs/aurora/z500.png)

Rows show +6h and +12h; columns show ERA5, Paddle and Paddle minus ERA5.
The temperature plot uses degrees Celsius. The geopotential plot divides by
9.80665 m/s² to display height in metres; the table retains native geopotential units.
Field colors span the truth's 1st–99th percentiles; error colors use its absolute
99th percentile, so extremes may saturate. This is a single initialization,
not a statistical benchmark. ACC is not reported without climatology.

## Supervised fine-tuning

```sh
CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 python examples/aurora/finetune.py \
  device=gpu:0 MODEL.checkpoint=checkpoints/aurora-pretrained.pdparams
python examples/aurora/finetune.py mode=eval \
  MODEL.checkpoint=outputs/aurora-finetune/model.pdparams
```

`conf/aurora.yaml` builds `AuroraArch` through `ppsci.arch.build_model` and connects
`NamedArrayDataset`, `SupervisedConstraint`, Adam, `SupervisedValidator` and `Solver`.
The loss averages nine per-variable MSEs divided by the squared official
variable/pressure-level scales. `DATA.spatial_stride=45` retains all pressure levels
on a bounded grid. Input/target timestamps, levels and coordinates are checked.

This example trains and evaluates the same single forecast pair, demonstrating
the workflow rather than held-out accuracy or convergence. Random initialization
requires `TRAIN.allow_random_init=true`; evaluation requires a checkpoint.
`model.pdparams` contains native model weights; Solver also saves training state.

## CINN training and performance verification

`run_cinn.sh` supports `--mode inference`, `--mode train` and `--mode benchmark`.
The default is direct training: set training mode, then run forward, backward and
SGD updates without a preceding inference pass. `--train-steps` accepts any
positive integer, including one.

Eager and CINN run in separate processes with identical seeds, native weights,
inputs, FP32 precision and SGD settings. Both set
`FLAGS_new_executor_serial_run=true` before importing Paddle to fix the PIR
executor scheduling configuration; reports record the actual flag value. The
CINN worker also enables compiler flags and wraps each Swin encoder/decoder stage with
`paddle.jit.to_static(backend="CINN", full_graph=True)`. The eager worker disables
CINN and primitive decomposition. Feature expansion, time conditioning, the model
encoder/decoder and Batch processing remain eager. Each stage is captured as a
complete graph, with supported subgraphs compiled by CINN. Attention-mask regions
and longitude connectivity depend only on grid configuration. NumPy constructs
constant masks to avoid repeated slice assignments and fills during execution.
Eager mode caches mask tensors; each static program owns its constants, avoiding
reuse of intermediate values from another program.

```sh
CUDA_VISIBLE_DEVICES=0 bash examples/aurora/run_cinn.sh --mode train --tiny \
  --input data/aurora-demo/input.nc --target data/aurora-demo/truth_001.nc \
  --train-steps 2 --output outputs/aurora-cinn-tiny
CUDA_VISIBLE_DEVICES=0 bash examples/aurora/run_cinn.sh --mode train \
  --checkpoint checkpoints/aurora-pretrained.pdparams --input data/aurora/input.nc \
  --target data/aurora/truth_001.nc --train-steps 2 --output outputs/aurora-cinn-train
CUDA_VISIBLE_DEVICES=0 bash examples/aurora/run_cinn.sh --mode inference \
  --checkpoint checkpoints/aurora-pretrained.pdparams --input data/aurora/input.nc \
  --output outputs/aurora-cinn-inference
```

Set `PYTHON` to select the interpreter. The full model defaults to
`--spatial-stride 45`, a 17x32 grid retaining all 13 pressure levels. Reduce the
stride if memory permits. Use `--backend eager` or `--backend cinn` to run only
one backend; omitting it runs and compares both.

The root `cinn.json` summarizes both backends. Subdirectories contain logs,
predictions and `result.json`; CINN also writes `execution.paddle_trace.json`.
Inference and training require actual `cinn_jit` executions and per-variable
normalized absolute error <=1e-4 and relative L2 <=1e-5. Training additionally
checks all present gradients for finiteness, actual parameter updates and
per-step loss agreement at `rtol=1e-5, atol=1e-6`. Up to 64 positions per parameter
compare first-step gradients and final parameters with sampled-vector relative
L2 <=1e-4. Sampling does not establish elementwise agreement of every gradient.

Benchmark identical resident inputs for inference or a complete training step
including forward, loss, backward, SGD and gradient clearing:

```sh
CUDA_VISIBLE_DEVICES=0 bash examples/aurora/run_cinn.sh --mode benchmark \
  --benchmark-task train --checkpoint checkpoints/aurora-pretrained.pdparams \
  --input data/aurora/input.nc --target data/aurora/truth_001.nc \
  --spatial-stride 20 --warmup 10 --iterations 50 --repeat 3 \
  --output outputs/aurora-benchmark-train
CUDA_VISIBLE_DEVICES=0 bash examples/aurora/run_cinn.sh --mode benchmark \
  --benchmark-task inference --checkpoint checkpoints/aurora-pretrained.pdparams \
  --input data/aurora/input.nc --spatial-stride 20 \
  --warmup 10 --iterations 50 --repeat 3 --output outputs/aurora-benchmark-inference
```

During training warmup, the first `--train-steps` updates, capped by the warmup
count, also check losses and gradients. One untimed diagnostic update follows
timing. The first training step includes its checks. First-step latency is
recorded separately and includes on-demand capture, compilation and execution;
compiler caches may be reused. Initial compilation can take several minutes.
Steady-state timing
follows warmup, synchronizes the GPU at step boundaries and excludes compilation,
diagnostics, profiling, NumPy copies and file I/O. Each backend uses an independent
process with `--repeat` consecutive timing rounds after warmup. Training continues
updating parameters between rounds with identical step counts in both backends.
Reports include individual timings,
mean/median/P95 latency, step throughput and peak memory allocated by the process.
Speedup is the ratio of the medians of per-repeat median latencies, eager / CINN;
a value above one indicates acceleration. Passing numerical checks does not
require a speedup above one. Use a GPU without competing compute workloads.

Sampled-grid results do not establish global full-resolution training performance
or whole-model full-graph compilation.

### Sampled-grid measurements

Measured on an exclusive RTX 3090 with PaddlePaddle 3.4.0 / CUDA 12.9, FP32 and
batch size 1. Both backends set `FLAGS_new_executor_serial_run=true`.
The complete 1,256,300,176-parameter native checkpoint uses ERA5 with all 13
pressure levels. `--spatial-stride 20` gives a 37x72 input cropped to 36x72 by the
model. Each backend warms up for 10 steps and runs three consecutive rounds of
50 steps. Training uses SGD with a learning rate of 1e-6.

| Task | Eager steady-state latency (ms/step) | CINN steady-state latency (ms/step) | Speedup | Eager / CINN peak allocated memory (GiB) |
| --- | ---: | ---: | ---: | ---: |
| Training | 186.54 | 131.96 | 1.414x | 10.38 / 11.43 |
| Inference | 76.30 | 48.03 | 1.588x | 4.74 / 4.81 |

Training executes 161 parameter updates without a preceding inference pass.
Maximum normalized absolute error is 1.99e-5 and maximum relative L2 is 2.59e-6.
First-step sampled-gradient relative L2 is 1.26e-5; final sampled-parameter
relative L2 is 7.50e-11. Inference maximum normalized absolute error is 1.52e-5
and maximum relative L2 is 3.52e-6. Training and inference diagnostic traces
record 1,113 and 466 CINN operator executions, respectively. Tiny direct
training matches both eager losses exactly: sampled-gradient relative L2 is
5.93e-8, maximum normalized absolute error is 2.84e-6 and maximum relative L2 is
9.32e-7.

The first CINN training step takes 347.36 seconds including capture, compilation,
execution and gradient checks, versus 5.23 seconds for eager. Its first inference
step takes 199.55 seconds, versus 0.57 seconds for eager. Steady-state gains use
more memory; short runs can take longer overall. These measurements apply to
this environment and sampled grid and do not guarantee acceleration at global
full resolution.

The earlier multithreaded executor configuration encountered Paddle profiler
`HostEventRecorder::GatherEvents` crashes or `vector::reserve` errors during
stop. This entry point fixes PIR scheduling to serial execution; Tiny training,
full training and full inference all pass trace export and numerical checks.
Revalidate stability and performance after changing executor settings. An
abnormal process exit is not a pass.
