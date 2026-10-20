# Aurora

## 1. Background and scope

This example ports [Microsoft Aurora](https://github.com/microsoft/aurora) to
native Paddle and integrates it with PaddleScience through `AuroraArch` and `Solver`.
The reference is pinned to `85dd3880683f0ac19aa8fe4386c5dcacfec99892`;
the upstream MIT license is preserved.

Variable patch embeddings and a Perceiver encoder tokenize weather fields, a
3D Swin U-Net processes spatial and vertical interactions, and a Perceiver
decoder produces requested variables and pressure levels. Autoregressive
forecasts feed predictions back into the following input history.

Full-checkpoint acceptance focuses on **AuroraPretrained**, with
**1,256,300,176 parameters**. Small supports debugging and alignment. The 12h,
HighRes, AirPollution, Wave, 1.5 and ensemble variants have lightweight structure
checks only; their published full checkpoints have not been accepted.

## 2. Installation and offline workflow

The validated environment is Python 3.10, PaddlePaddle GPU 3.4.0, CUDA 12.9 and
RTX 3090. Install PaddlePaddle following its official instructions, then run
from the PaddleScience repository root:

```sh
python -m pip install -e .
python -m pip install -r examples/aurora/requirements.txt
python examples/aurora/prepare_demo.py
python examples/aurora/finetune.py --config-name tiny
python examples/aurora/finetune.py --config-name tiny mode=eval \
  MODEL.checkpoint=outputs/aurora-tiny/model.pdparams
```

This offline demonstration uses synthetic fields and a reduced architecture to
exercise training, evaluation and checkpoint reload. Tiny checkpoints are
incompatible with full published architectures. The demonstration does not
measure real forecast skill. Native `.pdparams` inference and training do not
import Torch; the one-time checkpoint export runs in a separate Torch environment.

## 3. Data and checkpoint preparation

The example uses a public WeatherBench2 ERA5 subset at 0.25 degrees with 13
pressure levels. The source ends at 2023-01-10 18:00 UTC. Each download includes
two historical frames followed by `--steps` six-hour target frames.

```sh
python examples/aurora/download_era5.py --start 2023-01-01 --steps 2 \
  --directory data/aurora
python examples/aurora/prepare_era5.py --directory data/aurora
python examples/aurora/download_checkpoint.py --directory checkpoints
```

The downloader records local SHA256, source ETags and output manifests, reusing
verified chunks. Reserve disk space for source chunks, raw NetCDF, prepared
batches and checkpoint conversion; the original checkpoint is about 4.68 GiB.
Replace `TORCH_PYTHON` with the separate Torch environment's Python executable:

```sh
TORCH_PYTHON examples/aurora/export_torch_checkpoint.py \
  checkpoints/aurora-0.25-pretrained.ckpt checkpoints/aurora-torch.npz
python examples/aurora/convert_checkpoint.py --input checkpoints/aurora-torch.npz \
  --output checkpoints/aurora-pretrained.pdparams
```

Surface variables are `2t, 10u, 10v, msl`, atmospheric variables are `z, u, v, t, q`,
and static variables are `lsm, z, slt`. Surface shape is `(B,T,H,W)`, atmospheric
shape is `(B,T,C,H,W)`, and static shape is `(H,W)`. Inputs contain two historical
frames and targets contain one future frame. Pressure levels use hPa, longitude
is in `[0,360)`, latitude decreases, and metadata identifies the last history time
in UTC. Outputs crop incomplete latitude patches, commonly `721 -> 720`;
evaluation aligns coordinates and timestamps explicitly.

## 4. Inference, training and evaluation

```sh
CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 python examples/aurora/run_era5.py \
  --framework paddle --device gpu:0 --steps 2 \
  --checkpoint checkpoints/aurora-pretrained.pdparams \
  --input data/aurora/input.nc --output outputs/aurora-era5/paddle
```

FP32 inference chunks independent attention windows and decoder grid positions;
the default decoder batch is 2048 locations. Measured baseline peak allocation
was 17.83 GiB. Training needs substantially more memory and has not been validated
globally at full resolution.

```sh
CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 python examples/aurora/finetune.py \
  device=gpu:0 MODEL.checkpoint=checkpoints/aurora-pretrained.pdparams
python examples/aurora/finetune.py mode=eval \
  MODEL.checkpoint=outputs/aurora-finetune/model.pdparams
```

`conf/aurora.yaml` connects `NamedArrayDataset`, `SupervisedConstraint`, Adam,
`SupervisedValidator` and `Solver`. The loss averages per-variable MSEs divided
by the squared official variable/level scales. Default spatial stride 45 retains
all 13 levels on a bounded global grid. This example repeatedly trains and
evaluates **one forecast pair**; it does not establish convergence or held-out
skill. Native `model.pdparams` can be reused for inference. Solver also saves
its own training state.

After producing Torch and Paddle forecasts in separate environments,
`evaluate_era5.py` computes cosine-latitude-weighted RMSE independently for each
variable, pressure level and lead. See `examples/aurora/README.md` for commands.

## 5. Acceptance and limitations

| Check | Measured maximum | Project threshold |
| --- | --- | --- |
| Same-input normalized absolute error | 4.21e-5 | 1e-4 |
| Same-input variable relative L2 | 7.88e-7 | 1e-5 |
| Global full-gradient relative L2 | 2.31e-5 | 1e-4 |
| Updated-parameter relative L2 | 9.35e-10 | 1e-5 |
| 4,554 weather RMSE relative differences | 0.000943% | 1% per row |

These are frozen project migration criteria, not universal PaddleScience gates.
Training alignment uses all pretrained parameters, a 17x32 spatial sampling,
13 levels and three consecutive SGD updates. Forecast validation includes three
seasonal dates up to 240h and an existing two-step case. Independent long-rollout
pointwise checks retain `rtol=atol=2e-4` and **fail** at some leads. Weather metric
success does not change that result. The experimental FP32 fused attention backend
is excluded. ACC was not computed without climatology; full paper training and
statistical forecast-skill reproduction are not claimed. Portable results are
stored in `examples/aurora/validation/acceptance.json`.

```sh
python -m pytest test/arch/test_aurora.py -q
CUDA_VISIBLE_DEVICES=0 AURORA_TEST_DEVICE=gpu:0 NVIDIA_TF32_OVERRIDE=0 \
  python -m pytest test/arch/test_aurora.py -q
```
