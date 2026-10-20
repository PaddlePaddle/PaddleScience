# Aurora

Native Paddle implementation of [Microsoft Aurora](https://github.com/microsoft/aurora),
pinned to `85dd3880683f0ac19aa8fe4386c5dcacfec99892`.
`AuroraArch` adapts the model to PaddleScience's dictionary and `Solver` interfaces.
The upstream MIT notice is preserved in `ppsci/arch/aurora/LICENSE.txt`.

## Environment

Run commands from the PaddleScience repository root. The tested environment is
Python 3.10, PaddlePaddle GPU 3.4.0, CUDA 12.9 and RTX 3090 (24 GB).
Install PaddlePaddle using its official instructions, then:

```sh
python -m pip install -e .
python -m pip install -r examples/aurora/requirements.txt
```

NumPy, SciPy and einops are core dependencies. NetCDF and download dependencies
are example-only. Native `.pdparams` inference and training do not import Torch.
Other Python/Paddle combinations require validation; the project-wide minimum
Python version is not raised by this example.

## Offline workflow check

```sh
python examples/aurora/prepare_demo.py
python examples/aurora/finetune.py --config-name tiny
python examples/aurora/finetune.py --config-name tiny mode=eval \
  MODEL.checkpoint=outputs/aurora-tiny/model.pdparams
python examples/aurora/predict.py --tiny --steps 2 --output outputs/aurora-demo
```

The demo uses synthetic physical-scale fields and a reduced architecture. It
checks execution, gradients, saving and reloading; it does not measure forecast
skill. Tiny weights are incompatible with published full-model weights.

## Official weights

The full `AuroraPretrained` model has 1,256,300,176 parameters. Download the pinned
official checkpoint (about 4.68 GiB) and verify its LFS SHA256:

```sh
python examples/aurora/download_checkpoint.py --directory checkpoints
```

Export in a separate Torch reference environment; the export script does not
import Paddle. Use its Python executable in place of `TORCH_PYTHON`:

```sh
TORCH_PYTHON examples/aurora/export_torch_checkpoint.py \
  checkpoints/aurora-0.25-pretrained.ckpt checkpoints/aurora-torch.npz
python examples/aurora/convert_checkpoint.py --model AuroraPretrained \
  --input checkpoints/aurora-torch.npz --output checkpoints/aurora-pretrained.pdparams
```

Official SHA256:
`64e6c62a7043498292a6462e8c3a08171300bc166cba6b2b557021c4b684a4c9`.
Only native Linear matrices are transposed; missing keys, unexpected keys and
shape mismatches fail conversion. The source checkpoint is not overwritten.
No hosted native-weight URL is claimed; conversion above is the distribution path.

## ERA5 input and truth

Download the 0.25-degree, 13-pressure-level WeatherBench2 ERA5 subset. The source
contains data through 2023-01-10 18:00 UTC. `--start` is interpreted as UTC;
two history times and `--steps` future times are downloaded. Keep enough disk
space for compressed source chunks, raw NetCDF and prepared batches.

```sh
python examples/aurora/download_era5.py --start 2023-01-01 --steps 2 \
  --directory data/aurora
python examples/aurora/prepare_era5.py --directory data/aurora
```

The downloader records object ETags, local SHA256 and output manifests, and
reuses verified cached chunks. It uses proxy environment settings by default;
`--direct` selects direct HTTPS, and `--request-bytes 4194304` enables bounded
range responses when large object transfers are unreliable.

Required fields: surface `2t, 10u, 10v, msl`; atmosphere `z, u, v, t, q`;
static `lsm, z, slt`. Standard pressure levels are
`50,100,150,200,250,300,400,500,600,700,850,925,1000` hPa.
Surface arrays are `(B,T,H,W)`, atmosphere arrays `(B,T,C,H,W)`, static arrays
`(H,W)`; `T=2` for inputs and `T=1` for targets. Latitude decreases, longitude is
in `[0,360)`, and metadata contains the final history time in UTC. Forecasts crop
an incomplete latitude patch, commonly `721 -> 720`; evaluation aligns coordinates.

## Full-model inference

```sh
CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 python examples/aurora/run_era5.py \
  --framework paddle --device gpu:0 --steps 2 \
  --checkpoint checkpoints/aurora-pretrained.pdparams \
  --input data/aurora/input.nc --output outputs/aurora-era5/paddle
```

FP32 inference chunks independent attention windows and decoder grid positions
without changing variables, levels or resolution. The decoder chunk is 2048
locations by default. The measured baseline peak allocation was 17.83 GiB;
training requires substantially more memory and has not been validated globally
at full resolution. `run_era5.py` requires a GPU; `predict.py --tiny` supports CPU.

## Supervised Solver example

```sh
CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 python examples/aurora/finetune.py \
  device=gpu:0 MODEL.checkpoint=checkpoints/aurora-pretrained.pdparams
python examples/aurora/finetune.py mode=eval \
  MODEL.checkpoint=outputs/aurora-finetune/model.pdparams
```

`conf/aurora.yaml` connects `AuroraArch`, `NamedArrayDataset`,
`SupervisedConstraint`, Adam, `SupervisedValidator` and `Solver`. It averages
per-variable MSE after division by official variable/pressure-level scales.
Default `DATA.spatial_stride=45` retains all levels on a bounded global grid.
The example uses **one forecast pair**, checks target time/levels/grid, and
evaluates that same pair. It demonstrates fine-tuning integration; repeated
updates and same-pair evaluation do not establish convergence or held-out skill.
`model.pdparams` contains native underlying-model weights for subsequent inference;
Solver also saves its own training checkpoints.

## Migration validation

[Portable acceptance results](validation/acceptance.json) record the frozen
project criteria. They are not universal PaddleScience thresholds.

| Check | Measured maximum | Project threshold |
| --- | --- | --- |
| Same-input normalized absolute error | 4.21e-5 | 1e-4 |
| Same-input variable relative L2 | 7.88e-7 | 1e-5 |
| Three-update full-gradient relative L2 | 2.31e-5 | 1e-4 |
| Updated-parameter relative L2 | 9.35e-10 | 1e-5 |
| 4,554 weather RMSE relative differences | 0.000943% | 1% per row |

The training numerical check uses the full pretrained architecture on a 17x32
spatial sampling and all 13 levels. Forecast checks use three seasonal dates up
to 240h plus an existing two-step case. They do not reproduce the paper's full
training or a statistical benchmark. ACC was not computed without climatology.
Independent long-rollout pointwise checks retain `rtol=atol=2e-4` and **fail** at
some leads; weather metric success does not turn those checks into passes.
Other variants (12h, HighRes, AirPollution, Wave, 1.5 and ensemble) are ported but
have only lightweight structure checks, not full published-checkpoint acceptance.
The experimental FP32 memory-efficient backend is excluded from this submission.

The pinned upstream Small regression fixture can be prepared separately. Its
target is an official model output, not ERA5 ground truth:

```sh
python examples/aurora/download_official_sample.py --directory data/aurora-official
python examples/aurora/prepare_official_sample.py --directory data/aurora-official
```

```sh
python -m pytest test/arch/test_aurora.py -q
CUDA_VISIBLE_DEVICES=0 AURORA_TEST_DEVICE=gpu:0 NVIDIA_TF32_OVERRIDE=0 \
  python -m pytest test/arch/test_aurora.py -q
```

`check_alignment.py` and `check_training_alignment.py` run frameworks in separate
processes. Pin the Torch source to the commit above, expose it via `PYTHONPATH`,
and use the same GPU with TF32 disabled. See each script's `--help`.
For ERA5 metrics, first generate Torch forecasts with `run_era5.py --framework torch`
in that environment, then:

```sh
python examples/aurora/evaluate_era5.py --data data/aurora \
  --torch outputs/aurora-era5/torch --paddle outputs/aurora-era5/paddle \
  --steps 2 --output outputs/aurora-era5/metrics --enforce-acceptance
```

The native implementation does not enable global `paddle.enable_compat()`.
PaConvert initially handled 751/758 API occurrences (99.08% syntax coverage);
manual changes cover layouts, attention, gradients, recomputation, RNG, pooling,
UTC time handling and checkpoint conversion. This percentage is not model accuracy.
