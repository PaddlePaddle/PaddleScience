# Mosaic global probabilistic weather forecasting

[Mosaic](https://github.com/maxxxzdn/mosaic) combines a HEALPix U-Net with local,
compressed and selected-block attention. The Paddle port is registered as
`ppsci.arch.Mosaic` and includes GQA, RoPE, interpolation and noise conditioning.

## Setup and inference

Install PaddleScience and a CUDA-compatible Paddle build, then install the
example dependencies from the repository root:

```bash
python -m pip install -r examples/mosaic/requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
```

Use the official [ERA5 assets](https://huggingface.co/AMLab-UvA/mosaic) and extract
real contexts with `prepare_data.py`. Inputs contain two history frames and 82
variables on a 240-by-121 grid; the forecast lead is 24h. Outputs are normalized.
Follow the [reproduction guide](https://github.com/PaddlePaddle/PaddleScience/blob/develop/examples/mosaic/VALIDATION.md)
to verify and convert the checkpoint, including original geometry buffers.

```bash
python examples/mosaic/infer.py --model /path/to/paddle-era5-source-geometry \
  --static /path/to/static_vars.npz --stats /path/to/norm_stats_era5.npz \
  --data /path/to/era5_three_samples.npz --device gpu:0 \
  --output /path/to/predictions
```

The [example README](https://github.com/PaddlePaddle/PaddleScience/blob/develop/examples/mosaic/README.md)
also documents LoRA, weighted MSE/MAE/fair-CRPS, ensemble checks and rollouts.
Training is currently a fixed-sample diagnostic, not a production trainer.
Full-grid LoRA is checked; full-parameter backward has not passed on the 24GB GPU.

## Validation scope

All 131 parameter groups are converted (214,184,448 base parameters).
Worst per-case/per-variable weather RMSE regression is 0.5462% on three winter
24h forecasts. Single spring, summer and autumn cases have worst regression
1.5526%, 1.0420% and 0.0179%. The reference uses native Triton sparse attention
and Torch SDPA, not the complete original FlashAttention setup.

Full-model CINN on RTX3090 improves throughput by about 23%; the worst weather
RMSE regression on three winter inputs is 0.001085%. The strict absolute
pointwise check of 1e-4 still fails. That tolerance and the 1% weather screening
line are internal diagnostics, not confirmed client criteria. These limited
checks do not establish full-year reproduction, calibration or pointwise identity.

Run the CPU unit suite with:

```bash
PYTHONPATH=. python -m unittest discover -s test/arch -p 'test_mosaic*.py' -v
```
