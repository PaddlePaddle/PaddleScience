# Mosaic global probabilistic weather forecasting

[Mosaic](https://github.com/maxxxzdn/mosaic) combines a HEALPix U-Net with local,
compressed and selected-block attention. The Paddle implementation is registered
as `ppsci.arch.Mosaic` and supports GQA, RoPE, interpolation and noise conditioning.

## Setup and data

Install PaddleScience and a CUDA-compatible Paddle build, then run:

```bash
python -m pip install -r examples/mosaic/requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
```

The ERA5 model uses two history frames with 82 variables on a 240-by-121 grid
and predicts the next 24 hours. The Paddle model directory contains
`config.json` and `model.pdparams`, including the geometry buffers.
Static fields and normalization statistics are available in the
[official asset repository](https://huggingface.co/AMLab-UvA/mosaic/tree/main/data).
The data preparation script extracts selected dates from public WeatherBench2 ERA5:

```bash
python examples/mosaic/prepare_data.py --samples 3 \
  --init-time 2020-01-02T00:00 --output data/mosaic/era5_samples.npz
```

## Inference

With the Paddle weights in `data/mosaic/model/` and static fields and statistics
in `data/mosaic/`, run from the repository root:

```bash
python examples/mosaic/infer.py --model data/mosaic/model \
  --static data/mosaic/static_vars.npz --stats data/mosaic/norm_stats_era5.npz \
  --data data/mosaic/era5_samples.npz --device gpu:0 --output output/mosaic
```

The output includes physical and normalized forecasts, coordinates and per-variable
area-weighted RMSE. Add `--rollout-steps 3` for a 72-hour autoregressive forecast.

The [example README](https://github.com/PaddlePaddle/PaddleScience/blob/develop/examples/mosaic/README.md)
provides data download commands, LoRA training checks and results for selected dates.
The ERA5 base model has 214,184,448 parameters; rank-8 LoRA adds 1,837,056
trainable parameters. The training example checks updates on a fixed sample and
supports weighted MSE, MAE and fair-CRPS.

Paper: *(Sparse) Attention to the Details: Preserving Spectral Fidelity in
ML-based Weather Forecasting Models*, ICML 2026. Original code: Apache-2.0.
