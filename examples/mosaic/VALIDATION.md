# Mosaic validation and reproduction

This contribution ports the architecture and official checkpoint and provides
inference, LoRA training diagnostics and evaluation tools. It does not claim
full-year paper reproduction, a production trainer or strict pointwise identity
with the original BF16 kernel. Results below are measured, not CI promises.

## Source and environment

- Original code: [maxxxzdn/mosaic](https://github.com/maxxxzdn/mosaic), Apache-2.0,
  commit `30c5baef1dc76c5810caf84c449e46c7925b9aec`.
- Official assets: [AMLab-UvA/mosaic](https://huggingface.co/AMLab-UvA/mosaic/tree/c2aaa3e0799e2a0e8fba8c79beb6d3ba8e2a4954),
  revision `c2aaa3e0799e2a0e8fba8c79beb6d3ba8e2a4954`.
- ERA5 checkpoint SHA256:
  `187df31a0caef3e934e4eb8a11506c9fea518ff0e02ce6d1f804fc6c8a78a940`.
- Data: public WeatherBench2 ERA5 1.5-degree store, specified in
  `prepare_data.py`; 82 variables, 240 longitude by 121 latitude points,
  two input frames separated by 24h and a 24h forecast target.
- Full-model experiments: NVIDIA RTX 3090, 24GB, FP32, explicit seed 2026 noise.
  The original sparse kernel internally quantizes intermediate values to BF16.

| Package | Paddle environment | Torch reference environment |
| --- | --- | --- |
| Python | 3.12.13 | 3.10.20 |
| Framework | Paddle GPU 3.4.0.post20260812+a136cd3594a | Torch 2.11.0 |
| NumPy | 1.26.4 | 2.2.6 |
| einops | 0.8.2 | 0.8.2 |
| healpy | 1.19.0 | 1.20.1 |
| scikit-learn | 1.4.2 | 1.7.2 |
| Triton | not required | 3.6.0 |

Data extraction was run with zarr 2.18.7, fsspec 2026.2.0 and pandas 3.0.3.
This is a tested environment record, not a claim of compatibility with all
Paddle releases. CINN was validated on the recorded nightly build.
Install the Paddle build appropriate for the machine using the official
installation instructions, then the repository and example requirements.

## Results

Weather regression is `(RMSE_paddle / RMSE_reference - 1) * 100`, using
cosine-latitude weights in physical units. Positive values mean larger error.
Aggregation averages MSE before taking the square root. The reference uses
original Triton sparse attention and Torch SDPA for local/compressed attention;
it is not the complete original FlashAttention execution path.

| Check | Coverage | Measured result |
| --- | --- | --- |
| Unit tests | CPU, 11 tests | Passed |
| Mathematical alignment | Tiny model, 3 SGD updates, CPU/GPU | Output max <2e-7; parameter gradient max <9e-8 |
| Official conversion | Full ERA5 model | 131/131 parameter groups; 214,184,448 parameters; 65 original geometry buffers retained |
| Winter forecasts | Jan 2/3/4, 2020 initializations; 24h; 82 variables | Worst case/variable RMSE regression 0.546171%; worst aggregate 0.134837% |
| Autoregressive rollout | Jan 2, 2020 initialization; no target feedback | Worst variable regression at 24/48/72h: 0.439738% / 0.893076% / 0.793012% |
| Spring forecast | Apr 2, 2020; 24h | Worst 1.552577%, geopotential_100 |
| Summer forecast | Jul 2, 2020; 24h | Worst 1.042044%, vertical_velocity_400 |
| Autumn forecast | Oct 2, 2020; 24h | Worst 0.017902% |
| LoRA training | Full grid; rank 8/alpha 16; 1,837,056 trainable parameters | Finite gradients, unchanged frozen base and exact adapter save/load roundtrip |
| Ensemble training | Fixed winter sample 1; 2 members; 10 SGD steps, lr 0.001 | Fair-CRPS 0.081527159 to 0.081524864; peak allocated 19.6GiB |
| CINN | Full model; same graph on 3 winter inputs; timing on sample 0 | Dynamic 1.465533s, compiled 1.191436s; throughput +23.0056%; first compile/run 226.332s |

CINN maximum normalized pointwise differences are
0.00216770 / 0.00197840 / 0.00360155. Its worst per-case/per-variable weather
RMSE regression against Paddle dynamic inference is 0.00108478%.
Timing uses warmup 3 and 10 synchronized measured iterations; this does not
measure all inputs, ensemble inference or autoregressive compiler performance.

The internal 1% weather screening flags the spring and summer cases, and the
internal absolute output check `atol=1e-4` flags full-model CINN. These thresholds
were diagnostic choices, not confirmed client acceptance criteria. Reports and
nonzero exits remain unchanged. Upstream maintainers determine whether further
validation is needed for merging the model port.

All 82 variables outperform persistence on the three seasonal dates. At 72h in
the single winter rollout, temperature_50 is worse than persistence for both
Paddle and the original reference. Fixed-sample LoRA updates and adjacent-date
checks do not establish convergence, calibration or statistical improvement.
Full-parameter full-resolution FP32 backward has not passed on the 24GB card.

## Reproduce

Commands run from the repository root. Replace `python-paddle`, `python-torch`
and `/path/to/...` with separate environment executables and local paths.
See [README](README.md) for alignment, LoRA, ensemble, rollout and seasonal commands.
No checkpoint or weather binary is committed to this repository.

1. Download the three files at the pinned asset revision:
   `checkpoints/era5_best.pt`, `data/static_vars.npz` and
   `data/norm_stats_era5.npz`. Export and convert the checkpoint using the
   SHA256-checked commands in README, producing `config.json` and `model.pdparams`.
2. Extract three real contexts:

```bash
python-paddle examples/mosaic/prepare_data.py --samples 3 \
  --init-time 2020-01-02T00:00 --output /path/to/era5_three_samples.npz
```

3. For the geometry-preserving candidate used in the results, export the
   original buffers using the pinned original source and the converted config:

```bash
python-torch examples/mosaic/reference_forecast.py \
  --source /path/to/original-mosaic --config /path/to/paddle-era5/config.json \
  --checkpoint /path/to/era5_best.pt --static /path/to/static_vars.npz \
  --stats /path/to/norm_stats_era5.npz --data /path/to/era5_three_samples.npz \
  --output /path/to/native_reference --geometry-output /path/to/source_geometry.npz \
  --device cuda:0 --sparse-backend native --seed 2026

python-paddle examples/mosaic/convert_checkpoint.py --stage convert \
  --input /path/to/era5_weights.npz --variant era5 \
  --static /path/to/static_vars.npz --geometry /path/to/source_geometry.npz \
  --output /path/to/paddle-era5-source-geometry
```

4. Run Paddle inference using that geometry-preserving checkpoint, then the
   evaluation commands in README. Do not enable reference route overrides.
   Run the CINN diagnostic with the same model and data:

```bash
python-paddle examples/mosaic/benchmark.py \
  --model /path/to/paddle-era5-source-geometry \
  --static /path/to/static_vars.npz --stats /path/to/norm_stats_era5.npz \
  --data /path/to/era5_three_samples.npz --samples 3 --device gpu:0 \
  --warmup 3 --steps 10 --output /path/to/cinn.json
```

The last command saves JSON and full dynamic/compiled outputs, then exits
nonzero if the internal strict output tolerance is exceeded, even when weather
RMSE differences are small. Keep the report and exit status when reproducing.

5. Run the lightweight regression suite:

```bash
PYTHONPATH=. python-paddle -m unittest discover -s test/arch -p 'test_mosaic*.py' -v
```
