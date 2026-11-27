# U-Cast migration validation

This follow-up to PR #1273 uses the unmodified `ppsci.solver.Solver.train()`,
`Solver.eval()` and `train_epoch_func`, with supervised constraints/validators
and official callbacks. The synchronized model's custom attention backward,
convolution rounding, activation recomputation and ERA5 cache are retained.
No shared Solver or checkpoint utility code is modified by this follow-up.

## Validated behavior

Validation used Paddle GPU 3.3.1, CUDA 11.8, Python 3.10 and A100 80GB.

- 14 regression cases pass with FP32 training fixtures, including both native
  mid-epoch resume cases, epoch-boundary resume with public Hydra names and
  struct mode, legacy format-1 resume, EMA, both training stages, recomputation,
  deep ensembles, metrics, NetCDF readback and atomic AMP overflow handling.
- Five GPU AMP scenarios pass: both mid-epoch resume cases, epoch-boundary
  resume, probabilistic training/deep-ensemble inference and legacy resume.
- Five independent Hydra CLI runs pass: deterministic training, native resume,
  probabilistic fine-tuning, `Solver.eval()` and inference. These use the public
  `TRAIN.update_freq`, scheduler, stage and time-slice options.
- Full-size (895,389,203 parameters) synthetic training completes
  one optimizer update and one EMA update in each stage through the official
  Solver epoch function. All 518 parameter tensors have gradients and gradient
  norms are finite. WMAE loss: 1.070233583; two-member fair-CRPS
  loss: 1.028013349. Checkpoint I/O is covered by the smaller
  regression/CLI runs rather than writing full-size synthetic test checkpoints.
- Real WeatherBench-2 ERA5 evaluation and inference complete for the initial
  condition 2020-01-01 00:00, one member and one 12-hour step. All 83 forecast
  variables are finite and the exported NetCDF reopens with the correct lead
  time and dimensions. This is a functional smoke check, not a skill benchmark.

## PyTorch/Paddle numerical comparison

The independent reference executes the original local U-Cast inference and
optimizer source. The full model uses the released EMA weights converted to
Paddle. FP32 comparisons disable TF32 with `NVIDIA_TF32_OVERRIDE=0`.

| Comparison | Maximum absolute error | Relative L2 error |
| --- | ---: | ---: |
| Small network, nonzero random weights | 6.0796738e-06 | 2.5403265e-06 |
| Full released model, 240 x 121 | 3.4540892e-05 | 4.0156247e-06 |

The original Muon plus PyTorch AdamW reference and Paddle hybrid optimizer
also match after three prescribed gradient updates: maximum Muon parameter
error 7.4505806e-09; maximum AdamW parameter error
2.3283064e-10.

## Resume precision and limits

Checkpoints restore model/optimizer, adaptive EMA, AMP scaler, random-number
state and the epoch's next batch position. FP32 resume assertions use absolute
and relative tolerance `2e-6`. GPU AMP/BF16 assertions use `1e-3`; low-precision
reductions do not guarantee bitwise reproducibility. In a six-update diagnostic,
even two uninterrupted repetitions with identical seeds differed by up to
`5.69e-4` in weights (`5.38e-4` in EMA); the examined resumed run differed by
`4.97e-5`, below that measured repetition drift. The model implementation remains identical to the synchronized PR head.

Run the checked-in regressions from the repository root:

```bash
FLAGS_cudnn_deterministic=1 python -m pytest -q \
  test/arch/test_ucast.py test/arch/test_ucast_training.py
UCAST_TEST_DEVICE=gpu UCAST_TEST_AMP=1 FLAGS_cudnn_deterministic=1 \
  python -m pytest -q test/arch/test_ucast_training.py \
  -k 'training_resume or stage2 or legacy or pr_config'
```

Full-length training and paper-level forecast-skill reproduction have not been
run. Multi-GPU execution has not been measured. The old PR `.pdtrain` optimizer
layout is not a native resume checkpoint; its weights can warm-start training.
