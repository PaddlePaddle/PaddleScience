"""Aurora migration regressions: layouts, gradients, checkpoints, and rollout."""

import dataclasses
import inspect
import os
import time
from datetime import timedelta
from datetime import timezone

import numpy as np
import paddle
import pytest

import ppsci
from examples.aurora.sample import make_batch
from examples.aurora.sample import small_config
from ppsci.arch.aurora import Aurora
from ppsci.arch.aurora import Aurora12hPretrained
from ppsci.arch.aurora import AuroraAirPollution
from ppsci.arch.aurora import AuroraArch
from ppsci.arch.aurora import AuroraHighRes
from ppsci.arch.aurora import AuroraPretrained
from ppsci.arch.aurora import AuroraSmallPretrained
from ppsci.arch.aurora import AuroraV1p5
from ppsci.arch.aurora import AuroraV1p5Ensemble
from ppsci.arch.aurora import AuroraWave
from ppsci.arch.aurora import Batch
from ppsci.arch.aurora import Metadata
from ppsci.arch.aurora import make_generator
from ppsci.arch.aurora import rollout
from ppsci.arch.aurora import split_batch
from ppsci.arch.aurora import tile_batch
from ppsci.arch.aurora.model.posencoding import _patch_average
from ppsci.arch.aurora.model.swin3d import crop_3d
from ppsci.arch.aurora.model.swin3d import pad_3d
from ppsci.arch.aurora.model.util import scaled_dot_product_attention
from ppsci.arch.aurora.normalisation import level_to_str
from ppsci.arch.aurora.normalisation import locations
from ppsci.arch.aurora.normalisation import scales


@pytest.fixture(autouse=True)
def device():
    paddle.set_device(os.environ.get("AURORA_TEST_DEVICE", "cpu"))


def model_and_batch(**kwargs):
    model = AuroraSmallPretrained(**small_config(), **kwargs)
    return model, make_batch(paddle, Batch, Metadata)


@pytest.mark.skipif(
    not hasattr(time, "tzset"), reason="Requires process timezone support"
)
def test_meteorological_time_is_utc_independent_of_host_timezone():
    model, batch = model_and_batch()
    model.eval()
    original = os.environ.get("TZ")
    outputs = []
    try:
        for zone in ("UTC", "Asia/Shanghai"):
            os.environ["TZ"] = zone
            time.tzset()
            with paddle.no_grad():
                outputs.append(model(batch))
        aware = dataclasses.replace(
            batch,
            metadata=dataclasses.replace(
                batch.metadata,
                time=tuple(t.replace(tzinfo=timezone.utc) for t in batch.metadata.time),
            ),
        )
        with paddle.no_grad():
            outputs.append(model(aware))
    finally:
        if original is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original
        time.tzset()
    for value in outputs[1:]:
        for kind in ("surf_vars", "atmos_vars"):
            for key, expected in getattr(outputs[0], kind).items():
                np.testing.assert_array_equal(
                    getattr(value, kind)[key].numpy(), expected.numpy()
                )


def test_float64_model_rolls_out_with_float32_forecasts():
    model, batch = model_and_batch()
    model.to(dtype=paddle.float64)
    model.eval()
    with paddle.no_grad():
        predictions = list(rollout(model, batch, steps=2))
    assert len(predictions) == 2
    for pred in predictions:
        assert pred.surf_vars["2t"].dtype == paddle.float32
        assert pred.atmos_vars["t"].dtype == paddle.float32
        assert bool(paddle.isfinite(pred.surf_vars["2t"]).all())


def test_padding_preserves_values_and_gradients():
    x = paddle.randn([1, 4, 3, 5, 8])
    x.stop_gradient = False
    padded = pad_3d(x, (0, 1, 3))
    assert tuple(padded.shape) == (1, 4, 4, 8, 8)
    y = crop_3d(padded, (0, 1, 3))
    np.testing.assert_array_equal(x.numpy(), y.numpy())
    y.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), np.ones(x.shape))


def test_coordinate_pooling_preserves_upstream_float32_rounding():
    # Frozen Torch CUDA values from the official 400 x 800 sample. A native
    # Paddle avg_pool reduction differs here and changes short-wave Fourier
    # features enough to break the pretrained forecast regression.
    lat = (90 - np.arange(400) * 0.45).astype("float32")
    lon = (np.arange(800) * 0.45).astype("float32")
    grid = np.stack(np.meshgrid(lat, lon, indexing="ij"), axis=0)[None]
    pooled = _patch_average(paddle.to_tensor(grid), (4, 4)).numpy()
    expected = {
        (0, 0, 0, 0): 89.32499694824219,
        (0, 0, 16, 0): 60.52499008178711,
        (0, 0, 99, 199): -88.87500762939453,
        (0, 1, 0, 0): 0.6749999523162842,
        (0, 1, 0, 1): 2.4750001430511475,
        (0, 1, 0, 146): 263.4749755859375,
        (0, 1, 99, 199): 358.875,
    }
    for index, value in expected.items():
        assert pooled[index] == np.float32(value)
    padded = np.pad(grid, ((0, 0), (0, 0), (0, 1), (0, 1)), constant_values=12345)
    np.testing.assert_array_equal(
        _patch_average(paddle.to_tensor(padded), (4, 4)).numpy(), pooled
    )


def test_attention_layout_and_additive_mask():
    rng = np.random.default_rng(8)
    q, k, v = (
        rng.standard_normal(shape).astype("float32")
        for shape in ((2, 3, 4, 8), (2, 3, 5, 8), (2, 3, 5, 8))
    )
    mask = np.zeros((1, 1, 4, 5), dtype="float32")
    mask[..., -1] = -100
    logits = q @ k.swapaxes(-2, -1) / np.sqrt(8) + mask
    probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    expected = probabilities / probabilities.sum(axis=-1, keepdims=True) @ v
    actual = scaled_dot_product_attention(
        *(paddle.to_tensor(a) for a in (q, k, v)), attn_mask=paddle.to_tensor(mask)
    )
    np.testing.assert_allclose(actual.numpy(), expected, atol=5e-7, rtol=2e-5)


@pytest.mark.parametrize("batch_mask", [False, True])
@pytest.mark.parametrize("broadcast_key", [False, True])
def test_attention_chunking_preserves_windows_masks_and_gradients(
    batch_mask, broadcast_key
):
    q = paddle.randn([5, 2, 7, 8], dtype="float64")
    k = paddle.randn([1 if broadcast_key else 5, 2, 9, 8], dtype="float64")
    v = paddle.randn([1 if broadcast_key else 5, 2, 9, 8], dtype="float64")
    for value in (q, k, v):
        value.stop_gradient = False
    mask = paddle.zeros([5 if batch_mask else 1, 1, 7, 9], dtype="float64")
    mask[..., -2:] = -100
    reference = scaled_dot_product_attention(q, k, v, attn_mask=mask)
    ref_grads = paddle.grad(reference.square().sum(), (q, k, v))
    actual = scaled_dot_product_attention(
        q, k, v, attn_mask=mask, max_logit_elements=252
    )
    gradients = paddle.grad(actual.square().sum(), (q, k, v))
    np.testing.assert_allclose(
        actual.numpy(), reference.numpy(), rtol=1e-12, atol=1e-12
    )
    for observed, expected in zip(gradients, ref_grads):
        np.testing.assert_allclose(
            observed.numpy(), expected.numpy(), rtol=1e-11, atol=1e-11
        )


def test_level_decoder_chunking_preserves_each_location_and_pressure_level():
    model, _ = model_and_batch()
    model.to(dtype=paddle.float64)
    model.eval()
    dim = model.decoder.embed_dim
    levels = paddle.randn([1, 5, 7, dim], dtype="float64")
    context = paddle.randn([1, 5, 3, dim], dtype="float64")
    with paddle.no_grad():
        expected = model.decoder.deaggregate_levels(
            levels, context, model.decoder.level_decoder, max_batch_size=64
        )
        observed = model.decoder.deaggregate_levels(
            levels, context, model.decoder.level_decoder, max_batch_size=2
        )
    assert tuple(observed.shape) == (1, 5, 7, dim)
    np.testing.assert_allclose(
        observed.numpy(), expected.numpy(), rtol=1e-10, atol=1e-10
    )


def test_era5_rmse_uses_latitude_weights():
    from examples.aurora.evaluate_era5 import weighted_rmse

    actual = np.array([[1.0, 1.0], [3.0, 3.0]])
    truth = np.zeros_like(actual)
    observed = weighted_rmse(actual, truth, np.array([0.0, 60.0]))
    assert observed == pytest.approx(np.sqrt(11 / 3))
    assert weighted_rmse(np.full((2, 2), 3.0), truth, [0.0, 60.0]) == pytest.approx(3.0)


@pytest.mark.parametrize("use_lora", [False, True])
def test_forward_backward_and_rollout(use_lora):
    model, batch = model_and_batch(use_lora=use_lora)
    prediction = model(batch)
    assert tuple(prediction.surf_vars["2t"].shape) == (1, 1, 16, 32)
    assert tuple(prediction.atmos_vars["t"].shape) == (1, 1, 4, 16, 32)
    normalized = prediction.normalise(model.surf_stats)
    loss = sum(v.square().mean() for v in normalized.surf_vars.values())
    loss += sum(v.square().mean() for v in normalized.atmos_vars.values())
    loss.backward()
    assert model.encoder.surf_token_embeds.weights["2t"].grad is not None
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert bool(paddle.isfinite(parameter.grad).all())
    model.eval()
    with paddle.no_grad():
        predictions = list(rollout(model, batch, 2))
    assert predictions[-1].metadata.time[0] == batch.metadata.time[0] + timedelta(
        hours=12
    )
    assert predictions[-1].metadata.rollout_step == 2


def test_checkpoint_layouts_and_strict_validation(tmp_path):
    model, batch = model_and_batch(use_lora=True)
    state = model.state_dict()
    linear = {
        f"{name}.weight"
        for name, layer in model.named_sublayers()
        if isinstance(layer, paddle.nn.Linear)
    }
    torch_arrays = {
        k: v.numpy().T.copy() if k in linear else v.numpy().copy()
        for k, v in state.items()
    }
    other, _ = model_and_batch(use_lora=True)
    assert other.load_torch_state_dict(torch_arrays) == ([], [])
    for name, value in state.items():
        np.testing.assert_array_equal(value.numpy(), other.state_dict()[name].numpy())
    path = tmp_path / "model.pdparams"
    paddle.save(other.state_dict(), str(path))
    other.load_checkpoint_local(str(path))
    bad = dict(torch_arrays)
    bad.pop(next(iter(bad)))
    with pytest.raises(ValueError, match="missing"):
        other.load_torch_state_dict(bad)
    bad = dict(torch_arrays)
    bad["encoder.pos_embed.weight"] = np.zeros((1, 2), dtype="float32")
    with pytest.raises(ValueError, match="shape mismatch"):
        other.load_torch_state_dict(bad)


def test_recompute_preserves_gradients():
    first, batch = model_and_batch()
    second, _ = model_and_batch()
    second.set_state_dict(first.state_dict())
    second.configure_activation_checkpointing()
    second.configure_activation_checkpointing()  # Idempotent, no nested wrapping.
    for model in (first, second):
        pred = model(batch).normalise(model.surf_stats)
        sum(v.square().mean() for v in pred.surf_vars.values()).backward()
    for (name, p), (_, q) in zip(first.named_parameters(), second.named_parameters()):
        if p.grad is not None:
            np.testing.assert_allclose(
                p.grad.numpy(), q.grad.numpy(), atol=1e-5, rtol=2e-4, err_msg=name
            )


def v1p5_model(ensemble=False):
    cls = AuroraV1p5Ensemble if ensemble else AuroraV1p5
    return cls(
        **small_config(),
        surf_vars=("2t", "10u", "10v", "msl", "scaled_tp_1h", "insolation"),
        static_vars=("lsm", "z", "slt"),
        output_only_surf_vars=("scaled_tp_1h",),
        autocast=False,
    )


def test_v1p5_hourly_rollout_and_output_only_variables():
    model = v1p5_model()
    batch = make_batch(paddle, Batch, Metadata)
    batch.surf_vars["insolation"] = paddle.zeros_like(batch.surf_vars["2t"])
    with paddle.no_grad():
        predictions = list(rollout(model, batch, 2, fine_lead_times=[3.0, 6.0]))
    assert len(predictions) == 4
    assert "scaled_tp_1h" in predictions[0].surf_vars
    assert "insolation" in predictions[-1].surf_vars
    assert [p.metadata.time[0] - batch.metadata.time[0] for p in predictions] == [
        timedelta(hours=h) for h in (3, 6, 9, 12)
    ]


def test_ensemble_generator_replay_and_member_independence():
    model = v1p5_model(ensemble=True)
    for name, parameter in model.named_parameters():
        if "ln_modulation.1." in name:
            paddle.nn.initializer.Normal(std=0.02)(parameter)
    model.eval()
    batch = make_batch(paddle, Batch, Metadata, batch_size=2)

    def predict(seeds):
        model.reset_noise()
        streams = tuple(make_generator(seed) for seed in seeds)
        with paddle.no_grad():
            return model(batch, lead_times=paddle.full([2], 6.0), generator=streams)

    first, replay, changed = predict((1, 2)), predict((1, 2)), predict((1, 3))
    for key in first.surf_vars:
        np.testing.assert_array_equal(
            first.surf_vars[key].numpy(), replay.surf_vars[key].numpy()
        )
        np.testing.assert_array_equal(
            first.surf_vars[key][0].numpy(), changed.surf_vars[key][0].numpy()
        )
    assert not np.array_equal(
        first.surf_vars["2t"][1].numpy(), changed.surf_vars["2t"][1].numpy()
    )
    with pytest.raises(ValueError, match="one generator"):
        model(batch, lead_times=paddle.full([2], 6.0), generator=(make_generator(1),))


def test_batch_tiling_splitting_and_netcdf(tmp_path):
    batch = make_batch(paddle, Batch, Metadata)
    tiled = tile_batch(batch, 3)
    parts = split_batch(tiled, 3)
    assert len(parts) == 3
    np.testing.assert_array_equal(
        parts[1].surf_vars["2t"].numpy(), batch.surf_vars["2t"].numpy()
    )
    path = tmp_path / "batch.nc"
    batch.to_netcdf(path)
    restored = Batch.from_netcdf(path)
    np.testing.assert_array_equal(
        restored.atmos_vars["t"].numpy(), batch.atmos_vars["t"].numpy()
    )


@pytest.mark.parametrize(
    "cls",
    [
        Aurora,
        AuroraPretrained,
        Aurora12hPretrained,
        AuroraHighRes,
        AuroraAirPollution,
        AuroraWave,
        AuroraV1p5,
        AuroraV1p5Ensemble,
    ],
)
def test_model_family_forward(cls):
    model = cls(**small_config(), autocast=False)
    patch = model.patch_size
    height, width = patch * 4 + 1, patch * 8
    levels = (100, 250, 500, 850)
    surf_names = inspect.signature(cls.__init__).parameters.get("surf_vars")
    surf_names = (
        surf_names.default
        if surf_names and isinstance(surf_names.default, tuple)
        else model.surf_vars
    )
    output_only = set(model.output_only_surf_vars)

    def value(name, shape):
        return paddle.full(
            shape, locations.get(name, 1.0) + 0.1 * scales.get(name, 1.0)
        )

    batch = Batch(
        surf_vars={
            name: value(name, [1, 2, height, width])
            for name in surf_names
            if name not in output_only
        },
        static_vars={name: value(name, [height, width]) for name in model.static_vars},
        atmos_vars={
            name: paddle.stack(
                [
                    value(f"{name}_{level_to_str(level)}", [1, 2, height, width])
                    for level in levels
                ],
                axis=2,
            )
            for name in model.atmos_vars
        },
        metadata=Metadata(
            lat=paddle.linspace(90, -90, height),
            lon=paddle.arange(width, dtype="float32") * (360 / width),
            time=make_batch(paddle, Batch, Metadata).metadata.time,
            atmos_levels=levels,
        ),
    )
    model.eval()
    with paddle.no_grad():
        prediction = model(
            batch,
            lead_times=paddle.full([1], 6.0) if model.variable_lead_time else None,
        )
    assert tuple(prediction.surf_vars["2t"].shape) == (1, 1, height - 1, width)
    assert bool(paddle.isfinite(prediction.surf_vars["2t"]).all())


def test_paddlescience_dictionary_adapter():
    batch = make_batch(paddle, Batch, Metadata)
    model = AuroraArch(
        model_kwargs=small_config(),
        metadata=batch.metadata,
        static_vars=batch.static_vars,
        output_keys=("surf_2t", "atmos_t"),
    )
    inputs = {f"surf_{k}": v for k, v in batch.surf_vars.items()}
    inputs.update({f"atmos_{k}": v for k, v in batch.atmos_vars.items()})
    outputs = model(inputs)
    assert tuple(outputs["surf_2t"].shape) == (1, 1, 16, 32)
    assert tuple(outputs["atmos_t"].shape) == (1, 1, 4, 16, 32)
    sum(v.square().mean() for v in outputs.values()).backward()
    assert any(p.grad is not None for p in model.parameters())


def test_acceptance_does_not_hide_one_bad_metric_in_the_average():
    from examples.aurora.evaluate_era5 import assess_acceptance

    forward = [{"step": 1, "max_abs_normalized": 9e-5, "relative_l2_normalized": 9e-6}]
    metrics = [
        {"torch_weighted_rmse": 1.0, "paddle_weighted_rmse": 1.0} for _ in range(100)
    ]
    metrics[-1]["paddle_weighted_rmse"] = 1.02
    result = assess_acceptance(forward, metrics)
    assert result["same_input_first_forward"] == "passed"
    assert result["status"] == "failed"
    assert result["metric_failures"] == 1


@pytest.mark.parametrize("compression", [0, 1])
def test_verification_netcdf_writer_preserves_all_fields_exactly(tmp_path, compression):
    from examples.aurora.netcdf_io import write_batch

    batch = make_batch(paddle, Batch, Metadata)
    path = tmp_path / "batch.nc"
    write_batch(batch, path, compression)
    actual = Batch.from_netcdf(path)
    for kind in ("surf_vars", "atmos_vars", "static_vars"):
        assert set(getattr(actual, kind)) == set(getattr(batch, kind))
        for name, expected in getattr(batch, kind).items():
            np.testing.assert_array_equal(
                getattr(actual, kind)[name].numpy(), expected.numpy()
            )
    np.testing.assert_array_equal(
        actual.metadata.lat.numpy(), batch.metadata.lat.numpy()
    )
    np.testing.assert_array_equal(
        actual.metadata.lon.numpy(), batch.metadata.lon.numpy()
    )
    assert actual.metadata.time == batch.metadata.time
    assert actual.metadata.atmos_levels == batch.metadata.atmos_levels
    assert actual.metadata.rollout_step == batch.metadata.rollout_step


def test_acceptance_handles_zero_baseline_and_rejects_empty_results():
    from examples.aurora.evaluate_era5 import assess_acceptance

    forward = [{"step": 1, "max_abs_normalized": 0.0, "relative_l2_normalized": 0.0}]
    exact = [{"torch_weighted_rmse": 0.0, "paddle_weighted_rmse": 0.0}]
    assert assess_acceptance(forward, exact)["status"] == "passed"
    nonzero = [{"torch_weighted_rmse": 0.0, "paddle_weighted_rmse": 1e-8}]
    assert assess_acceptance(forward, nonzero)["status"] == "failed"
    with pytest.raises(ValueError, match="requires"):
        assess_acceptance([], exact)


def test_acceptance_reports_forward_failure_despite_equal_forecast_metrics():
    from examples.aurora.evaluate_era5 import assess_acceptance

    forward = [
        {"step": 1, "max_abs_normalized": 1.1e-4, "relative_l2_normalized": 1e-6}
    ]
    metrics = [{"torch_weighted_rmse": 1.0, "paddle_weighted_rmse": 1.0}]
    result = assess_acceptance(forward, metrics)
    assert result["all_lead_rmse_alignment"] == "passed"
    assert result["same_input_first_forward"] == "failed"
    assert result["status"] == "failed"


def test_paddlescience_solver_training(tmp_path):
    batch = make_batch(paddle, Batch, Metadata)
    model = AuroraArch(
        model_kwargs=small_config(),
        metadata=batch.metadata,
        static_vars=batch.static_vars,
        output_keys=("surf_2t",),
    )
    inputs = {f"surf_{k}": v.numpy() for k, v in batch.surf_vars.items()}
    inputs.update({f"atmos_{k}": v.numpy() for k, v in batch.atmos_vars.items()})
    labels = {"surf_2t": batch.surf_vars["2t"][:, -1:, :-1].numpy()}
    constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {"name": "NamedArrayDataset", "input": inputs, "label": labels},
            "sampler": {"name": "BatchSampler", "shuffle": False},
            "batch_size": 1,
        },
        ppsci.loss.MSELoss("mean"),
        name="forecast",
    )
    optimizer = ppsci.optimizer.Adam(1e-4)(model)
    parameter = model.model.encoder.surf_token_embeds.weights["2t"]
    before = parameter.numpy().copy()
    ppsci.utils.logger.init_logger()
    solver = ppsci.solver.Solver(
        model,
        {"forecast": constraint},
        optimizer=optimizer,
        epochs=1,
        iters_per_epoch=1,
        device=paddle.device.get_device(),
        output_dir=str(tmp_path),
    )
    solver.train()
    assert np.isfinite(parameter.numpy()).all()
    assert not np.array_equal(parameter.numpy(), before)


def test_supervised_loss_uses_each_official_pressure_level_scale():
    from examples.aurora.finetune import normalized_mse

    levels = (100, 850)
    surface_scale = scales["2t"]
    level_scales = paddle.to_tensor(
        [scales[f"q_{level_to_str(level)}"] for level in levels]
    ).reshape([1, 1, 2, 1, 1])
    outputs = {
        "surf_2t": paddle.full([1, 1, 2, 3], 2 * surface_scale),
        "atmos_q": 2 * level_scales * paddle.ones([1, 1, 2, 2, 3]),
    }
    for value in outputs.values():
        value.stop_gradient = False
    labels = {key: paddle.zeros_like(value) for key, value in outputs.items()}
    loss = normalized_mse(levels)(outputs, labels)["normalized_mse"]
    np.testing.assert_allclose(loss.numpy(), 4.0, rtol=1e-6)
    loss.backward()
    for value in outputs.values():
        assert bool(paddle.isfinite(value.grad).all())
        assert bool((value.grad != 0).all())


@pytest.mark.parametrize("mismatch", ["time", "levels", "longitude"])
def test_supervised_pair_rejects_misaligned_truth(mismatch):
    from examples.aurora.finetune import validate_pair

    _, batch = model_and_batch()
    target = dataclasses.replace(
        batch,
        surf_vars={k: v[:, -1:] for k, v in batch.surf_vars.items()},
        atmos_vars={k: v[:, -1:] for k, v in batch.atmos_vars.items()},
        metadata=dataclasses.replace(
            batch.metadata,
            time=tuple(t + timedelta(hours=6) for t in batch.metadata.time),
        ),
    )
    if mismatch == "time":
        metadata = dataclasses.replace(target.metadata, time=batch.metadata.time)
    elif mismatch == "levels":
        metadata = dataclasses.replace(
            target.metadata, atmos_levels=(100, 300, 500, 850)
        )
    else:
        metadata = dataclasses.replace(target.metadata, lon=target.metadata.lon + 0.01)
    target = dataclasses.replace(target, metadata=metadata)
    with pytest.raises((ValueError, AssertionError)):
        validate_pair(batch, target, timedelta(hours=6), 4)


def test_era5_bounded_resume_validates_ranges_and_content(tmp_path):
    from examples.aurora.download_era5 import fetch_bounded

    content = b"0123456789"
    partial = tmp_path / "chunk.partial"
    partial.write_bytes(content[:4])

    class Response:
        status_code = 206

        def __init__(self, start, end):
            end = min(end, len(content) - 1)
            self.content = content[start : end + 1]
            self.headers = {
                "Content-Range": f"bytes {start}-{end}/{len(content)}",
                "ETag": "frozen-object",
            }

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            pass

    class Session:
        def __init__(self):
            self.ranges = []

        def get(self, url, headers, timeout):
            requested = headers["Range"]
            self.ranges.append(requested)
            start, end = map(int, requested[len("bytes=") :].split("-"))
            return Response(start, end)

    session = Session()
    assert (
        fetch_bounded(session, "https://example.test/chunk", partial, 4)
        == "frozen-object"
    )
    assert session.ranges == ["bytes=4-7", "bytes=8-11"]
    assert partial.read_bytes() == content


@pytest.mark.skipif(
    not os.environ.get("AURORA_TEST_DEVICE", "cpu").startswith("gpu"),
    reason="GPU AMP check",
)
@pytest.mark.parametrize("dtype", [paddle.float16, paddle.bfloat16])
def test_gpu_amp_backward(dtype):
    model, batch = model_and_batch(
        autocast=True,
        autocast_dtype=dtype,
        use_fp16_safe_attention=dtype == paddle.float16,
    )
    pred = model(batch).normalise(model.surf_stats)
    loss = sum(v.square().mean() for v in pred.surf_vars.values())
    assert bool(paddle.isfinite(loss))
    loss.backward()
    assert bool(
        paddle.isfinite(model.encoder.surf_token_embeds.weights["2t"].grad).all()
    )


@pytest.mark.skipif(
    not os.environ.get("AURORA_TEST_DEVICE", "cpu").startswith("gpu"),
    reason="GPU fused attention",
)
def test_gpu_fused_attention_layout():
    q, k, v = [paddle.randn([2, 3, 7, 32]) for _ in range(3)]
    expected = scaled_dot_product_attention(q, k, v)
    actual = scaled_dot_product_attention(
        q.astype("float16"), k.astype("float16"), v.astype("float16")
    )
    np.testing.assert_allclose(
        actual.astype("float32").numpy(), expected.numpy(), atol=2e-3, rtol=2e-2
    )
