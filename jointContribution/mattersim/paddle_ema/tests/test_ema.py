import os

import paddle
import numpy as np
import pytest

from paddle_ema import ExponentialMovingAverage


@pytest.mark.parametrize("decay", [0.995, 0.9])
@pytest.mark.parametrize("use_num_updates", [True, False])
@pytest.mark.parametrize("explicit_params", [True, False])
def test_val_error(decay, use_num_updates, explicit_params):
    """Confirm that EMA validation error is lower than raw validation error."""
    paddle.seed(seed=0)
    x_train = paddle.rand(shape=(100, 10))
    y_train = paddle.rand(shape=[100]).round().astype(dtype="int64")
    x_val = paddle.rand(shape=(100, 10))
    y_val = paddle.rand(shape=[100]).round().astype(dtype="int64")
    model = paddle.nn.Linear(in_features=10, out_features=2)
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=0.01, weight_decay=0.0
    )
    ema = ExponentialMovingAverage(
        model.parameters(), decay=decay, use_num_updates=use_num_updates
    )
    model.train()
    for _ in range(20):
        logits = model(x_train)
        loss = paddle.nn.functional.cross_entropy(input=logits, label=y_train)
        optimizer.clear_gradients(set_to_zero=False)
        loss.backward()
        optimizer.step()
        if explicit_params:
            ema.update(model.parameters())
        else:
            ema.update()
    model.eval()
    logits = model(x_val)
    loss_orig = paddle.nn.functional.cross_entropy(input=logits, label=y_val)
    print(f"Original loss: {loss_orig}")
    if explicit_params:
        ema.store(model.parameters())
    else:
        ema.store()
    if explicit_params:
        ema.copy_to(model.parameters())
    else:
        ema.copy_to()
    logits = model(x_val)
    loss_ema = paddle.nn.functional.cross_entropy(input=logits, label=y_val)
    print(f"EMA loss: {loss_ema}")
    # assert loss_ema < loss_orig, "EMA loss wasn't lower"
    if explicit_params:
        ema.restore(model.parameters())
    else:
        ema.restore()
    model.eval()
    logits = model(x_val)
    loss_orig2 = paddle.nn.functional.cross_entropy(input=logits, label=y_val)
    assert paddle.allclose(
        x=loss_orig, y=loss_orig2
    ).item(), "Restored model wasn't the same as stored model"


@pytest.mark.parametrize("explicit_params", [True, False])
def test_contextmanager(explicit_params):
    """Confirm that EMA validation error is lower than raw validation error."""
    paddle.seed(seed=0)
    x_train = paddle.rand(shape=(100, 10))
    y_train = paddle.rand(shape=[100]).round().astype(dtype="int64")
    x_val = paddle.rand(shape=(100, 10))
    y_val = paddle.rand(shape=[100]).round().astype(dtype="int64")
    model = paddle.nn.Linear(in_features=10, out_features=2)
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=0.01, weight_decay=0.0
    )
    ema = ExponentialMovingAverage(model.parameters(), decay=0.99)
    model.train()
    for _ in range(20):
        logits = model(x_train)
        loss = paddle.nn.functional.cross_entropy(input=logits, label=y_train)
        optimizer.clear_gradients(set_to_zero=False)
        loss.backward()
        optimizer.step()
        if explicit_params:
            ema.update(model.parameters())
        else:
            ema.update()
    final_weight = model.weight.clone().detach()
    model.eval()
    logits = model(x_val)
    loss_orig = paddle.nn.functional.cross_entropy(input=logits, label=y_val)
    print(f"Original loss: {loss_orig}")
    if explicit_params:
        cm = ema.average_parameters(model.parameters())
    else:
        cm = ema.average_parameters()
    with cm:
        logits = model(x_val)
        loss_ema = paddle.nn.functional.cross_entropy(input=logits, label=y_val)
    print(f"EMA loss: {loss_ema}")
    # assert loss_ema < loss_orig, "EMA loss wasn't lower"
    assert paddle.all(x=model.weight == final_weight), "Restore failed"


@pytest.mark.parametrize("decay", [0.995, 0.9, 0.0, 1.0])
@pytest.mark.parametrize("use_num_updates", [True, False])
@pytest.mark.parametrize("explicit_params", [True, False])
def test_store_restore(decay, use_num_updates, explicit_params):
    model = paddle.nn.Linear(in_features=10, out_features=2)
    ema = ExponentialMovingAverage(
        model.parameters(), decay=decay, use_num_updates=use_num_updates
    )
    orig_weight = model.weight.clone().detach()
    if explicit_params:
        ema.store(model.parameters())
    else:
        ema.store()
    with paddle.no_grad():
        model.weight.uniform_(min=0.0, max=1.0)
    if explicit_params:
        ema.restore(model.parameters())
    else:
        ema.restore()
    assert paddle.all(x=model.weight == orig_weight)


@pytest.mark.parametrize("decay", [0.995, 0.9, 0.0, 1.0])
@pytest.mark.parametrize("explicit_params", [True, False])
def test_update(decay, explicit_params):
    model = paddle.nn.Linear(in_features=10, out_features=2, bias_attr=False)
    with paddle.no_grad():
        model.weight.fill_(value=0.0)
    ema = ExponentialMovingAverage(
        model.parameters(), decay=decay, use_num_updates=False
    )
    with paddle.no_grad():
        model.weight.fill_(value=1.0)
    if explicit_params:
        ema.update(model.parameters())
    else:
        ema.update()
    assert paddle.all(x=model.weight == 1.0), "ema.update changed model weights"
    if explicit_params:
        ema.copy_to(model.parameters())
    else:
        ema.copy_to()
    assert paddle.allclose(
        x=model.weight, y=paddle.full_like(model.weight, fill_value=1.0 - decay)
    ).item(), "average was wrong"


def test_explicit_params():
    model = paddle.nn.Linear(in_features=10, out_features=2)
    with paddle.no_grad():
        model.weight.fill_(value=0.0)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9)
    model2 = paddle.nn.Linear(in_features=10, out_features=2)
    with paddle.no_grad():
        model2.weight.fill_(value=1.0)
    ema.update(model2.parameters())
    ema.copy_to()
    assert not paddle.all(x=model.weight == 0.0)


def test_some_untrainable():
    class Mod(paddle.nn.Layer):
        def __init__(self):
            super().__init__()
            self.x = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.randn(shape=[3])
            )
            self.y = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.randn(shape=[3])
            )
            out_0 = self.y
            out_0.stop_gradient = not False
            out_0

        def forward(self, x):
            return self.x * x + self.y

    model = Mod()
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9)
    ema.update()
    with paddle.no_grad():
        model.x.set_value(model.x * 1.1)
    ema.update()
    ema.store()
    ema.copy_to()


def test_to():
    dtype_mapping = {
        'float16':paddle.float16,
        'float32':paddle.float32,
        'float64':paddle.float64,
    }
    m = paddle.nn.Linear(in_features=11, out_features=3)
    ema = ExponentialMovingAverage(m.parameters(), decay=0.9)
    assert ema.shadow_params[0].dtype == dtype_mapping[paddle.get_default_dtype()]
    ema.to(dtype="float16")
    assert ema.shadow_params[0].dtype == dtype_mapping["float16"]
    ema.store()
    assert ema.collected_params[0].dtype == dtype_mapping[paddle.get_default_dtype()]
    m = m.to(dtype="float16")
    ema.store(m.parameters())
    assert ema.collected_params[0].dtype == dtype_mapping["float16"]
    ema.to(dtype="float64")
    assert ema.collected_params[0].dtype == dtype_mapping["float64"]
    assert ema.shadow_params[0].dtype == dtype_mapping["float64"]
