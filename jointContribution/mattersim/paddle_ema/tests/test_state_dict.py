import copy

import paddle
import pytest

from paddle_ema import ExponentialMovingAverage


@pytest.mark.parametrize("decay", [0.995])
@pytest.mark.parametrize("use_num_updates", [True, False])
@pytest.mark.parametrize("explicit_params", [True, False])
def test_state_dict(decay, use_num_updates, explicit_params):
    model = paddle.nn.Linear(in_features=10, out_features=2, bias_attr=False)
    with paddle.no_grad():
        model.weight.fill_(value=0.0)
    ema = ExponentialMovingAverage(
        model.parameters(), decay=decay, use_num_updates=False
    )
    state_dict = copy.deepcopy(ema.state_dict())
    model2 = paddle.nn.Linear(in_features=10, out_features=2, bias_attr=False)
    ema2 = ExponentialMovingAverage(model2.parameters(), decay=0.0)
    ema2.set_state_dict(state_dict=state_dict)
    assert ema2.decay == decay
    assert paddle.allclose(x=ema2.shadow_params[0], y=ema.shadow_params[0]).item()
    with paddle.no_grad():
        model2.weight.fill_(value=1.0)
    if explicit_params:
        ema2.update(model2.parameters())
    else:
        ema2.update()
    assert paddle.all(x=model2.weight == 1.0), "ema.update changed model weights"
    ema.set_state_dict(state_dict=ema2.state_dict())
    if explicit_params:
        ema.copy_to(model.parameters())
    else:
        ema.copy_to()
    assert paddle.allclose(
        x=model.weight, y=paddle.full(shape=(1,), fill_value=1.0 - decay)
    ).item(), "average was wrong"


def test_state_dict_types():
    m1 = paddle.nn.Linear(in_features=10, out_features=2, bias_attr=False)
    m2 = paddle.nn.Linear(in_features=10, out_features=2, bias_attr=False)
    m2.to("float16")
    ema1 = ExponentialMovingAverage(m1.parameters(), decay=0.9)
    ema2 = ExponentialMovingAverage(m2.parameters(), decay=0.9)
    ema1.update()
    ema2.update()
    ema2.set_state_dict(state_dict=ema1.state_dict())
    ema1.copy_to()
    ema2.copy_to()
    assert m1.weight.dtype == paddle.get_default_dtype()
    assert m2.weight.dtype == "float16"
    assert paddle.allclose(x=m1.weight.to("float16"), y=m2.weight).item()


def test_bad_state_dict1():
    m = paddle.nn.Linear(in_features=10, out_features=2, bias_attr=False)
    ema = ExponentialMovingAverage(m.parameters(), decay=0.9)
    sd = ema.state_dict()
    sd["shadow_params"][0] = paddle.zeros(shape=[3, 7])
    ema.set_state_dict(state_dict=sd)
    with pytest.raises(RuntimeError):
        ema.copy_to()
    assert paddle.any(x=m.weight.abs() > 0)


def test_bad_state_dict2():
    m = paddle.nn.Linear(in_features=10, out_features=2, bias_attr=False)
    ema = ExponentialMovingAverage(m.parameters(), decay=0.9)
    sd = ema.state_dict()
    sd["shadow_params"] = sd["shadow_params"][:-1]
    with pytest.raises(ValueError):
        ema.set_state_dict(state_dict=sd)
