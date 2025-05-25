# pytorch_ema

A small library for computing exponential moving averages of model
parameters.

This library was originally written for personal use. Nevertheless, if you run into issues
or have suggestions for improvement, feel free to open either a new issue or
pull request.

## Installation
For the stable version from PyPI:
```bash
pip install torch-ema
```

For the latest GitHub version:
```
pip install -U git+https://github.com/fadel/pytorch_ema
```

## Usage

### Example

```python
import torch
import torch.nn.functional as F

from torch_ema import ExponentialMovingAverage

torch.manual_seed(0)
x_train = torch.rand((100, 10))
y_train = torch.rand(100).round().long()
x_val = torch.rand((100, 10))
y_val = torch.rand(100).round().long()
model = torch.nn.Linear(10, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

# Train for a few epochs
model.train()
for _ in range(20):
    logits = model(x_train)
    loss = F.cross_entropy(logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Update the moving average with the new parameters from the last optimizer step
    ema.update()

# Validation: original
model.eval()
logits = model(x_val)
loss = F.cross_entropy(logits, y_val)
print(loss.item())

# Validation: with EMA
# the .average_parameters() context manager
# (1) saves original parameters before replacing with EMA version
# (2) copies EMA parameters to model
# (3) after exiting the `with`, restore original parameters to resume training later
with ema.average_parameters():
    logits = model(x_val)
    loss = F.cross_entropy(logits, y_val)
    print(loss.item())
```

### Manual validation mode

While the `average_parameters()` context manager is convenient, you can also manually execute the same series of operations:
```python
ema.store()
ema.copy_to()
# ...
ema.restore()
```

### Custom parameters

By default the methods of `ExponentialMovingAverage` act on the model parameters the object was constructed with, but any compatible iterable of parameters can be passed to any method (such as `store()`, `copy_to()`, `update()`, `restore()`, and `average_parameters()`):
```python
model = torch.nn.Linear(10, 2)
model2 = torch.nn.Linear(10, 2)
ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
# train
# calling `ema.update()` will use `model.parameters()`
ema.copy_to(model2)
# model2 now contains the averaged weights
```

### Resuming training

Like a PyTorch optimizer, `ExponentialMovingAverage` objects have `state_dict()`/`load_state_dict()` methods to allow pausing, serializing, and restarting training without losing shadow parameters, stored parameters, or the update count.

### GPU/device support

`ExponentialMovingAverage` objects have a `.to()` function (like `torch.Tensor`) that can move the object's internal state to a different device or floating-point dtype.


For more details on individual methods, please check the docstrings.
