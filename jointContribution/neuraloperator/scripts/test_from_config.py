import time

import paddle
from configmypy import ArgparseConfig
from configmypy import ConfigPipeline
from configmypy import YamlConfig
from neuralop import get_model
from tensorly import tenalg

tenalg.set_backend("einsum")


# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./test_config.yaml", config_name="default", config_folder="../config"
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder="../config"),
    ]
)
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name

batch_size = config.data.batch_size
size = config.data.size

if paddle.device.cuda.device_count() >= 1:
    device = "gpu"
else:
    device = "cpu"

paddle.device.set_device(device=device)

model = get_model(config)
model = model

in_data = paddle.randn([batch_size, 3, size, size])
print(model.__class__)
print(model)

t1 = time.time()
out = model(in_data)
t = time.time() - t1
print(f"Output of size {out.shape} in {t}.")

loss = out.sum()
loss.backward()

# Check for unused params
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"Usused parameter {name}!")
