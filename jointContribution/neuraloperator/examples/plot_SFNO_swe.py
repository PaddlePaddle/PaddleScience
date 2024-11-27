"""
Training a SFNO on the spherical Shallow Water equations
==========================================================

In this example, we demonstrate how to use the small Spherical Shallow Water Equations example we ship with the package
to train a Spherical Fourier-Neural Operator
"""

# %%
#


import sys

import matplotlib.pyplot as plt
import paddle
from neuralop import LpLoss
from neuralop import Trainer
from neuralop.datasets import load_spherical_swe
from neuralop.models import SFNO
from neuralop.utils import count_model_params

if paddle.device.cuda.device_count() >= 1:
    paddle.device.set_device("gpu")
else:
    paddle.device.set_device("cpu")

# %%
# Loading the Navier-Stokes dataset in 128x128 resolution
train_loader, test_loaders = load_spherical_swe(
    n_train=200,
    batch_size=4,
    train_resolution=(32, 64),
    test_resolutions=[(32, 64), (64, 128)],
    n_tests=[50, 50],
    test_batch_sizes=[10, 10],
)


# %%
# We create a tensorized FNO model

model = SFNO(
    n_modes=(32, 32),
    in_channels=3,
    out_channels=3,
    hidden_channels=32,
    projection_channels=64,
    factorization="dense",
)

n_params = count_model_params(model)
print(f"\nOur model has {n_params} parameters.")
sys.stdout.flush()


# %%
# Create the optimizer
scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=8e-4, T_max=30)
optimizer = paddle.optimizer.Adam(
    learning_rate=scheduler, parameters=model.parameters(), weight_decay=1e-4
)


# %%
# Creating the losses
l2loss = LpLoss(d=2, p=2, reduce_dims=(0, 1))
# h1loss = H1Loss(d=2, reduce_dims=(0,1))

train_loss = l2loss
eval_losses = {"l2": l2loss}  # 'h1': h1loss,


# %%


print("\n### MODEL ###\n", model)
print("\n### OPTIMIZER ###\n", optimizer)
print("\n### SCHEDULER ###\n", scheduler)
print("\n### LOSSES ###")
print(f"\n * Train: {train_loss}")
print(f"\n * Test: {eval_losses}")
sys.stdout.flush()


# %%
# Create the trainer
trainer = Trainer(
    model=model,
    n_epochs=20,
    wandb_log=False,
    log_test_interval=3,
    use_distributed=False,
    verbose=True,
)


# %%
# Actually train the model on our small Darcy-Flow dataset

trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)


# %%
# Plot the prediction, and compare with the ground-truth
# Note that we trained on a very small resolution for
# a very small number of epochs
# In practice, we would train at larger resolution, on many more samples.
#
# However, for practicity, we created a minimal example that
# i) fits in just a few Mb of memory
# ii) can be trained quickly on CPU
#
# In practice we would train a Neural Operator on one or multiple GPUs

fig = plt.figure(figsize=(7, 7))
for index, resolution in enumerate([(32, 64), (64, 128)]):
    test_samples = test_loaders[resolution].dataset
    data = test_samples[0]
    # Input x
    x = data["x"]
    # Ground-truth
    y = data["y"][0, ...].numpy()
    # Model prediction
    x_in = x.unsqueeze(0)
    out = model(x_in).squeeze()[0, ...].numpy()
    x = x[0, ...].numpy()

    ax = fig.add_subplot(2, 3, index * 3 + 1)
    ax.imshow(x)
    ax.set_title(f"Input x {resolution}")
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(2, 3, index * 3 + 2)
    ax.imshow(y)
    ax.set_title("Ground-truth y")
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(2, 3, index * 3 + 3)
    ax.imshow(out)
    ax.set_title("Model prediction")
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle("Inputs, ground-truth output and prediction.", y=0.98)
plt.tight_layout()
fig.show()
