"""
U-NO on Darcy-Flow
==================

In this example, we demonstrate how to train a U-shaped Neural Operator on 
the small Darcy-Flow example we ship with the package
"""

# %%
#


import paddle
import matplotlib.pyplot as plt
import sys
from ppsci.contrib.neuralop.models import TFNO, UNO
from ppsci.contrib.neuralop import Trainer
from ppsci.contrib.neuralop.datasets import load_darcy_flow_small
from ppsci.contrib.neuralop.utils import count_model_params
from ppsci.contrib.neuralop import LpLoss, H1Loss

if paddle.device.cuda.device_count() >= 1:
    paddle.device.set_device('gpu')
else:
    paddle.device.set_device("cpu")

# %%
# Loading the Darcy Flow dataset
train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000, batch_size=32,
        test_resolutions=[16, 32], n_tests=[100, 50],
        test_batch_sizes=[32, 32],
)

model = UNO(3, 1, hidden_channels=64, projection_channels=64, uno_out_channels=[32, 64, 64, 64, 32],
            uno_n_modes=[[16, 16], [8, 8], [8, 8], [8, 8], [16, 16]],
            uno_scalings=[[1.0, 1.0], [0.5, 0.5], [1, 1], [2, 2], [1, 1]],
            horizontal_skips_map=None, n_layers=5, domain_padding=0.2)

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()


# %%
# Create the optimizer
optimizer = paddle.optimizer.Adam(
    learning_rate=8e-3,
    parameters=model.parameters(),
    weight_decay=1e-4
)
scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=optimizer.get_lr(), T_max=30)
optimizer.set_lr_scheduler(scheduler=scheduler)


# %%
# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses = {'h1': h1loss, 'l2': l2loss}


# %%


print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()


# %%
# Create the trainer
trainer = Trainer(model=model,
                  n_epochs=20,
                  data_processor=data_processor,
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True)


# %%
# Actually train the model on our small Darcy-Flow dataset

trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler,
              regularizer=False,
              training_loss=train_loss,
              eval_losses=eval_losses)


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

test_samples = test_loaders[32].dataset

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y']
    # Model prediction
    out = model(x.unsqueeze(0)).cpu()

    ax = fig.add_subplot(3, 3, index*3 + 1)
    ax.imshow(x[0], cmap='gray')
    if index == 0:
        ax.set_title('Input x')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 2)
    ax.imshow(y.squeeze())
    if index == 0:
        ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 3)
    ax.imshow(out.squeeze().detach().numpy())
    if index == 0:
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
fig.show()
fig.savefig('a.png')