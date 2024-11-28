import sys

import paddle
import wandb
from configmypy import ArgparseConfig
from configmypy import ConfigPipeline
from configmypy import YamlConfig
from neuralop import H1Loss
from neuralop import LpLoss
from neuralop import Trainer
from neuralop import get_model
from neuralop.datasets import load_darcy_flow_small
from neuralop.datasets.data_transforms import MGPatchingDataProcessor
from neuralop.training import setup
from neuralop.training.callbacks import BasicLoggerCallback
from neuralop.utils import count_model_params
from neuralop.utils import get_wandb_api_key
from paddle import DataParallel as DDP

paddle.device.set_device("gpu")

# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./darcy_config.yaml", config_name="default", config_folder="../config"
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder="../config"),
    ]
)
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name

# Set-up distributed communication, if using
device, is_logger = setup(config)

# Set up WandB logging
wandb_args = None
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                config_name,
                config.tfno2d.n_layers,
                config.tfno2d.hidden_channels,
                config.tfno2d.n_modes_width,
                config.tfno2d.n_modes_height,
                config.tfno2d.factorization,
                config.tfno2d.rank,
                config.patching.levels,
                config.patching.padding,
            ]
        )
    wandb_args = dict(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]

# Make sure we only print information when needed
config.verbose = config.verbose and is_logger

# Print config to screen
if config.verbose and is_logger:
    pipe.log()
    sys.stdout.flush()

# Loading the Darcy flow dataset
train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=config.data.n_train,
    batch_size=config.data.batch_size,
    positional_encoding=config.data.positional_encoding,
    test_resolutions=config.data.test_resolutions,
    n_tests=config.data.n_tests,
    test_batch_sizes=config.data.test_batch_sizes,
    encode_input=False,
    encode_output=False,
)
# convert dataprocessor to an MGPatchingDataprocessor if patching levels > 0
if config.patching.levels > 0:
    data_processor = MGPatchingDataProcessor(
        in_normalizer=data_processor.in_normalizer,
        out_normalizer=data_processor.out_normalizer,
        positional_encoding=data_processor.positional_encoding,
        padding_fraction=config.patching.padding,
        stitching=config.patching.stitching,
        levels=config.patching.levels,
    )
data_processor = data_processor

model = get_model(config)
model = model

# Use distributed data parallel
if config.distributed.use_distributed:
    model = DDP(
        model, device_ids=[device.index], output_device=device.index, static_graph=True
    )

# Create the optimizer
if config.opt.scheduler == "ReduceLROnPlateau":
    scheduler = paddle.optimizer.lr.ReduceOnPlateau(
        learning_rate=config.opt.learning_rate,
        factor=config.opt.gamma,
        patience=config.opt.scheduler_patience,
        mode="min",
    )
elif config.opt.scheduler == "CosineAnnealingLR":
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=config.opt.learning_rate, T_max=config.opt.scheduler_T_max
    )
elif config.opt.scheduler == "StepLR":
    scheduler = paddle.optimizer.lr.StepDecay(
        learning_rate=config.opt.learning_rate,
        step_size=config.opt.step_size,
        gamma=config.opt.gamma,
    )
else:
    raise ValueError(f"Got scheduler={config.opt.scheduler}")

optimizer = paddle.optimizer.Adam(
    parameters=model.parameters(),
    learning_rate=scheduler,
    weight_decay=config.opt.weight_decay,
)

# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
if config.opt.training_loss == "l2":
    train_loss = l2loss
elif config.opt.training_loss == "h1":
    train_loss = h1loss
else:
    raise ValueError(
        f"Got training_loss={config.opt.training_loss} "
        f'but expected one of ["l2", "h1"]'
    )
eval_losses = {"h1": h1loss, "l2": l2loss}

if config.verbose and is_logger:
    print("\n### MODEL ###\n", model)
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULER ###\n", scheduler)
    print("\n### LOSSES ###")
    print(f"\n * Train: {train_loss}")
    print(f"\n * Test: {eval_losses}")
    print("\n### Beginning Training...\n")
    sys.stdout.flush()

trainer = Trainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    device=device,
    data_processor=data_processor,
    amp_autocast=config.opt.amp_autocast,
    wandb_log=config.wandb.log,
    log_test_interval=config.wandb.log_test_interval,
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose and is_logger,
    callbacks=[BasicLoggerCallback(wandb_args)],
)

# Log parameter count
if is_logger:
    n_params = count_model_params(model)

    if config.verbose:
        print(f"\nn_params: {n_params}")
        sys.stdout.flush()

    if config.wandb.log:
        to_log = {"n_params": n_params}
        if config.n_params_baseline is not None:
            to_log["n_params_baseline"] = (config.n_params_baseline,)
            to_log["compression_ratio"] = (config.n_params_baseline / n_params,)
            to_log["space_savings"] = 1 - (n_params / config.n_params_baseline)
        wandb.log(to_log)
        wandb.watch(model)

trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)

if config.wandb.log and is_logger:
    wandb.finish()
