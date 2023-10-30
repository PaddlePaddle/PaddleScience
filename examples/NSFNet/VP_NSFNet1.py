import ppsci
import paddle
paddle.set_default_dtype("float32")
import numpy as np
from ppsci.utils import logger

OUTPUT_DIR = "./output"
import ppsci
import paddle
paddle.set_default_dtype("float32")
import numpy as np
from ppsci.utils import logger

OUTPUT_DIR = "./output"
logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")
# set random seed for reproducibility
SEED = 1234
ppsci.utils.misc.set_random_seed(SEED)

ITERS_PER_EPOCH=1
# set model
input_key=("x", "y")
output_key=("u","v","p")
model = ppsci.arch.MLP(
         input_key, output_key, 4, 50, "tanh", input_dim=len(input_key),output_dim=len(output_key),Xavier=True
     )


N_train = 2601
Nb_train = 400


# Load Data
Re = 40
lam = 0.5 * Re - np.sqrt(0.25 * (Re ** 2) + 4 * (np.pi ** 2))

x = np.linspace(-0.5, 1.0, 101)
y = np.linspace(-0.5, 1.5, 101)

yb1 = np.array([-0.5] * 100)
yb2 = np.array([1] * 100)
xb1 = np.array([-0.5] * 100)
xb2 = np.array([1.5] * 100)

y_train1 = np.concatenate([y[1:101], y[0:100], xb1, xb2], 0).astype('float32')
x_train1 = np.concatenate([yb1, yb2, x[0:100], x[1:101]], 0).astype('float32')

xb_train = x_train1.reshape(x_train1.shape[0], 1).astype('float32')
yb_train = y_train1.reshape(y_train1.shape[0], 1).astype('float32')
ub_train = 1 - np.exp(lam * xb_train) * np.cos(2 * np.pi * yb_train)
vb_train = lam / (2 * np.pi) * np.exp(lam * xb_train) * np.sin(2 * np.pi * yb_train)

x_train = ((np.random.rand(N_train, 1) - 1 / 3) * 3 / 2)
y_train = ((np.random.rand(N_train, 1) - 1 / 4) * 2)

# Test Data
np.random.seed(SEED)
x_star = ((np.random.rand(1000, 1) - 1 / 3) * 3 / 2).astype('float32')
y_star = ((np.random.rand(1000, 1) - 1 / 4) * 2).astype('float32')

u_star = 1 - np.exp(lam * x_star) * np.cos(2 * np.pi * y_star)
v_star = (lam / (2 * np.pi)) * np.exp(lam * x_star) * np.sin(2 * np.pi * y_star)
p_star = 0.5 * (1 - np.exp(2 * lam * x_star))


## Dimensionless

# Xb = np.concatenate([xb_train, yb_train], 1)
# lowb = Xb.min(0)
# upb = Xb.max(0)
# Xb = 2.0 * (Xb - lowb) / (upb - lowb) - 1.0
# xb_train, yb_train = Xb[:, 0:1], Xb[:, 1:2]
# X = np.concatenate([x_train, y_train], 1)
# X = 2.0 * (X - lowb) / (upb - lowb) - 1.0

# x_train, y_train = X[:, 0:1], X[:, 1:2]

# X_star = np.concatenate([x_star, y_star], 1)
# X_star = 2.0 * (X_star - lowb) / (upb - lowb) - 1.0
# x_star = X_star[:, 0:1]
# y_star = X_star[:, 1:2]

input_train = {"x": xb_train,"y":yb_train}
label_train = {"u": ub_train,"v":vb_train}

train_dataloader_cfg = {
    "dataset": {
        "name": "NamedArrayDataset",
        "input":input_train,
        "label":label_train
    },
    "batch_size": Nb_train,
    'iters_per_epoch':ITERS_PER_EPOCH,
    "sampler": {
        "name": "BatchSampler",
        "drop_last": False,
        "shuffle": False,
    },
}
input_test = {"x": x_star,"y":y_star}
label_test = {"u": u_star,"v":v_star,"p":p_star}
valida_dataloader_cfg = {
    "dataset": {
        "name": "NamedArrayDataset",
        "input":input_test,
        "label":label_test},
    'total_size':u_star.shape[0],
    "batch_size": u_star.shape[0],
    "sampler": {
        "name": "BatchSampler",
        "drop_last": False,
        "shuffle": False,
    },
}

geom = ppsci.geometry.PointCloud({"x": x_train,"y":y_train}, ("x","y"))

## supervised constraint s.t ||u-u_0||
sup_constraint = ppsci.constraint.SupervisedConstraint(
    train_dataloader_cfg,
    ppsci.loss.MSELoss("mean"),
    name="Sup",
)

# set equation constarint s.t. ||F(u)||


equation = {
        "NavierStokes": ppsci.equation.NavierStokes(nu=1.0/Re, rho=1.0, dim=2, time=False),
    }



pde_constraint = ppsci.constraint.InteriorConstraint(
    equation["NavierStokes"].equations,
    {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
    geom,
    {
        "dataset": {"name":"IterableNamedArrayDataset"},
        "batch_size": N_train,
        "iters_per_epoch": ITERS_PER_EPOCH,
    },
    ppsci.loss.MSELoss("mean"),
    name="EQ2",
)

constraint={sup_constraint.name:sup_constraint,pde_constraint.name:pde_constraint}

residual_validator = ppsci.validate.SupervisedValidator(
    valida_dataloader_cfg,
    ppsci.loss.L2RelLoss(),
    metric={"L2R": ppsci.metric.L2Rel()},
    name="Residual",
)
residual_validator_mse = ppsci.validate.SupervisedValidator(
    valida_dataloader_cfg,
    ppsci.loss.MSELoss("mean"),
    metric={"MSE": ppsci.metric.MSE()},
    name="Residual",
)

# Wrap validator 
validator = {residual_validator.name: residual_validator}

# set optimizer
#decay_epochs=[5000,10000,60000]
EPOCHS=5000
# lr = ppsci.optimizer.lr_scheduler.Piecewise(EPOCHS, ITERS_PER_EPOCH, decay_epochs, (1e-3, 1e-4,1e-5,1e-6))()
optimizer = ppsci.optimizer.Adam(1e-3)(model)

logger.init_logger("ppsci", f"{OUTPUT_DIR}/eval.log", "info")
# initialize solver
solver = ppsci.solver.Solver(
    model=model,
    constraint=constraint,
    optimizer=optimizer,
    epochs=EPOCHS,
    iters_per_epoch=ITERS_PER_EPOCH,
    eval_during_train=False,
    log_freq=5000,
    eval_freq=1000,
    seed=SEED,
    equation=equation,
    geom=geom,
    validator=validator,
    visualizer=None,
    eval_with_no_grad=False,
    output_dir='/home/aistudio/'
)
# train model
solver.train()
solver.eval()
solver.plot_loss_history()

# set optimizer

optimizer = ppsci.optimizer.Adam(1e-4)(model)

logger.init_logger("ppsci", f"{OUTPUT_DIR}/eval.log", "info")
# initialize solver
solver = ppsci.solver.Solver(
    model=model,
    constraint=constraint,
    optimizer=optimizer,
    epochs=EPOCHS,
    iters_per_epoch=ITERS_PER_EPOCH,
    eval_during_train=False,
    log_freq=2000,
    eval_freq=2000,
    seed=SEED,
    equation=equation,
    geom=geom,
    validator=validator,
    visualizer=None,
    eval_with_no_grad=False,
    output_dir='/home/aistudio/'
)
# train model
solver.train()
solver.eval()

#set optimizer
EPOCHS=50000
optimizer = ppsci.optimizer.Adam(1e-5)(model)

logger.init_logger("ppsci", f"{OUTPUT_DIR}/eval.log", "info")
# initialize solver
solver = ppsci.solver.Solver(
    model=model,
    constraint=constraint,
    optimizer=optimizer,
    epochs=EPOCHS,
    iters_per_epoch=ITERS_PER_EPOCH,
    eval_during_train=False,
    log_freq=2000,
    eval_freq=2000,
    seed=SEED,
    equation=equation,
    geom=geom,
    validator=validator,
    visualizer=None,
    eval_with_no_grad=False,
    output_dir='/home/aistudio/'
)
# train model
solver.train()
solver.eval()

#set optimizer
EPOCHS=50000
optimizer = ppsci.optimizer.Adam(1e-6)(model)

logger.init_logger("ppsci", f"{OUTPUT_DIR}/eval.log", "info")
# initialize solver
solver = ppsci.solver.Solver(
    model=model,
    constraint=constraint,
    optimizer=optimizer,
    epochs=EPOCHS,
    iters_per_epoch=ITERS_PER_EPOCH,
    eval_during_train=False,
    log_freq=2000,
    eval_freq=2000,
    seed=SEED,
    equation=equation,
    geom=geom,
    validator=validator,
    visualizer=None,
    eval_with_no_grad=False,
    output_dir='/home/aistudio/'
)
# train model
solver.train()
solver.eval()


# set optimizer
EPOCHS=5000
optimizer = ppsci.optimizer.LBFGS(max_iter=50000,tolerance_change=np.finfo(float).eps,history_size=50)(model)

logger.init_logger("ppsci", f"{OUTPUT_DIR}/eval.log", "info")
# initialize solver
solver = ppsci.solver.Solver(
    model=model,
    constraint=constraint,
    optimizer=optimizer,
    epochs=EPOCHS,
    iters_per_epoch=ITERS_PER_EPOCH,
    eval_during_train=False,
    log_freq=2000,
    eval_freq=2000,
    seed=SEED,
    equation=equation,
    geom=geom,
    validator=validator,
    visualizer=None,
    eval_with_no_grad=False,
    output_dir='/home/aistudio/'
)
# train model
solver.train()

# evaluate after finished training
solver.eval()