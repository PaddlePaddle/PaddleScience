import numpy as np

import ppsci
from ppsci.utils import logger

# set random seed(42) for reproducibility
ppsci.utils.misc.set_random_seed(42)

# set output directory
OUTPUT_DIR = "./output_quick_start_case1"

# initialize logger while create output directory
logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

# set 1D-geometry domain([-π, π])
l_limit, r_limit = -np.pi, np.pi
x_domain = ppsci.geometry.Interval(l_limit, r_limit)

# set model to 3-layer MLP
model = ppsci.arch.MLP(("x",), ("u",), 3, 64)

# standard solution of sin(x)
def sin_compute_func(data: dict):
    return np.sin(data["x"])


# set constraint on 1D-geometry([-π, π])
ITERS_PER_EPOCH = 100  # use 100 iterations per training epoch
interior_constraint = ppsci.constraint.InteriorConstraint(
    output_expr={"u": lambda out: out["u"]},
    label_dict={"u": sin_compute_func},
    geom=x_domain,
    dataloader_cfg={
        "dataset": "NamedArrayDataset",
        "iters_per_epoch": ITERS_PER_EPOCH,
        "sampler": {
            "name": "BatchSampler",
            "shuffle": True,
        },
        "batch_size": 32,  # use 32 samples(points) per iteration for interior constraint
    },
    loss=ppsci.loss.MSELoss(),
)
# wrap constraint(s) into one dict
constraint = {
    interior_constraint.name: interior_constraint,
}

# set training hyper-parameters
EPOCHS = 10
# set optimizer
optimizer = ppsci.optimizer.Adam(2e-3)(model)

# set visualizer
visual_input_dict = {
    "x": np.linspace(l_limit, r_limit, 1000, dtype="float32").reshape(1000, 1)
}
visual_input_dict["u_ref"] = np.sin(visual_input_dict["x"])
visualizer = {
    "visualize_u": ppsci.visualize.VisualizerScatter1D(
        visual_input_dict,
        ("x",),
        {"u_pred": lambda out: out["u"], "u_ref": lambda out: out["u_ref"]},
        prefix="u=sin(x)",
    ),
}

# initialize solver
solver = ppsci.solver.Solver(
    model,
    constraint,
    OUTPUT_DIR,
    optimizer,
    epochs=EPOCHS,
    iters_per_epoch=ITERS_PER_EPOCH,
    visualizer=visualizer,
)
# train model
solver.train()

# compute l2-relative error of trained model
pred_u = solver.predict(visual_input_dict, return_numpy=True)["u"]
l2_rel = np.linalg.norm(pred_u - visual_input_dict["u_ref"]) / np.linalg.norm(
    visual_input_dict["u_ref"]
)
logger.info(f"l2_rel = {l2_rel:.5f}")

# visualize prediction after finished training
solver.visualize()
