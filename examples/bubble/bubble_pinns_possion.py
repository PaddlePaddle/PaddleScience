import numpy as np
import paddle
import scipy.io

import ppsci
from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.equation.pde import base
from ppsci.utils import config
from ppsci.utils import logger
from ppsci.utils import reader


# 定义压力泊松方程
class pressure_Poisson_equation(base.PDE):
    def __init__(self, rho_1, rho_2, mu_1, mu_2: float, dim: int, time: bool):
        super().__init__()
        self.mu_1 = mu_1
        self.rho_1 = rho_1
        self.mu_2 = mu_2
        self.rho_2 = rho_2
        self.dim = dim
        self.time = time

        def pressure_Poisson_func(out):
            x, y = out["x"], out["y"]
            u, v, p, phil = out["u"], out["v"], out["p"], out["phil"]
            rho = self.rho_1 + phil * (self.rho_2 - self.rho_1)
            # u = jacobian(psi,y)
            # v = -jacobian(psi,x)
            duiliu = 2 * (
                jacobian(u, x) * jacobian(v, y) - jacobian(u, y) * jacobian(v, x)
            )
            pressure_Poisson = (
                # duiliu -
                hessian(p, y)
                + hessian(p, x)
            )
            return pressure_Poisson

        self.add_equation("pressure_Poisson", pressure_Poisson_func)


if __name__ == "__main__":
    args = config.parse_args()
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)
    # set output directory
    OUTPUT_DIR = "./output_bubble_pinns_p" if not args.output_dir else args.output_dir
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    DATASET_PATH = "/home/project/test/bubble_train4.mat"
    DATASET_PATH_VALID = "/home/project/test/bubble_test4.mat"

    # set model
    model_psi = ppsci.arch.MLP(("x", "y", "t"), ("psi",), 9, 30, "tanh", False, False)
    model_p = ppsci.arch.MLP(("x", "y", "t"), ("p",), 9, 30, "tanh", False, False)
    model_phil = ppsci.arch.MLP(("x", "y", "t"), ("phil",), 9, 30, "tanh", False, False)

    def transform_in(_in):
        global input_dict
        input_dict = _in
        return _in

    def transform_out(out):
        psi_y = out["psi"]
        y = input_dict["y"]
        x = input_dict["x"]
        u_out = jacobian(psi_y, y)
        v_out = -jacobian(psi_y, x)
        return {"u": u_out, "v": v_out}

    model_psi.register_input_transform(transform_in)
    model_psi.register_output_transform(transform_out)
    model_list = ppsci.arch.ModelList((model_psi, model_p, model_phil))

    # set equation
    equation = {
        "pressure_Poisson_equation": pressure_Poisson_equation(
            958, 0.25, 5.5, 0.0002, 2, True
        )
    }

    # set time-geometry
    # set timestamps(including initial t0)
    timestamps = np.linspace(0, 126, 127, endpoint=True)
    geom = {
        "time_rect": ppsci.geometry.PointCloud(
            reader.load_mat_file(
                DATASET_PATH,
                ("t", "x", "y"),
            ),
            ("t", "x", "y"),
        ),
        "time_rect_eval": ppsci.geometry.TimeXGeometry(
            ppsci.geometry.TimeDomain(1, 126, timestamps=timestamps),
            ppsci.geometry.Rectangle((0, 0), (15, 5)),
        ),
        # "time_rect_eval": ppsci.geometry.PointCloud(
        # reader.load_mat_file(
        #    DATASET_PATH_VALID,
        #    ("t", "x", "y"),
        # ),
        # ("t", "x", "y"),
        # ),
    }

    # set dataloader config
    ITERS_PER_EPOCH = 1
    # set timestamps(including initial t0)
    timestamps = np.linspace(1, 126, 126, endpoint=True)
    train_dataloader_cfg = {
        "dataset": {
            "name": "MatDataset",
            "file_path": DATASET_PATH,
            "input_keys": ("x", "t", "y"),
            "label_keys": ("u", "v", "p", "phil"),
            # "weight_dict": {"eta": 100},
            "timestamps": timestamps,
        },
        "batch_size": 2419,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
    }

    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["pressure_Poisson_equation"].equations,
        {"pressure_Poisson": 0},
        geom["time_rect"],
        {
            "dataset": "IterableNamedArrayDataset",
            "batch_size": 228595,
            "iters_per_epoch": ITERS_PER_EPOCH,
        },
        ppsci.loss.MSELoss("sum"),
        name="EQ",
    )
    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        ppsci.loss.MSELoss("sum"),
        name="Sup",
    )

    # wrap constraints together
    constraint = {
        sup_constraint.name: sup_constraint,
        pde_constraint.name: pde_constraint,
    }

    # set training hyper-parameters
    EPOCHS = 10000
    EVAL_FREQ = 1000
    # set optimizer
    optimizer = ppsci.optimizer.Adam(0.001)((model_psi, model_p, model_phil))

    # set validator
    valida_dataloader_cfg = {
        "dataset": {
            "name": "MatDataset",
            "file_path": DATASET_PATH_VALID,
            "input_keys": ("t", "x", "y"),
            "label_keys": ("u", "v", "p", "phil"),
        },
        "batch_size": 100,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }
    eta_mse_validator = ppsci.validate.SupervisedValidator(
        valida_dataloader_cfg,
        ppsci.loss.MSELoss("mean"),
        metric={"MSE": ppsci.metric.MSE()},
        name="eta_mse",
    )
    validator = {
        eta_mse_validator.name: eta_mse_validator,
    }

    visu_mat = ppsci.utils.reader.load_mat_file(
        DATASET_PATH_VALID,
        ("x", "y", "t"),  # "u",'v','p', "phil"),
    )
    # visu_mat = geom["time_rect_eval"].sample_interior(
    #     300*100*126,
    #     evenly=True
    # )

    datafile = "/home/project/test/norm.mat"
    data = scipy.io.loadmat(datafile)
    u_max = data["u_max"][0][0]
    u_min = data["u_min"][0][0]
    v_max = data["v_max"][0][0]
    v_min = data["v_min"][0][0]
    p_max = data["p_max"][0][0]
    p_min = data["p_min"][0][0]
    phil_max = data["phil_max"][0][0]
    phil_min = data["phil_min"][0][0]
    visualizer = {
        "visulzie_u_v_p": ppsci.visualize.VisualizerVtu(
            visu_mat,
            # vis_points,
            {
                "u": lambda d: d["u"] * (u_max - u_min) + u_min,
                "v": lambda d: d["v"] * (v_max - v_min) + v_min,
                "p": lambda d: d["p"] * (p_max - p_min) + p_min,
                "phil": lambda d: d["phil"],
            },  # *(phil_max-phil_min)+phil_min},
            # num_timestamps=126,
            prefix="result_u_v_p",
        )
    }

    solver = ppsci.solver.Solver(
        model_list,
        constraint,
        OUTPUT_DIR,
        optimizer,
        None,
        EPOCHS,
        ITERS_PER_EPOCH,
        eval_during_train=True,
        eval_freq=EVAL_FREQ,
        equation=equation,
        geom=geom,
        validator=validator,
        visualizer=visualizer,
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()
    # visualize prediction after finished training
    solver.visualize()

    # directly evaluate pretrained model(optional)
    solver = ppsci.solver.Solver(
        model_list,
        constraint,
        OUTPUT_DIR,
        equation=equation,
        geom=geom,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=f"{OUTPUT_DIR}/checkpoints/latest",
    )
    solver.eval()
    # visualize prediction for pretrained model(optional)
    solver.visualize()
