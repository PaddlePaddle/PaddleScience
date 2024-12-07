# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reference: https://github.com/omron-sinicx/transformer4sr
"""

import hydra
import numpy as np
import paddle
from functions_data import DataFuncs
from functions_data import SRSDDataFuncs
from functions_loss_metric import compute_inaccuracy
from functions_loss_metric import cross_entropy_loss_func
from functions_vis import VisualizeFuncs
from omegaconf import DictConfig
from tqdm import tqdm
from utils import compute_norm_zss_dist
from utils import is_tree_complete
from utils import simplify_output

import ppsci


def train(cfg: DictConfig):
    # data
    data_funcs = DataFuncs(
        cfg.DATA.data_path,
        cfg.DATA.vocab_library,
        cfg.DATA.seq_length_max,
        cfg.DATA.ratio,
        shuffle=True,
    )

    # set model
    num_var_max = len(cfg.DATA.response_variable)
    vocab_size = len(cfg.DATA.vocab_library) + 2
    model = ppsci.arch.Transformer(
        **cfg.MODEL,
        num_var_max=num_var_max,
        vocab_size=vocab_size,
        seq_length=data_funcs.seq_length_max,
    )

    # set optimizer
    def lr_lambda(step, d_model=cfg.MODEL.d_model, warmup=cfg.TRAIN.lr_warmup):
        if step == 0:
            step = 1
        lr = d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
        return lr

    lr_scheduler = ppsci.optimizer.lr_scheduler.LambdaDecay(
        **cfg.TRAIN.lr_scheduler,
        lr_lambda=lr_lambda,
    )()
    optimizer = ppsci.optimizer.Adam(lr_scheduler, **cfg.TRAIN.adam)(model)

    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": {
                    "input": data_funcs.values_train.astype(paddle.get_default_dtype()),
                    "target_seq": data_funcs.targets_train[:, :-1],
                },
                "label": {"output": data_funcs.targets_train[:, 1:]},
            },
            "batch_size": cfg.TRAIN.batch_size,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": True,
            },
            "num_workers": 1,
        },
        ppsci.loss.FunctionalLoss(cross_entropy_loss_func),
        name="sup_constraint",
    )

    # wrap constraints together
    constraint = {sup_constraint.name: sup_constraint}

    # set validator
    sup_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": {
                    "input": data_funcs.values_val.astype(paddle.get_default_dtype()),
                    "target_seq": data_funcs.targets_val[:, :-1],
                },
                "label": {"output": data_funcs.targets_val[:, 1:]},
            },
            "batch_size": cfg.TRAIN.batch_size,
            "num_workers": 1,
        },
        ppsci.loss.FunctionalLoss(cross_entropy_loss_func),
        metric={"metric": ppsci.metric.FunctionalMetric(compute_inaccuracy)},
        name="sup_validator",
    )

    # wrap validator together
    validator = {sup_validator.name: sup_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        optimizer=optimizer,
        validator=validator,
        cfg=cfg,
    )

    # train model
    solver.train()

    # evaluate after finished training
    solver.eval()


def evaluate(cfg: DictConfig):
    # data
    data_funcs = SRSDDataFuncs(
        cfg.DATA.data_path_srsd,
        cfg.DATA.sampling_times,
        cfg.DATA.response_variable,
        cfg.DATA.vocab_library,
        cfg.DATA.seq_length_max,
        shuffle=True,
    )

    # set model
    num_var_max = len(cfg.DATA.response_variable)
    vocab_size = len(cfg.DATA.vocab_library) + 2
    model = ppsci.arch.Transformer(
        **cfg.MODEL,
        num_var_max=num_var_max,
        vocab_size=vocab_size,
        seq_length=data_funcs.seq_length_max,
    )
    ppsci.utils.save_load.load_pretrain(model, path=cfg.EVAL.pretrained_model_path)
    model.eval()

    # evaluate
    num_repeat = cfg.EVAL.num_repeat if isinstance(data_funcs, SRSDDataFuncs) else 1
    num_samples = data_funcs.values_test.shape[0]
    zss_dist = np.zeros((num_repeat, num_samples))
    for i in tqdm(range(num_repeat), desc="Evaluating"):
        encoder_input = paddle.to_tensor(
            data_funcs.values_test, dtype=paddle.get_default_dtype()
        )
        preds = model.decode_process(encoder_input, is_tree_complete)
        labels = paddle.to_tensor(data_funcs.targets_test)

        for j in range(num_samples):
            try:
                pred_simplify = simplify_output(preds[j], "tensor")
                zss_dist[i][j] = compute_norm_zss_dist(pred_simplify[0], labels[j])
            except Exception:
                zss_dist[i][j] = np.nan

        if i != num_repeat - 1:
            # reload data to increase randomness
            data_funcs.init_data("test")

    zss_dist_mean = np.nanmean(zss_dist, axis=0)
    zss_dist_std = np.nanstd(zss_dist, axis=0)
    zss_dist_min = np.nanmin(zss_dist, axis=0)
    zss_dist_max = np.nanmax(zss_dist, axis=0)

    try:
        keys = data_funcs.keys_test
        assert len(keys) == num_samples
    except Exception:
        keys = [f"sample_{i}" for i in range(num_samples)]

    print(
        f"zss_distance and accuracy in {num_repeat} attempts of {num_samples} samples with format: name => mean +- std | min ~ max"
    )
    for i in range(num_samples):
        key = keys[i]
        print(
            f"{key} => {zss_dist_mean[i]:.3f} +- {zss_dist_std[i]:.3f} | {zss_dist_min[i]:.3f} ~ {zss_dist_max[i]:.3f}"
        )

    print("-----------")
    print(
        f"=> Mean ZSS distance: {np.nanmean(zss_dist):.3f} +- {np.nanstd(zss_dist):.3f}"
    )
    print(f"=> Hit rate: {np.sum(np.any(zss_dist==0, axis=0))}/{zss_dist.shape[1]}")

    # visualize prediction
    visualizer = VisualizeFuncs(model)
    visualizer.visualize_valid_data(data_funcs.targets_test, data_funcs.values_test, 10)
    visualizer.visualize_demo()


def export(cfg: DictConfig):
    def temporary_complete_func(seq_indices):
        ".utils.is_tree_complete is not work in static gragh now."
        arity = 1
        for n in seq_indices:
            n = n.item()
            if n == 0 or n == 1:
                continue
                print("Predict padding or <SOS>, which is bad...")
            if n == 2 or n == 3:
                arity = arity + 2 - 1
            elif n in range(4, 13):
                arity = arity + 1 - 1
            elif n in range(13, 20):
                arity = arity + 0 - 1
        if arity == 0:
            return True
        else:
            return False

    class WarppedModel(ppsci.arch.Transformer):
        def __init__(self, *args, complete_func, **kwargs):
            super().__init__(*args, **kwargs)
            self.complete_func = complete_func

        def forward(self, x):
            return {"output": self.decode_process(x["input"], self.complete_func)}

    # set model
    num_var_max = len(cfg.DATA.response_variable)
    vocab_size = len(cfg.DATA.vocab_library) + 2
    warpped_model = WarppedModel(
        **cfg.MODEL,
        num_var_max=num_var_max,
        vocab_size=vocab_size,
        seq_length_max=cfg.DATA.seq_length_max,
        complete_func=temporary_complete_func,
    )
    warpped_model.eval()

    # initialize solver
    solver = ppsci.solver.Solver(
        warpped_model,
        pretrained_model_path=cfg.INFER.pretrained_model_path,
    )

    # export model
    from paddle.static import InputSpec

    input_spec = [
        {
            "input": InputSpec(
                [None, cfg.DATA.sampling_times, len(cfg.DATA.response_variable), 1],
                "float32",
                name="input",
            )
        }
    ]
    solver.export(input_spec, cfg.INFER.export_path)


def inference(cfg: DictConfig):
    import sympy

    from deploy.python_infer import pinn_predictor

    predictor = pinn_predictor.PINNPredictor(cfg)

    C, y, x1, x2, x3, x4, x5, x6 = sympy.symbols(
        "C, y, x1, x2, x3, x4, x5, x6", real=True, positive=True
    )
    y = 25 * x1 + x2 * sympy.log(x1)
    print("The ground truth is:", y)

    x1_values = np.power(10.0, np.random.uniform(-1.0, 1.0, size=50))
    x2_values = np.power(10.0, np.random.uniform(-1.0, 1.0, size=50))
    f = sympy.lambdify([x1, x2], y)
    y_values = f(x1_values, x2_values)
    dataset = np.zeros((50, 7))
    dataset[:, 0] = y_values
    dataset[:, 1] = x1_values
    dataset[:, 2] = x2_values
    encoder_input = dataset[np.newaxis, :, :, np.newaxis].astype(np.float32)
    output_dict = predictor.predict({"input": encoder_input}, cfg.INFER.batch_size)
    output_dict = {
        store_key: output_dict[infer_key]
        for store_key, infer_key in zip(("output",), output_dict.keys())
    }
    sympy_pred = simplify_output(output_dict["output"][0], "sympy")
    print("The prediction is:", sympy_pred)


@hydra.main(version_base=None, config_path="./conf", config_name="transformer4sr.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    elif cfg.mode == "export":
        export(cfg)
    elif cfg.mode == "infer":
        inference(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
