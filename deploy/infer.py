# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hydra
from omegaconf import DictConfig

import ppsci
from deploy.python_infer import pinn_predictor
from ppsci.utils import reader


@hydra.main(
    version_base=None,
    config_path="../examples/aneurysm/conf",
    config_name="aneurysm.yaml",
)
def main(cfg: DictConfig):
    predictor = pinn_predictor.PINNPredictor(
        cfg.INFER.pdmodel_path,
        cfg.INFER.pdpiparams_path,
        device=cfg.INFER.device,
        engine=cfg.INFER.engine,
        precision=cfg.INFER.precision,
        onnx_path=cfg.INFER.onnx_path,
        ir_optim=cfg.INFER.ir_optim,
        min_subgraph_size=cfg.INFER.min_subgraph_size,
        gpu_mem=cfg.INFER.gpu_mem,
        gpu_id=cfg.INFER.gpu_id,
        max_batch_size=cfg.INFER.max_batch_size,
        num_cpu_threads=cfg.INFER.num_cpu_threads,
    )
    eval_data_dict = reader.load_csv_file(
        cfg.EVAL_CSV_PATH,
        ("x", "y", "z", "u", "v", "w", "p"),
        {
            "x": "Points:0",
            "y": "Points:1",
            "z": "Points:2",
            "u": "U:0",
            "v": "U:1",
            "w": "U:2",
            "p": "p",
        },
    )
    input_dict = {
        "x": (eval_data_dict["x"] - cfg.CENTER[0])[:1024] * cfg.SCALE,
        "y": (eval_data_dict["y"] - cfg.CENTER[1])[:1024] * cfg.SCALE,
        "z": (eval_data_dict["z"] - cfg.CENTER[2])[:1024] * cfg.SCALE,
    }
    if "area" in input_dict.keys():
        input_dict["area"] *= cfg.SCALE**cfg.DIM

    output_dict = predictor.predict(input_dict, cfg.INFER.batch_size)

    output_dict["u"] = output_dict.pop("save_infer_model/scale_0.tmp_1")
    output_dict["v"] = output_dict.pop("save_infer_model/scale_1.tmp_1")
    output_dict["w"] = output_dict.pop("save_infer_model/scale_2.tmp_1")
    output_dict["p"] = output_dict.pop("save_infer_model/scale_3.tmp_1")

    ppsci.visualize.save_vtu_from_dict(
        "./aneurysm",
        {**input_dict, **output_dict},
        input_dict.keys(),
        ("u", "v", "w", "p"),
    )


if __name__ == "__main__":
    main()

"""
paddle2onnx --model_dir=./ \
    --model_filename=aneurysm.pdmodel \
    --params_filename=aneurysm.pdiparams \
    --save_file=./aneurysm.onnx \
    --opset_version=10 \
    --enable_onnx_checker=True
"""
