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

from typing import Dict
from typing import Union

import hydra
import numpy as np
import paddle
from omegaconf import DictConfig

from deploy.python_infer import base
from ppsci.utils import misc
from ppsci.utils import reader


class PINNPredictor(base.Predictor):
    def __init__(
        self,
        pdmodel_path: str,
        pdpiparams_path: str,
        *,
        use_gpu: bool = False,
        use_fp16: bool = False,
        use_int8: bool = False,
        use_tensorrt: bool = False,
        use_onnx: bool = False,
        onnx_path: str = None,
        ir_optim: bool = False,
        cpu_num_threads: int = 1,
        use_npu: bool = False,
        use_xpu: bool = False,
    ):
        super().__init__(
            pdmodel_path,
            pdpiparams_path,
            use_gpu=use_gpu,
            use_fp16=use_fp16,
            use_int8=use_int8,
            use_tensorrt=use_tensorrt,
            use_onnx=use_onnx,
            onnx_path=onnx_path,
            ir_optim=ir_optim,
            cpu_num_threads=cpu_num_threads,
            use_npu=use_npu,
            use_xpu=use_xpu,
        )

    def predict(
        self,
        input_dict: Dict[str, Union[np.ndarray, paddle.Tensor]],
        batch_size: int = 1024,
    ) -> Dict[str, np.ndarray]:
        # prepare input handle(s)
        input_handles = {
            name: self.predictor.get_input_handle(name) for name in input_dict
        }
        # prepare output handle(s)
        output_handles = {
            name: self.predictor.get_output_handle(name)
            for name in self.predictor.get_output_names()
        }

        num_samples = len(next(iter(input_dict.values())))
        num_pad = 0
        # pad with last element if `num_samples` is not divisible by `world_size`
        # ensuring every device get same number of data.
        if num_pad > 0:
            for k, v in input_dict.items():
                repeat_times = (num_pad, *(1 for _ in range(v.ndim - 1)))
                input_dict[k] = paddle.concat(
                    (
                        v,
                        paddle.tile(v[num_samples - 1 : num_samples], repeat_times),
                    ),
                )

        num_samples_pad = num_samples + num_pad
        local_num_samples_pad = num_samples_pad // 1
        local_input_dict = (
            {k: v[self.rank :: 1] for k, v in input_dict.items()}
            if 1 > 1
            else input_dict
        )
        local_batch_num = (local_num_samples_pad + (batch_size - 1)) // batch_size
        pred_dict = misc.Prettydefaultdict(list)

        for batch_id in range(local_batch_num):
            batch_input_dict = {}
            st = batch_id * batch_size
            ed = min(local_num_samples_pad, (batch_id + 1) * batch_size)

            # prepare batch input dict
            for key in local_input_dict:
                batch_input_dict[key] = local_input_dict[key][st:ed]

            # send batch input data to input handle(s)
            for name, handle in input_handles.items():
                handle.copy_from_cpu(batch_input_dict[name])

            # run predictor
            self.predictor.run()

            # receive batch output data from output handle(s)
            batch_output_dict = {
                name: output_handles[name].copy_to_cpu() for name in output_handles
            }

            # collect batch data
            for key, batch_output in batch_output_dict.items():
                pred_dict[key].append(batch_output)

        pred_dict = {key: np.concatenate(value) for key, value in pred_dict.items()}
        return pred_dict


@hydra.main(
    version_base=None,
    config_path="/workspace/hesensen/PaddleScience_setup/examples/aneurysm/conf",
    config_name="aneurysm.yaml",
)
def main(cfg: DictConfig):
    pinn_predictor = PINNPredictor(
        cfg.INFERENCE.pdmodel_path,
        cfg.INFERENCE.pdpiparams_path,
        use_gpu=True,
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
        "x": (eval_data_dict["x"] - cfg.CENTER[0]) * cfg.SCALE,
        "y": (eval_data_dict["y"] - cfg.CENTER[1]) * cfg.SCALE,
        "z": (eval_data_dict["z"] - cfg.CENTER[2]) * cfg.SCALE,
    }
    if "area" in input_dict.keys():
        input_dict["area"] *= cfg.SCALE**cfg.DIM

    output_dict = pinn_predictor.predict(input_dict)

    output_dict["u"] = output_dict["save_infer_model/scale_0.tmp_1"]
    output_dict["v"] = output_dict["save_infer_model/scale_1.tmp_1"]
    output_dict["w"] = output_dict["save_infer_model/scale_2.tmp_1"]
    output_dict["p"] = output_dict["save_infer_model/scale_3.tmp_1"]

    # for k, v in output_dict.items():
    #     print(k, v.shape)
    import ppsci

    ppsci.visualize.save_vtu_from_dict(
        "./aneurysm",
        {**input_dict, **output_dict},
        input_dict.keys(),
        ("u", "v", "w", "p"),
    )


if __name__ == "__main__":
    main()
