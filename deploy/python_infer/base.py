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

from __future__ import annotations

import importlib
import platform
from os import path as osp
from typing import TYPE_CHECKING
from typing import Optional
from typing import Tuple

from paddle import inference as paddle_inference
from typing_extensions import Literal

from ppsci.utils import logger

if TYPE_CHECKING:
    import onnxruntime


class Predictor:
    """
    Initializes the inference engine with the given parameters.

    Args:
        pdmodel_path (Optional[str]): Path to the PaddlePaddle model file. Defaults to None.
        pdpiparams_path (Optional[str]): Path to the PaddlePaddle model parameters file. Defaults to None.
        device (Literal["gpu", "cpu", "npu", "xpu"], optional): Device to use for inference. Defaults to "cpu".
        engine (Literal["native", "tensorrt", "onnx", "mkldnn"], optional): Inference engine to use. Defaults to "native".
        precision (Literal["fp32", "fp16", "int8"], optional): Precision to use for inference. Defaults to "fp32".
        onnx_path (Optional[str], optional): Path to the ONNX model file. Defaults to None.
        ir_optim (bool, optional): Whether to use IR optimization. Defaults to True.
        min_subgraph_size (int, optional): Minimum subgraph size for IR optimization. Defaults to 15.
        gpu_mem (int, optional): Initial size of GPU memory pool(MB). Defaults to 500(MB).
        gpu_id (int, optional): GPU ID to use. Defaults to 0.
        num_cpu_threads (int, optional): Number of CPU threads to use. Defaults to 1.
    """

    def __init__(
        self,
        pdmodel_path: Optional[str] = None,
        pdpiparams_path: Optional[str] = None,
        *,
        device: Literal["gpu", "cpu", "npu", "xpu"] = "cpu",
        engine: Literal["native", "tensorrt", "onnx", "mkldnn"] = "native",
        precision: Literal["fp32", "fp16", "int8"] = "fp32",
        onnx_path: Optional[str] = None,
        ir_optim: bool = True,
        min_subgraph_size: int = 15,
        gpu_mem: int = 500,
        gpu_id: int = 0,
        max_batch_size: int = 10,
        num_cpu_threads: int = 10,
    ):
        self.pdmodel_path = pdmodel_path
        self.pdpiparams_path = pdpiparams_path

        self._check_device(device)
        self.device = device
        self._check_engine(engine)
        self.engine = engine
        self._check_precision(precision)
        self.precision = precision
        self._compatibility_check()

        self.onnx_path = onnx_path
        self.ir_optim = ir_optim
        self.min_subgraph_size = min_subgraph_size
        self.gpu_mem = gpu_mem
        self.gpu_id = gpu_id
        self.max_batch_size = max_batch_size
        self.num_cpu_threads = num_cpu_threads

        if self.engine == "onnx":
            self.predictor, self.config = self._create_onnx_predictor()
        else:
            self.predictor, self.config = self._create_paddle_predictor()

        logger.message(
            f"Inference with engine: {self.engine}, precision: {self.precision}, "
            f"device: {self.device}."
        )

    def predict(self, input_dict):
        raise NotImplementedError

    def _create_paddle_predictor(
        self,
    ) -> Tuple[paddle_inference.Predictor, paddle_inference.Config]:
        if not osp.exists(self.pdmodel_path):
            raise FileNotFoundError(
                f"Given 'pdmodel_path': {self.pdmodel_path} does not exist. "
                "Please check if it is correct."
            )
        if not osp.exists(self.pdpiparams_path):
            raise FileNotFoundError(
                f"Given 'pdpiparams_path': {self.pdpiparams_path} does not exist. "
                "Please check if it is correct."
            )

        config = paddle_inference.Config(self.pdmodel_path, self.pdpiparams_path)
        if self.device == "gpu":
            config.enable_use_gpu(self.gpu_mem, self.gpu_id)
            if self.engine == "tensorrt":
                if self.precision == "fp16":
                    precision = paddle_inference.Config.Precision.Half
                elif self.precision == "int8":
                    precision = paddle_inference.Config.Precision.Int8
                else:
                    precision = paddle_inference.Config.Precision.Float32
                config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    precision_mode=precision,
                    max_batch_size=self.max_batch_size,
                    min_subgraph_size=self.min_subgraph_size,
                    use_calib_mode=False,
                )
                # collect shape
                pdmodel_dir = osp.dirname(self.pdmodel_path)
                trt_shape_path = osp.join(pdmodel_dir, "trt_dynamic_shape.txt")

                if not osp.exists(trt_shape_path):
                    config.collect_shape_range_info(trt_shape_path)
                    logger.message(
                        f"Save collected dynamic shape info to: {trt_shape_path}"
                    )
                try:
                    config.enable_tuned_tensorrt_dynamic_shape(trt_shape_path, True)
                except Exception as e:
                    logger.warning(e)
                    logger.warning(
                        "TRT dynamic shape is disabled for your paddlepaddle < 2.3.0"
                    )

        elif self.device == "npu":
            config.enable_custom_device("npu")
        elif self.device == "xpu":
            config.enable_xpu(10 * 1024 * 1024)
        else:
            config.disable_gpu()
            if self.engine == "mkldnn":
                # 'set_mkldnn_cache_capatity' is not available on macOS
                if platform.system() != "Darwin":
                    ...
                    # cache 10 different shapes for mkldnn to avoid memory leak
                    # config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()

                if self.precision == "fp16":
                    config.enable_mkldnn_bfloat16()

                config.set_cpu_math_library_num_threads(self.num_cpu_threads)

        # enable memory optim
        config.enable_memory_optim()
        # config.disable_glog_info()
        # enable zero copy
        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(self.ir_optim)

        predictor = paddle_inference.create_predictor(config)
        return predictor, config

    def _create_onnx_predictor(
        self,
    ) -> Tuple["onnxruntime.InferenceSession", "onnxruntime.SessionOptions"]:
        if not osp.exists(self.onnx_path):
            raise FileNotFoundError(
                f"Given 'onnx_path' {self.onnx_path} does not exist. "
                "Please check if it is correct."
            )

        try:
            import onnxruntime as ort
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install onnxruntime with `pip install onnxruntime`."
            )

        # set config for onnx predictor
        config = ort.SessionOptions()
        config.intra_op_num_threads = self.num_cpu_threads
        if self.ir_optim:
            config.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # instantiate onnx predictor
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device != "cpu"
            else ["CPUExecutionProvider"]
        )
        predictor = ort.InferenceSession(
            self.onnx_path, sess_options=config, providers=providers
        )
        return predictor, config

    def _check_device(self, device: str):
        if device not in ["gpu", "cpu", "npu", "xpu"]:
            raise ValueError(
                "Inference only supports 'gpu', 'cpu', 'npu' and 'xpu' devices, "
                f"but got {device}."
            )

    def _check_engine(self, engine: str):
        if engine not in ["native", "tensorrt", "onnx", "mkldnn"]:
            raise ValueError(
                "Inference only supports 'native', 'tensorrt', 'onnx' and 'mkldnn' "
                f"engines, but got {engine}."
            )

    def _check_precision(self, precision: str):
        if precision not in ["fp32", "fp16", "int8"]:
            raise ValueError(
                "Inference only supports 'fp32', 'fp16' and 'int8' "
                f"precision, but got {precision}."
            )

    def _compatibility_check(self):
        if self.engine == "onnx":
            if not (
                importlib.util.find_spec("onnxruntime")
                or importlib.util.find_spec("onnxruntime-gpu")
            ):
                raise ModuleNotFoundError(
                    "\nPlease install onnxruntime first when engine is 'onnx'\n"
                    "* For CPU inference, use `pip install onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple`\n"
                    "* For GPU inference, use `pip install onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple`"
                )
            import onnxruntime as ort

            if self.device == "gpu" and ort.get_device() != "GPU":
                raise RuntimeError(
                    "Please install onnxruntime-gpu with `pip install onnxruntime-gpu`"
                    " when device is set to 'gpu'\n"
                )
