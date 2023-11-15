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

import platform
from os import path as osp

from paddle import inference as pd_inference


class Predictor(object):
    def __init__(
        self,
        pdmodel_path: str,
        pdpiparams_path: str,
        *,
        use_gpu: bool,
        use_fp16: bool,
        use_int8: bool,
        use_tensorrt: bool,
        use_onnx: bool = False,
        onnx_path: str = None,
        ir_optim: bool = False,
        cpu_num_threads: int = 1,
        use_npu: bool = False,
        use_xpu: bool = False,
    ):
        self.pdmodel_path = pdmodel_path
        self.pdpiparams_path = pdpiparams_path
        self.use_gpu = use_gpu
        self.use_fp16 = use_fp16
        self.use_int8 = use_int8
        self.use_tensorrt = use_tensorrt
        self.use_onnx = use_onnx
        self.onnx_path = onnx_path
        self.ir_optim = ir_optim
        self.cpu_num_threads = cpu_num_threads
        self.use_npu = use_npu
        self.use_xpu = use_xpu

        if self.use_fp16 is True:
            if not self.use_tensorrt:
                raise ValueError(
                    "fp16 mode is only available with tensorrt enabled, please set "
                    "'use_tensorrt' as True during inference."
                )
            if self.use_int8:
                raise ValueError(
                    "fp16 mode is not available with int8 enabled, please set "
                    "'use_int8' as False during inference."
                )

        if self.use_onnx:
            self.predictor, self.config = self._create_onnx_predictor()
        else:
            self.predictor, self.config = self._create_paddle_predictor()

    def predict(self, image):
        raise NotImplementedError

    def _create_paddle_predictor(self):
        config = pd_inference.Config(self.pdmodel_path, self.pdpiparams_path)
        if self.use_gpu:
            config.enable_use_gpu(4000, 0)
        # elif self.use_npu:
        #     config.enable_custom_device("npu")
        # elif self.use_xpu:
        #     config.enable_xpu()
        else:
            config.disable_gpu()
            if self.enable_mkldnn:
                # there is no set_mkldnn_cache_capatity() on macOS
                if platform.system() != "Darwin":
                    # cache 10 different shapes for mkldnn to avoid memory leak
                    config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
        config.set_cpu_math_library_num_threads(self.cpu_num_threads)

        config.disable_glog_info()
        config.switch_ir_optim(self.ir_optim)  # default true

        if self.use_tensorrt:
            precision = pd_inference.Config.Precision.Float32
            if self.use_int8:
                precision = pd_inference.Config.Precision.Int8
            elif self.use_fp16:
                precision = pd_inference.Config.Precision.Half

            config.enable_tensorrt_engine(
                precision_mode=precision,
                max_batch_size=self.batch_size,
                workspace_size=1 << 30,
                min_subgraph_size=30,
                use_calib_mode=False,
            )

        config.enable_memory_optim()
        # enable zero copy
        config.switch_use_feed_fetch_ops(False)

        # instantiate paddle predictor
        predictor = pd_inference.create_predictor(config)
        return predictor, config

    def _create_onnx_predictor(self):
        if self.onnx_path is None:
            raise ValueError(
                "onnx_path is not provided, please set "
                "'use_onnx' to True and provide 'onnx_path' during inference."
            )
        if not osp.exists(self.onnx_path):
            raise FileNotFoundError(
                f"Given 'onnx_path' {self.onnx_path} does not exist. "
                "Please check if it is correct."
            )
        if self.use_gpu:
            raise RuntimeError(
                "Onnx runtime does not support GPU inference, "
                "please set 'use_gpu' to False during inference."
            )

        try:
            import onnxruntime as ort
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install onnxruntime with `pip install onnxruntime`."
            )

        # set config for onnx predictor
        config = ort.SessionOptions()
        config.intra_op_num_threads = self.cpu_num_threads
        if self.ir_optim:
            config.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # instantiate onnx predictor
        predictor = ort.InferenceSession(self.onnx_path, sess_options=config)
        return predictor, config
