# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# refs: https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/external_aerodynamics/domino

import os
import time

import numpy as np


def process_files(*args_list):
    ids = args_list[0]
    processor_id = args_list[1]
    fm_data = args_list[2]
    output_dir = args_list[3]
    for j in ids:
        fname = fm_data.filenames[j]
        if len(os.listdir(os.path.join(fm_data.data_path, fname))) == 0:
            print(f"Skipping {fname} - empty.")
            continue
        outname = os.path.join(output_dir, fname)
        print("Filename:%s on processor: %d" % (outname, processor_id))
        filename = f"{outname}.npy"
        if os.path.exists(filename):
            print(f"Skipping {filename} - already exists.")
            continue
        start_time = time.time()
        data_dict = fm_data[j]
        np.save(filename, data_dict)
        print("Time taken for %d = %f" % (j, time.time() - start_time))
