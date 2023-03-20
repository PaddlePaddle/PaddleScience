# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import json
import argparse


def parse_args():
    """
    # 
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--speed_log_file', type=str, default="", help="speed_log_file")
    parser.add_argument('--AVG_CPU_USE', type=str, default="0", help='%')
    parser.add_argument('--MAX_GPU_MEMORY_USE', type=str, default='0', help='M')
    parser.add_argument('--AVG_GPU_USE', type=str, default='0', help='one card gpu use')
    args = parser.parse_args()
    # args.separator = None if args.separator == "None" else args.separator
    return args

if __name__ == "__main__":
    """
    """
    args = parse_args()
    with open(args.speed_log_file, "r") as read_file:
        data = json.load(read_file)
        print("origin_json:\n{}".format(json.dumps(data)))  
        data["AVG_CPU_USE"] = round(float(args.AVG_CPU_USE), 3)
        data["MAX_GPU_MEMORY_USE"] = int(args.MAX_GPU_MEMORY_USE)
        data["AVG_GPU_USE"] = round(float(args.AVG_GPU_USE), 3)
    
    print("new_json:\n{}".format(json.dumps(data)))  # it's required, for the log file path  insert to the database
    with open(args.speed_log_file, "w") as f:
        f.write(json.dumps(data))
