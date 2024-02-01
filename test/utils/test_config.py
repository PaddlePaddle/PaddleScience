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
import paddle
import pytest
from omegaconf import DictConfig

paddle.seed(1024)


@pytest.mark.parametrize(
    "epochs,mode,seed",
    [
        (-1, "train", 1024),
        (20, "wrong_mode", 1024),
        (10, "eval", -1),
    ],
)
def test_invalid_epochs(
    epochs,
    mode,
    seed,
):
    @hydra.main(version_base=None, config_path="./", config_name="test_config.yaml")
    def main(cfg: DictConfig):
        pass

    # sys.exit will be called when validation error in pydantic, so there we use
    # SystemExit instead of other type of errors.
    with pytest.raises(SystemExit):
        cfg_dict = dict(
            {
                "TRAIN": {
                    "epochs": epochs,
                },
                "mode": mode,
                "seed": seed,
                "hydra": {
                    "callbacks": {
                        "init_callback": {
                            "_target_": "ppsci.utils.callbacks.InitCallback"
                        }
                    }
                },
            }
        )
        # print(cfg_dict)
        import yaml

        with open("test_config.yaml", "w") as f:
            yaml.dump(dict(cfg_dict), f)

        main()


if __name__ == "__main__":
    pytest.main()
