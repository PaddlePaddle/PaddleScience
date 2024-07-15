import os

import hydra
import paddle
import pytest
import yaml

from ppsci.utils.callbacks import InitCallback

paddle.seed(1024)


@pytest.mark.parametrize(
    "epochs,mode,seed",
    [
        (-1, "train", 1024),
        (20, "wrong_mode", 1024),
        (10, "eval", -1),
    ],
)
def test_invalid_epochs(tmpdir, epochs, mode, seed):
    cfg_dict = {
        "hydra": {
            "callbacks": {
                "init_callback": {"_target_": "ppsci.utils.callbacks.InitCallback"}
            }
        },
        "mode": mode,
        "seed": seed,
        "TRAIN": {
            "epochs": epochs,
        },
    }

    dir_ = os.path.dirname(__file__)
    config_abs_path = os.path.join(dir_, "test_config.yaml")
    with open(config_abs_path, "w") as f:
        f.write(yaml.dump(cfg_dict))

    with hydra.initialize(config_path="./", version_base=None):
        cfg = hydra.compose(config_name="test_config.yaml")

        with pytest.raises(SystemExit) as exec_info:
            InitCallback().on_job_start(config=cfg)
        assert exec_info.value.code == 2


# 这部分通常不需要，除非你想直接从脚本运行测试
if __name__ == "__main__":
    pytest.main()
