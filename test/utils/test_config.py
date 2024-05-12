import os

import hydra
import paddle
import pytest
import yaml

# 假设你的回调类在这个路径下
from ppsci.utils.callbacks import InitCallback

# 设置 Paddle 的 seed
paddle.seed(1024)

# 测试函数不需要装饰器
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
    # 创建一个临时的配置文件
    dir_ = os.path.dirname(__file__)
    config_abs_path = os.path.join(dir_, "test_config.yaml")
    with open(config_abs_path, "w") as f:
        f.write(yaml.dump(cfg_dict))

    # 使用 hydra 的 compose API 来创建配置，而不是使用 main
    with hydra.initialize(config_path="./", version_base=None):
        cfg = hydra.compose(config_name="test_config.yaml")
        # 手动触发回调
        with pytest.raises(SystemExit) as exec_info:
            InitCallback().on_job_start(config=cfg)
        assert exec_info.value.code == 2
        # 你现在可以根据需要对 cfg 进行断言或进一步处理


# 这部分通常不需要，除非你想直接从脚本运行测试
if __name__ == "__main__":
    pytest.main()
