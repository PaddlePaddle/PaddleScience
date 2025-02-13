import hydra
from angle_model import eval
from angle_model import train
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="./config", config_name="angle.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        eval(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
