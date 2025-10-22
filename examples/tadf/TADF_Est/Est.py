import hydra
from Est_model import evaluate
from Est_model import featurize_molecules
from Est_model import load_data
from Est_model import train
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="./config", config_name="est.yaml")
def main(cfg: DictConfig):
    data, smis = load_data(cfg)
    X = featurize_molecules(smis)
    if cfg.mode == "train":
        train(cfg, X, data)
    elif cfg.mode == "eval":
        evaluate(cfg, X, data)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
