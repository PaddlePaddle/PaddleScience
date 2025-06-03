import hydra
from f_model import train, eval, load_data, featurize_molecules
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="./config", config_name="f.yaml")
def main(cfg: DictConfig):
    
    data, smis = load_data(cfg)
    
    X = featurize_molecules(smis)
    
    if cfg.mode == "train":
        train(cfg, X, data)
    elif cfg.mode == "eval":
        eval(cfg, X, data)
    else:
        raise ValueError(f"cfg.mode should be 'train'or 'eval', but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
