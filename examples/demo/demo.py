# GAOT (Geometry-Aware Physics-informed Neural Transport)
# to reproduce the GAOT Poisson--Gauss benchmark results in the paper with ppsci framework

import ppsci

def train(cfg: DictConfig):
    pass

def evaluate(cfg: DictConfig):
    pass

def export(cfg: DictConfig):
    pass

def inference(cfg: DictConfig):
    pass

@hydra.main(
    version_base=None, config_path="./conf", config_name="demo.yaml"
)
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    elif cfg.mode == "export":
        export(cfg)
    elif cfg.mode == "infer":
        inference(cfg)
    else:
        raise ValueError(
            f"cfg.mode should in ['train', 'eval', 'export', 'infer'], but got '{cfg.mode}'"
        )

if __name__ == "__main__":
    main()
