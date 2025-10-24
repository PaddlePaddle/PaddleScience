from model import TwoModalMultiLabelModel
import hydra
import paddle
from omegaconf import DictConfig
import ppsci
def train(cfg: DictConfig):
    ERA5_UKBiobank = ppsci.constraint.SupervisedConstraint(
       {
            "dataset": {
                "name": "ToyTwoModalDataset",            # Dataset class name
                "file_path": "/path/to/file.csv",        # Path to the dataset (placeholder)
                "input_keys": ("video", "vec"),          # Input dictionary keys
                "label_keys": ("y"),                    # Label dictionary keys
                "n": 3000,                               # Total number of samples
                "seed": 0,                               # Random seed for reproducibility
                "T": 12,                                 # Temporal dimension (e.g., 12 months or years)
                "C": 10,                                 # Number of exposure variables or channels
                "H": 10,                                 # Spatial height (latitude grid)
                "W": 10,                                 # Spatial width (longitude grid)
                "N": 24                                  # Inner temporal resolution (e.g., 24 hours)
            },
           "batch_size":8,
       },
        ppsci.loss.BCELoss(),
        name="ERA5_UKBiobank",
    )
    model = TwoModalMultiLabelModel(
        vid_channels=cfg.MODEL.C, vid_h=cfg.MODEL.H, vid_w=cfg.MODEL.W, vid_frames=cfg.MODEL.T, depth_n=cfg.MODEL.N,
        vec_dim=424,
        d_model=512, nhead=4, n_trans_layers=2, trans_ff=1024,
        tabm_hidden=512, dropout=0.1, num_labels=4,
        moe_temporal_attn=True, moe_temporal_afno=True,
        moe_fused=False, moe_tabm=False,
        afno_modes=32
    )
    optimizer = paddle.optimizer.Adam(learning_rate=3e-4, parameters=model.parameters())
    constraint={ERA5_UKBiobank.name:ERA5_UKBiobank,}
    solver = ppsci.solver.Solver(
            model,
            constraint,
            cfg.output_dir,
            optimizer,
            epochs=cfg.TRAIN.epochs,
            iters_per_epoch=cfg.TRAIN.iters_per_epoch,
            eval_during_train=cfg.TRAIN.eval_during_train,
            eval_freq=cfg.TRAIN.eval_freq,
            #equation=equation,
            #geom=geom,
            #validator=validator,
            #visualizer=visualizer,
        )
    solver.train()


def evaluate(cfg: DictConfig):
    pass


def export(cfg: DictConfig):
    pass


def inference(cfg: DictConfig):
    pass


@hydra.main(version_base=None, config_path="./config", config_name="era5_ukb.yaml")# joint contribution文件夹下
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