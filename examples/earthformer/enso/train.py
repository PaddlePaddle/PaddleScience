import hydra
import numpy as np
import paddle
import xarray as xr
from helps import eval_rmse_func
from helps import get_parameter_names
from helps import train_mse_func
from omegaconf import DictConfig
from paddle import nn

import ppsci
from ppsci.data.dataset.enso_dataset import fold
from ppsci.data.dataset.enso_dataset import prepare_inputs_targets


def train(cfg: DictConfig):
    # set train dataloader config
    train_dataloader_cfg = {
        "dataset": {
            "name": "ENSODataset",
            "data_dir": cfg.FILE_PATH,
            "input_keys": cfg.MODEL.afno.input_keys,
            "label_keys": cfg.DATASET.label_keys,
            "in_len": cfg.DATASET.in_len,
            "out_len": cfg.DATASET.out_len,
            "in_stride": cfg.DATASET.in_stride,
            "out_stride": cfg.DATASET.out_stride,
            "train_samples_gap": cfg.DATASET.train_samples_gap,
            "eval_samples_gap": cfg.DATASET.eval_samples_gap,
            "normalize_sst": cfg.DATASET.normalize_sst,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": True,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "num_workers": 8,
    }

    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(train_mse_func),
        name="Sup",
    )
    constraint = {sup_constraint.name: sup_constraint}

    # set iters_per_epoch by dataloader length
    ITERS_PER_EPOCH = len(sup_constraint.data_loader)
    # set eval dataloader config
    eval_dataloader_cfg = {
        "dataset": {
            "name": "ENSODataset",
            "data_dir": cfg.FILE_PATH,
            "input_keys": cfg.MODEL.afno.input_keys,
            "label_keys": cfg.DATASET.label_keys,
            "in_len": cfg.DATASET.in_len,
            "out_len": cfg.DATASET.out_len,
            "in_stride": cfg.DATASET.in_stride,
            "out_stride": cfg.DATASET.out_stride,
            "train_samples_gap": cfg.DATASET.train_samples_gap,
            "eval_samples_gap": cfg.DATASET.eval_samples_gap,
            "normalize_sst": cfg.DATASET.normalize_sst,
            "training": "eval",
        },
        "batch_size": cfg.EVAL.batch_size,
    }

    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(train_mse_func),
        metric={
            "rmse": ppsci.metric.FunctionalMetric(eval_rmse_func, keep_batch=False),
        },
        name="Sup_Validator",
    )
    validator = {sup_validator.name: sup_validator}

    num_blocks = len(cfg.MODEL.afno["enc_depth"])
    if isinstance(cfg.MODEL["self_pattern"], str):
        enc_attn_patterns = [cfg.MODEL["self_pattern"]] * num_blocks

    if isinstance(cfg.MODEL["cross_self_pattern"], str):
        dec_self_attn_patterns = [cfg.MODEL["cross_self_pattern"]] * num_blocks

    if isinstance(cfg.MODEL["cross_pattern"], str):
        dec_cross_attn_patterns = [cfg.MODEL["cross_pattern"]] * num_blocks

    model = ppsci.arch.CuboidTransformerModel(
        **cfg.MODEL.afno,
        enc_attn_patterns=enc_attn_patterns,
        dec_self_attn_patterns=dec_self_attn_patterns,
        dec_cross_attn_patterns=dec_cross_attn_patterns,
    )

    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": cfg.TRAIN.wd,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if n not in decay_parameters
            ],
            "weight_decay": 0.0,
        },
    ]

    # # init optimizer and lr scheduler
    lr_scheduler_cfg = dict(cfg.TRAIN.lr_scheduler)
    lr_scheduler_cfg.update({"iters_per_epoch": ITERS_PER_EPOCH})
    lr_scheduler = ppsci.optimizer.lr_scheduler.Cosine(
        **lr_scheduler_cfg,
        eta_min=cfg.TRAIN.min_lr_ratio * cfg.TRAIN.lr_scheduler.learning_rate,
        warmup_epoch=int(0.2 * cfg.TRAIN.epochs),
    )()
    # optimizer = ppsci.optimizer.AdamW(lr_scheduler)(model)
    optimizer = paddle.optimizer.AdamW(
        lr_scheduler, parameters=optimizer_grouped_parameters, weight_decay=cfg.TRAIN.wd
    )

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        lr_scheduler,
        cfg.TRAIN.epochs,
        ITERS_PER_EPOCH,
        eval_during_train=True,
        seed=cfg.seed,
        validator=validator,
        compute_metric_by_batch=cfg.EVAL.compute_metric_by_batch,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()


def evaluate(cfg: DictConfig):
    # set eval dataloader config
    eval_dataloader_cfg = {
        "dataset": {
            "name": "ENSODataset",
            "data_dir": cfg.FILE_PATH,
            "input_keys": cfg.MODEL.afno.input_keys,
            "label_keys": cfg.DATASET.label_keys,
            "in_len": cfg.DATASET.in_len,
            "out_len": cfg.DATASET.out_len,
            "in_stride": cfg.DATASET.in_stride,
            "out_stride": cfg.DATASET.out_stride,
            "train_samples_gap": cfg.DATASET.train_samples_gap,
            "eval_samples_gap": cfg.DATASET.eval_samples_gap,
            "normalize_sst": cfg.DATASET.normalize_sst,
            "training": "test",
        },
        "batch_size": cfg.EVAL.batch_size,
    }

    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(train_mse_func),
        metric={
            "rmse": ppsci.metric.FunctionalMetric(eval_rmse_func, keep_batch=False),
        },
        name="Sup_Validator",
    )
    validator = {sup_validator.name: sup_validator}
    num_blocks = len(cfg.MODEL.afno["enc_depth"])
    if isinstance(cfg.MODEL["self_pattern"], str):
        enc_attn_patterns = [cfg.MODEL["self_pattern"]] * num_blocks

    if isinstance(cfg.MODEL["cross_self_pattern"], str):
        dec_self_attn_patterns = [cfg.MODEL["cross_self_pattern"]] * num_blocks

    if isinstance(cfg.MODEL["cross_pattern"], str):
        dec_cross_attn_patterns = [cfg.MODEL["cross_pattern"]] * num_blocks

    model = ppsci.arch.CuboidTransformerModel(
        **cfg.MODEL.afno,
        enc_attn_patterns=enc_attn_patterns,
        dec_self_attn_patterns=dec_self_attn_patterns,
        dec_cross_attn_patterns=dec_cross_attn_patterns,
    )

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        log_freq=cfg.log_freq,
        seed=cfg.seed,
        validator=validator,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
        compute_metric_by_batch=cfg.EVAL.compute_metric_by_batch,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )
    # evaluate
    solver.eval()


def export(cfg: DictConfig):
    # set model
    num_blocks = len(cfg.MODEL.afno["enc_depth"])
    if isinstance(cfg.MODEL["self_pattern"], str):
        enc_attn_patterns = [cfg.MODEL["self_pattern"]] * num_blocks

    if isinstance(cfg.MODEL["cross_self_pattern"], str):
        dec_self_attn_patterns = [cfg.MODEL["cross_self_pattern"]] * num_blocks

    if isinstance(cfg.MODEL["cross_pattern"], str):
        dec_cross_attn_patterns = [cfg.MODEL["cross_pattern"]] * num_blocks

    model = ppsci.arch.CuboidTransformerModel(
        **cfg.MODEL.afno,
        enc_attn_patterns=enc_attn_patterns,
        dec_self_attn_patterns=dec_self_attn_patterns,
        dec_cross_attn_patterns=dec_cross_attn_patterns,
    )

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        pretrained_model_path=cfg.INFER.pretrained_model_path,
    )
    # export model
    from paddle.static import InputSpec

    input_spec = [
        {
            key: InputSpec([1, 12, 24, 48, 1], "float32", name=key)
            for key in model.input_keys
        },
    ]
    solver.export(input_spec, cfg.INFER.export_path)


def inference(cfg: DictConfig):
    from deploy.python_infer import pinn_predictor

    predictor = pinn_predictor.PINNPredictor(cfg)

    train_cmip = xr.open_dataset(cfg.INFER.data_path).transpose(
        "year", "month", "lat", "lon"
    )
    # select longitudes
    lon = train_cmip.lon.values
    lon = lon[np.logical_and(lon >= 95, lon <= 330)]
    train_cmip = train_cmip.sel(lon=lon)
    data = train_cmip.sst.values
    data = fold(data)

    idx_sst = prepare_inputs_targets(
        len_time=data.shape[0],
        input_length=cfg.INFER.in_len,
        input_gap=cfg.INFER.in_stride,
        pred_shift=cfg.INFER.out_len * cfg.INFER.out_stride,
        pred_length=cfg.INFER.out_len,
        samples_gap=cfg.INFER.samples_gap,
    )
    data = data[idx_sst].astype("float32")

    sst_data = data[..., np.newaxis]
    idx = np.random.choice(len(data), None, False)
    in_seq = sst_data[idx, : cfg.INFER.in_len, ...]  # ( in_len, lat, lon, 1)
    in_seq = in_seq[np.newaxis, ...]
    target_seq = sst_data[idx, cfg.INFER.in_len :, ...]  # ( out_len, lat, lon, 1)
    target_seq = target_seq[np.newaxis, ...]

    output_dict = predictor.predict({"sst_data": in_seq}, cfg.INFER.batch_size)
    output_dict = {
        store_key: output_dict[infer_key]
        for store_key, infer_key in zip({"output"}, output_dict.keys())
    }


@hydra.main(
    version_base=None,
    config_path="./conf",
    config_name="earthformer_enso_pretrain.yaml",
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
