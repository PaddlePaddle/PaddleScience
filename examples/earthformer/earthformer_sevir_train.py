import h5py
import hydra
import numpy as np
import paddle
import sevir_metric
import sevir_vis_seq
from omegaconf import DictConfig
from paddle import nn

import ppsci


def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def train(cfg: DictConfig):
    # set train dataloader config
    train_dataloader_cfg = {
        "dataset": {
            "name": "SEVIRDataset",
            "data_dir": cfg.FILE_PATH,
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.DATASET.label_keys,
            "data_types": cfg.DATASET.data_types,
            "seq_len": cfg.DATASET.seq_len,
            "raw_seq_len": cfg.DATASET.raw_seq_len,
            "sample_mode": cfg.DATASET.sample_mode,
            "stride": cfg.DATASET.stride,
            "batch_size": cfg.DATASET.batch_size,
            "layout": cfg.DATASET.layout,
            "in_len": cfg.DATASET.in_len,
            "out_len": cfg.DATASET.out_len,
            "split_mode": cfg.DATASET.split_mode,
            "start_date": cfg.TRAIN.start_date,
            "end_date": cfg.TRAIN.end_date,
            "preprocess": cfg.DATASET.preprocess,
            "rescale_method": cfg.DATASET.rescale_method,
            "shuffle": True,
            "verbose": False,
            "training": True,
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
        loss=ppsci.loss.FunctionalLoss(sevir_metric.train_mse_func),
        name="Sup",
    )
    constraint = {sup_constraint.name: sup_constraint}

    # set iters_per_epoch by dataloader length
    ITERS_PER_EPOCH = len(sup_constraint.data_loader)
    # set eval dataloader config
    eval_dataloader_cfg = {
        "dataset": {
            "name": "SEVIRDataset",
            "data_dir": cfg.FILE_PATH,
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.DATASET.label_keys,
            "data_types": cfg.DATASET.data_types,
            "seq_len": cfg.DATASET.seq_len,
            "raw_seq_len": cfg.DATASET.raw_seq_len,
            "sample_mode": cfg.DATASET.sample_mode,
            "stride": cfg.DATASET.stride,
            "batch_size": cfg.DATASET.batch_size,
            "layout": cfg.DATASET.layout,
            "in_len": cfg.DATASET.in_len,
            "out_len": cfg.DATASET.out_len,
            "split_mode": cfg.DATASET.split_mode,
            "start_date": cfg.TRAIN.end_date,
            "end_date": cfg.EVAL.end_date,
            "preprocess": cfg.DATASET.preprocess,
            "rescale_method": cfg.DATASET.rescale_method,
            "shuffle": False,
            "verbose": False,
            "training": False,
        },
        "batch_size": cfg.EVAL.batch_size,
    }

    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        loss=ppsci.loss.MSELoss(),
        metric={
            "rmse": ppsci.metric.FunctionalMetric(
                sevir_metric.eval_rmse_func(
                    out_len=cfg.DATASET.seq_len,
                    layout=cfg.DATASET.layout,
                    metrics_mode=cfg.EVAL.metrics_mode,
                    metrics_list=cfg.EVAL.metrics_list,
                    threshold_list=cfg.EVAL.threshold_list,
                )
            ),
        },
        name="Sup_Validator",
    )
    validator = {sup_validator.name: sup_validator}

    model = ppsci.arch.CuboidTransformer(
        **cfg.MODEL,
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

    # init optimizer and lr scheduler
    lr_scheduler_cfg = dict(cfg.TRAIN.lr_scheduler)
    lr_scheduler = ppsci.optimizer.lr_scheduler.Cosine(
        **lr_scheduler_cfg,
        iters_per_epoch=ITERS_PER_EPOCH,
        eta_min=cfg.TRAIN.min_lr_ratio * cfg.TRAIN.lr_scheduler.learning_rate,
        warmup_epoch=int(0.2 * cfg.TRAIN.epochs),
    )()
    optimizer = paddle.optimizer.AdamW(
        lr_scheduler, parameters=optimizer_grouped_parameters
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
        eval_during_train=cfg.TRAIN.eval_during_train,
        seed=cfg.seed,
        validator=validator,
        compute_metric_by_batch=cfg.EVAL.compute_metric_by_batch,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )
    # train model
    solver.train()
    # evaluate after finished training
    metric = sevir_metric.eval_rmse_func(
        out_len=cfg.DATASET.seq_len,
        layout=cfg.DATASET.layout,
        metrics_mode=cfg.EVAL.metrics_mode,
        metrics_list=cfg.EVAL.metrics_list,
        threshold_list=cfg.EVAL.threshold_list,
    )

    with solver.no_grad_context_manager(True):
        for index, (input_, label, _) in enumerate(sup_validator.data_loader):
            truefield = label["vil"].squeeze(0)
            prefield = model(input_)["vil"].squeeze(0)
            metric.sevir_score.update(prefield, truefield)

    metric_dict = metric.sevir_score.compute()
    print(metric_dict)


def evaluate(cfg: DictConfig):
    # set eval dataloader config
    eval_dataloader_cfg = {
        "dataset": {
            "name": "SEVIRDataset",
            "data_dir": cfg.FILE_PATH,
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.DATASET.label_keys,
            "data_types": cfg.DATASET.data_types,
            "seq_len": cfg.DATASET.seq_len,
            "raw_seq_len": cfg.DATASET.raw_seq_len,
            "sample_mode": cfg.DATASET.sample_mode,
            "stride": cfg.DATASET.stride,
            "batch_size": cfg.DATASET.batch_size,
            "layout": cfg.DATASET.layout,
            "in_len": cfg.DATASET.in_len,
            "out_len": cfg.DATASET.out_len,
            "split_mode": cfg.DATASET.split_mode,
            "start_date": cfg.TEST.start_date,
            "end_date": cfg.TEST.end_date,
            "preprocess": cfg.DATASET.preprocess,
            "rescale_method": cfg.DATASET.rescale_method,
            "shuffle": False,
            "verbose": False,
            "training": False,
        },
        "batch_size": cfg.EVAL.batch_size,
    }

    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        loss=ppsci.loss.MSELoss(),
        metric={
            "rmse": ppsci.metric.FunctionalMetric(
                sevir_metric.eval_rmse_func(
                    out_len=cfg.DATASET.seq_len,
                    layout=cfg.DATASET.layout,
                    metrics_mode=cfg.EVAL.metrics_mode,
                    metrics_list=cfg.EVAL.metrics_list,
                    threshold_list=cfg.EVAL.threshold_list,
                )
            ),
        },
        name="Sup_Validator",
    )
    validator = {sup_validator.name: sup_validator}

    model = ppsci.arch.CuboidTransformer(
        **cfg.MODEL,
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
    metric = sevir_metric.eval_rmse_func(
        out_len=cfg.DATASET.seq_len,
        layout=cfg.DATASET.layout,
        metrics_mode=cfg.EVAL.metrics_mode,
        metrics_list=cfg.EVAL.metrics_list,
        threshold_list=cfg.EVAL.threshold_list,
    )

    with solver.no_grad_context_manager(True):
        for index, (input_, label, _) in enumerate(sup_validator.data_loader):
            truefield = label["vil"].reshape([-1, *label["vil"].shape[2:]])
            prefield = model(input_)["vil"].reshape([-1, *label["vil"].shape[2:]])
            metric.sevir_score.update(prefield, truefield)

    metric_dict = metric.sevir_score.compute()
    print(metric_dict)


def export(cfg: DictConfig):
    # set model
    model = ppsci.arch.CuboidTransformer(
        **cfg.MODEL,
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
            key: InputSpec([1, 13, 384, 384, 1], "float32", name=key)
            for key in model.input_keys
        },
    ]
    solver.export(input_spec, cfg.INFER.export_path)


def inference(cfg: DictConfig):
    import predictor

    from ppsci.data.dataset import sevir_dataset

    predictor = predictor.EarthformerPredictor(cfg)

    if cfg.INFER.rescale_method == "sevir":
        scale_dict = sevir_dataset.PREPROCESS_SCALE_SEVIR
        offset_dict = sevir_dataset.PREPROCESS_OFFSET_SEVIR
    elif cfg.INFER.rescale_method == "01":
        scale_dict = sevir_dataset.PREPROCESS_SCALE_01
        offset_dict = sevir_dataset.PREPROCESS_OFFSET_01
    else:
        raise ValueError(f"Invalid rescale option: {cfg.INFER.rescale_method}.")

    # read h5 data
    h5data = h5py.File(cfg.INFER.data_path, "r")
    data = np.array(h5data[cfg.INFER.data_type]).transpose([0, 3, 1, 2])

    idx = np.random.choice(len(data), None, False)
    data = (
        scale_dict[cfg.INFER.data_type] * data[idx] + offset_dict[cfg.INFER.data_type]
    )

    input_data = data[: cfg.INFER.in_len, ...]
    input_data = input_data.reshape(1, *input_data.shape, 1).astype(np.float32)
    target_data = data[cfg.INFER.in_len : cfg.INFER.in_len + cfg.INFER.out_len, ...]
    target_data = target_data.reshape(1, *target_data.shape, 1).astype(np.float32)

    pred_data = predictor.predict(input_data, cfg.INFER.batch_size)

    sevir_vis_seq.save_example_vis_results(
        save_dir=cfg.INFER.sevir_vis_save,
        save_prefix=f"data_{idx}",
        in_seq=input_data,
        target_seq=target_data,
        pred_seq=pred_data,
        layout=cfg.INFER.layout,
        plot_stride=cfg.INFER.plot_stride,
        label=cfg.INFER.logging_prefix,
        interval_real_time=cfg.INFER.interval_real_time,
    )


@hydra.main(
    version_base=None,
    config_path="./conf",
    config_name="earthformer_sevir_pretrain.yaml",
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
