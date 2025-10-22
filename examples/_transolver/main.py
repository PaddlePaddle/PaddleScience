"""
Reference: https://github.com/thuml/Transolver
"""
from __future__ import annotations

import hydra
import numpy as np
import paddle
import paddle.nn.functional as F
import scipy as sc
from drag_coefficient import cal_coefficient
from omegaconf import DictConfig
from tqdm import tqdm

import ppsci
from ppsci.data.dataset.shapenet_car import load_train_val_fold
from ppsci.data.dataset.shapenet_car import load_train_val_fold_file
from ppsci.utils import logger
from ppsci.utils import save_load

dtype = paddle.get_default_dtype()


def train(cfg: DictConfig):
    # set model
    model = ppsci.arch.Transolver(**cfg.MODEL)

    # set constraint
    train_data, val_data, coef_norm = load_train_val_fold(
        cfg.DATA.data_dir,
        cfg.DATA.val_fold_id,
        cfg.DATA.save_dir,
        preprocessed=cfg.DATA.preprocessed,
    )
    train_dataset = ppsci.data.dataset.ShapeNetCarDataset(
        cfg.DATA.input_keys,
        cfg.DATA.label_keys,
        train_data,
        use_cfd_mesh=cfg.DATA.use_cfd_mesh,
        r=cfg.DATA.r,
        training=True,
    )

    def loss_func(output, label, _):
        velo_label = label[cfg.DATA.label_keys[0]]
        press_label = label[cfg.DATA.label_keys[1]]
        surf_mask = label["surf"]
        mse_velo = F.mse_loss(output[cfg.DATA.label_keys[0]], velo_label)
        mse_press = F.mse_loss(
            output[cfg.MODEL.output_keys[1]][surf_mask], press_label[surf_mask]
        )

        return {
            "mse_velo_press": mse_velo + cfg.TRAIN.press_weight * mse_press,
        }

    sup_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": train_dataset,
            "batch_size": cfg.TRAIN.batch_size,
            "num_workers": 0,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": True,
                "shuffle": True,
            },
        },
        loss=ppsci.loss.FunctionalLoss(loss_func),
        name="Sup",
    )

    # wrap constraints together
    constraint = {
        sup_constraint.name: sup_constraint,
    }

    # set optimizer
    ## Slightly differnt from transolver's lr setting(OneCycleLR)
    lr = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        cfg.TRAIN.epochs,
        len(sup_constraint.data_loader),
        cfg.TRAIN.lr.max_lr,
        gamma=cfg.TRAIN.lr.gamma,
        decay_steps=cfg.TRAIN.lr.decay_steps,
        warmup_epoch=int(cfg.TRAIN.epochs * 0.3),
        warmup_start_lr=cfg.TRAIN.lr.max_lr / 25.0,
    )()
    optimizer = ppsci.optimizer.Adam(lr)(model)

    # set validator
    val_dataset = ppsci.data.dataset.ShapeNetCarDataset(
        cfg.DATA.input_keys,
        cfg.DATA.label_keys,
        val_data,
        use_cfd_mesh=cfg.DATA.use_cfd_mesh,
        r=cfg.DATA.r,
        training=False,
    )

    def val_metric_func(output, label):
        velo_label = label[cfg.DATA.label_keys[0]]
        press_label = label[cfg.DATA.label_keys[1]]
        surf_mask = label["surf"]
        loss_velo_vec = F.mse_loss(
            output[cfg.DATA.label_keys[0]], velo_label, "none"
        ).mean(axis=0)
        loss_velo = loss_velo_vec.mean()
        loss_press = F.mse_loss(
            output[cfg.DATA.output_keys[1]][surf_mask], press_label[surf_mask]
        )

        return {
            "press": loss_press,
            "velo_vec": loss_velo,
        }

    validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": val_dataset,
            "batch_size": cfg.EVAL.batch_size,
            "num_workers": 0,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": False,
            },
        },
        metric={
            "mse": ppsci.metric.FunctionalMetric(val_metric_func),
        },
        name="validator",
    )
    validator = {validator.name: validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        optimizer=optimizer,
        validator=validator,
        cfg=cfg,
    )
    solver.eval()
    # train model
    solver.train()


def eval_on_dataloadr(model_forward, dataloader, coef_norm, val_list):
    with paddle.no_grad():
        l2errs_press = []
        l2errs_velo = []
        mses_press = []
        mses_velo_var = []
        gt_coef_list = []
        pred_coef_list = []
        coef_error = 0
        index = 0
        pbar = tqdm(dataloader, desc="Testing", unit="batch")
        for i, (inp, label, _) in enumerate(pbar, start=1):
            out = model_forward(inp)
            velo_vec = out["velo_vec"]
            press = out["press"]
            targets_velo_vec = label["velo_vec"]
            targets_press = label["press"]
            surf_mask = label["surf"]

            if coef_norm is not None:
                mean = paddle.tensor(coef_norm[2], dtype=dtype)
                std = paddle.tensor(coef_norm[3], dtype=dtype)

                pred_press: paddle.Tensor = press[surf_mask] * std[-1] + mean[-1]
                gt_press: paddle.Tensor = targets_press[surf_mask] * std[-1] + mean[-1]

                pred_surf_velo: paddle.Tensor = (
                    velo_vec[surf_mask] * std[:-1] + mean[:-1]
                )
                gt_surf_velo: paddle.Tensor = (
                    targets_velo_vec[surf_mask] * std[:-1] + mean[:-1]
                )

                pred_velo: paddle.Tensor = velo_vec[~surf_mask] * std[:-1] + mean[:-1]
                gt_velo: paddle.Tensor = (
                    targets_velo_vec[~surf_mask] * std[:-1] + mean[:-1]
                )

                # out_denorm: paddle.Tensor = out * std + mean
                # y_denorm: paddle.Tensor = targets * std + mean
                # np.save('./results/' + args.cfd_model + '/' + str(index) + '_pred.npy', out_denorm.numpy())
                # np.save('./results/' + args.cfd_model + '/' + str(index) + '_gt.npy', y_denorm.numpy())

            pred_coef = cal_coefficient(
                val_list[index].split("/")[1],
                pred_press[:, None].numpy(),
                pred_surf_velo.numpy(),
            )
            gt_coef = cal_coefficient(
                val_list[index].split("/")[1],
                gt_press[:, None].numpy(),
                gt_surf_velo.numpy(),
            )

            gt_coef_list.append(gt_coef)
            pred_coef_list.append(pred_coef)
            coef_error += abs(pred_coef - gt_coef) / gt_coef
            pbar.set_postfix(
                {
                    "batch": f"{i}/{len(dataloader)}, coef_error: {coef_error / (index + 1):.10f}",
                }
            )

            l2err_press = paddle.norm(pred_press - gt_press) / paddle.norm(gt_press)
            l2err_velo = paddle.norm(pred_velo - gt_velo) / paddle.norm(gt_velo)

            mse_press = F.mse_loss(
                press[surf_mask], targets_press[surf_mask], "none"
            ).mean(axis=0)
            mse_velo_var = F.mse_loss(
                velo_vec[~surf_mask], targets_velo_vec[~surf_mask], "none"
            ).mean(axis=0)

            l2errs_press.append(l2err_press.numpy())
            l2errs_velo.append(l2err_velo.numpy())
            mses_press.append(mse_press.numpy())
            mses_velo_var.append(mse_velo_var.numpy())
            index += 1

        gt_coef_list = np.array(gt_coef_list)
        pred_coef_list = np.array(pred_coef_list)
        spear = sc.stats.spearmanr(gt_coef_list, pred_coef_list)[0]
        logger.info(f"rho_d:, {spear:.5f}")
        logger.info(f"c_d: {coef_error / index:.5f}")
        l2err_press = np.mean(l2errs_press)
        l2err_velo = np.mean(l2errs_velo)
        rmse_press = np.sqrt(np.mean(mses_press))
        rmse_velo_var = np.sqrt(np.mean(mses_velo_var, axis=0))
        if coef_norm is not None:
            rmse_press *= coef_norm[3][-1]
            rmse_velo_var *= coef_norm[3][:-1]
        logger.info(f"relative l2 error of press: {l2err_press:.5f}")
        logger.info(f"relative l2 error of velocity: {l2err_velo:.5f}")
        logger.info(f"press: {rmse_press:.5f}")
        logger.info(
            f"velocity: {rmse_velo_var} {np.sqrt(np.mean(np.square(rmse_velo_var))):.5f}"
        )


def evaluate(cfg: DictConfig):
    # set model
    model = ppsci.arch.Transolver(**cfg.MODEL)

    # load pretrained model
    save_load.load_pretrain(model, cfg.EVAL.pretrained_model_path)
    model.eval()

    # evaluate manually
    _, val_data, coef_norm, val_list = load_train_val_fold_file(
        cfg.DATA.data_dir,
        cfg.DATA.val_fold_id,
        cfg.DATA.save_dir,
        preprocessed=cfg.DATA.preprocessed,
    )
    test_dataset = ppsci.data.dataset.ShapeNetCarDataset(
        cfg.DATA.input_keys,
        cfg.DATA.label_keys,
        val_data,
        use_cfd_mesh=cfg.DATA.use_cfd_mesh,
        r=cfg.DATA.r,
        training=False,
    )
    test_dataloader = ppsci.data.build_dataloader(
        test_dataset,
        {
            "batch_size": cfg.EVAL.batch_size,
            "num_workers": 0,
        },
    )

    eval_on_dataloadr(model, test_dataloader, coef_norm, val_list)


def export(cfg: DictConfig):
    # set model
    model = ppsci.arch.Transolver(**cfg.MODEL)

    # initialize solver
    solver = ppsci.solver.Solver(model, cfg=cfg)
    # export model
    from paddle.static import InputSpec

    print(model.input_keys)
    input_spec = [
        {
            model.input_keys[0]: InputSpec(
                [1, None, 7], "float32", name=model.input_keys[0]
            ),
        },
    ]
    import einops

    solver.export(
        input_spec, cfg.INFER.export_path, with_onnx=False, ignore_modules=[einops]
    )


def inference(cfg: DictConfig):
    from deploy import python_infer

    predictor = python_infer.GeneralPredictor(cfg)

    # inference manually
    _, val_data, coef_norm, val_list = load_train_val_fold_file(
        cfg.DATA.data_dir,
        cfg.DATA.val_fold_id,
        cfg.DATA.save_dir,
        preprocessed=cfg.DATA.preprocessed,
    )
    test_dataset = ppsci.data.dataset.ShapeNetCarDataset(
        cfg.DATA.input_keys,
        cfg.DATA.label_keys,
        val_data,
        use_cfd_mesh=cfg.DATA.use_cfd_mesh,
        r=cfg.DATA.r,
        training=False,
    )
    test_dataloader = ppsci.data.build_dataloader(
        test_dataset,
        {
            "batch_size": cfg.TRAIN.batch_size,
            "num_workers": 0,
        },
    )

    def wrap_predict(x):
        raw_out = predictor.predict(
            {k: v.numpy() for k, v in x.items()}, batch_size=None
        )
        return {
            store_key: paddle.tensor(raw_out[infer_key])
            for store_key, infer_key in zip(cfg.MODEL.output_keys[::-1], raw_out.keys())
        }

    eval_on_dataloadr(wrap_predict, test_dataloader, coef_norm, val_list)


@hydra.main(version_base=None, config_path="./conf", config_name="shapenet_car.yaml")
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
