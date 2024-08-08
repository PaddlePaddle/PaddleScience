import ppsci
from ppsci.utils import logger

# 导入其他必要的模块
# import ...
import os
import numpy as np
from paddle.nn import BCEWithLogitsLoss, MSELoss
from ednn_utils import Meter
mseloss = MSELoss(reduction='none')

'''
(pthIFM) [tanhao.hu@localhost IFM]$ CUDA_VISIBLE_DEVICES=4 python ifm_mlp.py --data_label tox21 --embed IFM --epochs 300 --runseed 43 --batch_size 128 --patience 50 --opt_iters
50 --repetitions 50
the retained features for tox21 is 116
inputs: 117
******hyper-parameter optimization is starting now******
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [52:06<00:00, 62.53s/it, best loss: 0.15596856397717218]
******hyper-parameter optimization is over******
the best hyper-parameters settings for tox21 are:   {'d_out': 76, 'dropout': 0.02659859559082154, 'hidden_unit1': 1, 'hidden_unit2': 0, 'hidden_unit3': 0, 'l2': 0.00062648269067
85799, 'omega0': 0.48799501140521095, 'omega1': 0.37676389088248685, 'sigma': 0.186724743661628}
training set: {'roc_auc': 0.9037117675137881, 'prc_auc': 0.5111155738929765}
validation set: {'roc_auc': 0.8440314360228278, 'prc_auc': 0.4216592594326023}
test set: {'roc_auc': 0.8507254099703315, 'prc_auc': 0.446064682681851}
'''

def train_loss_func(output_dict, label_dict, *args):
    pass
    return {"pred": (mseloss(output_dict["pred"], label_dict['y']) * (label_dict['mask'] != 0).astype('float32')).mean()}

def get_val_loss_func(reg, metric):
    def val_loss_func(output_dict, label_dict, *args):
        pass
        eval_metric = Meter()
        
        #for (pred, y, mask) in zip(output_dict["pred"], label_dict["y"], label_dict['mask']):
        #    eval_metric.update(pred, y, mask) #F.mse_loss(pred, label.y)
        eval_metric.update(output_dict["pred"], label_dict["y"], label_dict['mask']) #F.mse_loss(pred, label.y)

        if reg:
            rmse_score = np.mean(eval_metric.compute_metric(metric))  # in case of multi-tasks
            mae_score = np.mean(eval_metric.compute_metric('mae'))  # in case of multi-tasks
            r2_score = np.mean(eval_metric.compute_metric('r2'))  # in case of multi-tasks
            return {'rmse': rmse_score, 'mae': mae_score, 'r2': r2_score}
        else:
            roc_score = np.mean(eval_metric.compute_metric(metric))  # in case of multi-tasks
            prc_score = np.mean(eval_metric.compute_metric('prc_auc'))  # in case of multi-tasks
            return {'roc_auc': roc_score, 'prc_auc': prc_score}
    return val_loss_func

def train():
    pass
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(43)#cfg.seed)
    # initialize logger
    #logger.init_logger("ppsci", os.path.join(cfg.output_dir, "train.log"), "info")

    # tmp args:
    data_label = 'tox21'
    learning_rate = 0.001
    output_dir = "./output_example"
    epochs = 80
    iters_per_epoch = 2
    save_freq = 2
    eval_during_train = False
    eval_freq = 2
    eval_with_no_grad = True
    checkpoint_path = None
    

    # set dataloader config
    train_dataloader_cfg = {
        "dataset": {
            "name": "IFMMoeDataset",
            "input_keys": ("x", "mask", ),
            "label_keys": ("y", "mask", ),
            "data_dir": './dataset/',
            "data_mode": 'train',
            "data_label": data_label,
        },
        "batch_size": 128,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "num_workers": 1,
    }

    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        output_expr={"pred": lambda out: out["pred"]},
        loss=ppsci.loss.FunctionalLoss(train_loss_func),
        name="Sup",
    )
    # wrap constraints together
    constraint = {sup_constraint.name: sup_constraint}
    #process_sim = sup_constraint.data_loader.dataset._preprocess
    #fine_marker_dict = sup_constraint.data_loader.dataset.marker_dict
    inputs = sup_constraint.data_loader.dataset.data_tr_x.shape[1]
    tasks = sup_constraint.data_loader.dataset.task_dict[data_label]

    # overwirte cfg
    iters_per_epoch = len(sup_constraint.data_loader)

    print(f'inputs is: {inputs}, iters_per_epoch: {iters_per_epoch}')


    if data_label == 'esol' or data_label == 'freesolv' or data_label == 'lipop':
        task_type = 'reg'
        reg = True
        metric = 'rmse'
    else:
        task_type = 'cla'
        reg = False
        metric = 'roc_auc'

    hyper_paras = {'l2': 0.0006264826906785799, #hp.uniform('l2', 0, 0.01),
                    'dropout': 0.02659859559082154, #hp.uniform('dropout', 0, 0.5),
                    'd_out': 76, #hp.randint('d_out', 127),
                    'omega0': 0.48799501140521095, #hp.uniform('omega0', 0.001, 1), #1
                    'omega1': 0.37676389088248685,  #hp.uniform('omega1', 0.001, 1), #1
                    'sigma': 0.186724743661628, #hp.loguniform('sigma', np.log(0.01), np.log(100)),
                    'hidden_unit1': 128, #hp.choice('hidden_unit1', [64, 128, 256, 512]),
                    'hidden_unit2': 64, #hp.choice('hidden_unit2', [64, 128, 256, 512]),
                    'hidden_unit3': 64, #hp.choice('hidden_unit3', [64, 128, 256, 512])
                    }
    
    hidden_units = [hyper_paras['hidden_unit1'], hyper_paras['hidden_unit2'], hyper_paras['hidden_unit3']]
    # set model
    model = ppsci.arch.IFMMLP(
        #**cfg.MODEL,
        input_keys = ('x', 'mask'),
        output_keys = ('y', 'mask'),
        hidden_units = hidden_units,
        embed_name = 'IFM',
        inputs = inputs,
        outputs = len(tasks),
        d_out = hyper_paras['d_out'],
        sigma = hyper_paras["sigma"],
        dp_ratio = hyper_paras["dropout"],
        reg = reg,
        first_omega_0 = hyper_paras['omega0'],
        hidden_omega_0 = hyper_paras['omega1'],
        #process_sim=process_sim,
        #fine_marker_dict=fine_marker_dict,
        #su2_module=su2paddle.SU2Module,
    )

    # set optimizer
    optimizer = ppsci.optimizer.Adam(learning_rate=learning_rate, weight_decay=hyper_paras['l2'])(model)

    # val
    # ...

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        output_dir,
        optimizer,
        None,
        epochs,
        iters_per_epoch,
        save_freq=save_freq,
        eval_during_train=eval_during_train,
        eval_freq=eval_freq,
        validator=None,
        eval_with_no_grad=eval_with_no_grad,
        checkpoint_path=checkpoint_path,
    )

    # train model
    solver.train()

def eval():
    pass
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(43)#cfg.seed)
    # initialize logger
    #logger.init_logger("ppsci", os.path.join(cfg.output_dir, "train.log"), "info")

    # tmp args:
    data_label = 'tox21'
    learning_rate = 0.001
    output_dir = "./output_example"
    epochs = 80
    iters_per_epoch = 2
    save_freq = 2
    eval_during_train = False
    eval_freq = 2
    eval_with_no_grad = True
    checkpoint_path = None
    

    # set dataloader config
    train_dataloader_cfg = {
        "dataset": {
            "name": "IFMMoeDataset",
            "input_keys": ("x", "mask", ),
            "label_keys": ("y", "mask", ),
            "data_dir": './dataset/',
            "data_mode": 'train',
            "data_label": data_label,
        },
        "batch_size": 128,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "num_workers": 1,
    }

    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        output_expr={"pred": lambda out: out["pred"]},
        loss=ppsci.loss.FunctionalLoss(train_loss_func),
        name="Sup",
    )
    # wrap constraints together
    constraint = {sup_constraint.name: sup_constraint}
    #process_sim = sup_constraint.data_loader.dataset._preprocess
    #fine_marker_dict = sup_constraint.data_loader.dataset.marker_dict
    inputs = sup_constraint.data_loader.dataset.data_tr_x.shape[1]
    tasks = sup_constraint.data_loader.dataset.task_dict[data_label]

    # overwirte cfg
    iters_per_epoch = len(sup_constraint.data_loader)

    print(f'inputs is: {inputs}, iters_per_epoch: {iters_per_epoch}')


    if data_label == 'esol' or data_label == 'freesolv' or data_label == 'lipop':
        task_type = 'reg'
        reg = True
        metric = 'rmse'
    else:
        task_type = 'cla'
        reg = False
        metric = 'roc_auc'

    hyper_paras = {'l2': 0.0006264826906785799, #hp.uniform('l2', 0, 0.01),
                    'dropout': 0.02659859559082154, #hp.uniform('dropout', 0, 0.5),
                    'd_out': 76, #hp.randint('d_out', 127),
                    'omega0': 0.48799501140521095, #hp.uniform('omega0', 0.001, 1), #1
                    'omega1': 0.37676389088248685,  #hp.uniform('omega1', 0.001, 1), #1
                    'sigma': 0.186724743661628, #hp.loguniform('sigma', np.log(0.01), np.log(100)),
                    'hidden_unit1': 128, #hp.choice('hidden_unit1', [64, 128, 256, 512]),
                    'hidden_unit2': 64, #hp.choice('hidden_unit2', [64, 128, 256, 512]),
                    'hidden_unit3': 64, #hp.choice('hidden_unit3', [64, 128, 256, 512])
                    }
    
    hidden_units = [hyper_paras['hidden_unit1'], hyper_paras['hidden_unit2'], hyper_paras['hidden_unit3']]
    # set model
    model = ppsci.arch.IFMMLP(
        #**cfg.MODEL,
        input_keys = ('x', 'mask'),
        #output_keys = ('y', 'mask'),
        output_keys = ('y', 'mask'),
        hidden_units = hidden_units,
        embed_name = 'IFM',
        inputs = inputs,
        outputs = len(tasks),
        d_out = hyper_paras['d_out'],
        sigma = hyper_paras["sigma"],
        dp_ratio = hyper_paras["dropout"],
        reg = reg,
        first_omega_0 = hyper_paras['omega0'],
        hidden_omega_0 = hyper_paras['omega1'],
        #process_sim=process_sim,
        #fine_marker_dict=fine_marker_dict,
        #su2_module=su2paddle.SU2Module,
    )

    # set validator
    eval_dataloader_cfg = {
        "dataset": {
            "name": "IFMMoeDataset",
            "input_keys": ("x", "mask", ),
            "label_keys": ("y", "mask", ),
            "data_dir": './dataset/',
            "data_mode": 'test',
            "data_label": data_label,
        },
        "batch_size": 128,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "num_workers": 1,
    }

    rmse_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(train_loss_func),
        output_expr={"pred": lambda out: out["pred"]},
        metric={"MyMeter": ppsci.metric.FunctionalMetric(get_val_loss_func(reg, metric))},
        name="MyMeter_validator",
    )
    validator = {rmse_validator.name: rmse_validator}

    solver = ppsci.solver.Solver(
        model,
        output_dir=output_dir,
        log_freq=20,
        seed=43,
        validator=validator,
        pretrained_model_path="./output_example/checkpoints/epoch_74.pdparams",
        eval_with_no_grad=eval_with_no_grad,
    )

    # evaluate model
    solver.eval()


if __name__ == "__main__":
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(43)
    # set output directory
    OUTPUT_DIR = "./output_example"
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    #train()
    eval()