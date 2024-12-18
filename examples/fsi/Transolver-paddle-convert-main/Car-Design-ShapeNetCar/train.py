import os
import paddle
import numpy as np
import time, json
from paddle.io import DataLoader
from tqdm import tqdm
from typing import List, Tuple
from dataset.dataset import Data


def custom_collate_fn(batch: Tuple['Data', paddle.Tensor]):
    """自定义collate_fn，用于处理单个Data类型的数据项，直接返回单个数据和shape。"""
    data, shape = batch[0]

    # 提取 cfd_data 的各属性
    pos = data.pos
    x = data.x
    y = data.y
    surf = data.surf
    edge_index = data.edge_index

    # 创建新的 Data 对象
    single_data = Data(pos=pos, x=x, y=y, surf=surf, edge_index=edge_index)

    # 直接返回单个 Data 和 shape
    return single_data, shape



def get_nb_trainable_params(model):
    """
    Return the number of trainable parameters
    """
    model_parameters = filter(lambda p: not p.stop_gradient, model.parameters()
                              )
    return sum([np.prod(tuple(p.shape)) for p in model_parameters])


def train(device, model, train_loader, optimizer, scheduler, reg=1):
    model.train()
    criterion_func = paddle.nn.MSELoss(reduction='none')
    losses_press = []
    losses_velo = []
    for cfd_data, geom in train_loader:
        cfd_data = cfd_data.to(device)
        geom = geom.to(device)
        optimizer.clear_gradients(set_to_zero=False)
        out = model((cfd_data, geom))
        targets = cfd_data.y
        loss_press = criterion_func(out[cfd_data.surf, -1], targets[
            cfd_data.surf, -1]).mean(axis=0)
        loss_velo_var = criterion_func(out[:, :-1], targets[:, :-1]).mean(axis
                                                                          =0)
        loss_velo = loss_velo_var.mean()
        total_loss = loss_velo + reg * loss_press
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        losses_press.append(loss_press.item())
        losses_velo.append(loss_velo.item())
    return np.mean(losses_press), np.mean(losses_velo)


@paddle.no_grad()
def test(device, model, test_loader):
    model.eval()
    criterion_func = paddle.nn.MSELoss(reduction='none')
    losses_press = []
    losses_velo = []
    for cfd_data, geom in test_loader:
        cfd_data = cfd_data.to(device)
        geom = geom.to(device)
        out = model((cfd_data, geom))
        targets = cfd_data.y
        loss_press = criterion_func(out[cfd_data.surf, -1], targets[
            cfd_data.surf, -1]).mean(axis=0)
        loss_velo_var = criterion_func(out[:, :-1], targets[:, :-1]).mean(axis
                                                                          =0)
        loss_velo = loss_velo_var.mean()
        losses_press.append(loss_press.item())
        losses_velo.append(loss_velo.item())
    return np.mean(losses_press), np.mean(losses_velo)


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(device, train_dataset, val_dataset, Net, hparams, path, reg=1,
         val_iter=1, coef_norm=[]):
    model = Net.to(device)
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
                                      learning_rate=hparams['lr'], weight_decay=0.0)
    tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(T_max=(len(
        train_dataset) // hparams['batch_size'] + 1) * hparams['nb_epochs'],
                                                      eta_min=hparams['lr'] / 1000, learning_rate=optimizer.get_lr())
    optimizer.set_lr_scheduler(tmp_lr)
    lr_scheduler = tmp_lr
    start = time.time()
    train_loss, val_loss = 100000.0, 100000.0
    pbar_train = tqdm(range(hparams['nb_epochs']), position=0)
    for epoch in pbar_train:
        train_loader = DataLoader(train_dataset, batch_size=hparams[
            'batch_size'], shuffle=True, drop_last=True, collate_fn=custom_collate_fn)
        loss_velo, loss_press = train(device, model, train_loader,
                                      optimizer, lr_scheduler, reg=reg)
        train_loss = loss_velo + reg * loss_press
        del train_loader
        if val_iter is not None and (epoch == hparams['nb_epochs'] - 1 or
                                     epoch % val_iter == 0):
            val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=custom_collate_fn)
            loss_velo, loss_press = test(device, model, val_loader)
            val_loss = loss_velo + reg * loss_press
            del val_loader
            pbar_train.set_postfix(train_loss=train_loss, val_loss=val_loss)
        else:
            pbar_train.set_postfix(train_loss=train_loss)
    end = time.time()
    time_elapsed = end - start
    params_model = float(get_nb_trainable_params(model))  # 确保 params_model 是浮点数
    print('Number of parameters:', params_model)
    print('Time elapsed: {0:.2f} seconds'.format(time_elapsed))

    # 保存模型权重
    model_path = os.path.join(path, f"model_{hparams['nb_epochs']}.pdparams")
    paddle.save(model.state_dict(), model_path)

    # 记录日志
    if val_iter is not None:
        log_path = os.path.join(path, f"log_{hparams['nb_epochs']}.json")
        with open(log_path, 'a') as f:
            log_data = {
                'nb_parameters': params_model,
                'time_elapsed': time_elapsed,
                'hparams': hparams,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'coef_norm': list(coef_norm)
            }
            json.dump(log_data, f, indent=4, cls=NumpyEncoder)

    return model
