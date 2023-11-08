import numpy as np
import paddle
from tqdm import tqdm
from .utils import save_checkpoint
from .losses import LpLoss, darcy_loss, PINO_loss

def train_2d_operator(model,
                      train_loader,
                      optimizer, scheduler,
                      config,
                      rank=0, log=False,
                      project='PINO-2d-default',
                      group='default',
                      tags=['default'],
                      use_tqdm=True,
                      profile=False):

    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)
    mesh = train_loader.dataset.mesh
    mollifier = paddle.sin(np.pi * mesh[..., 0]) * paddle.sin(np.pi * mesh[..., 1]) * 0.001
    pde_mesh = train_loader.dataset.pde_mesh
    pde_mol = paddle.sin(np.pi * pde_mesh[..., 0]) * paddle.sin(np.pi * pde_mesh[..., 1]) * 0.001
    for e in pbar:
        loss_dict = {'train_loss': 0.0,
                     'data_loss': 0.0,
                     'f_loss': 0.0,
                     'test_error': 0.0}
        for data_ic, u, pde_ic in train_loader:
            data_ic, u, pde_ic = data_ic.to(rank), u.to(rank), pde_ic.to(rank)

            optimizer.zero_grad()

            # data loss
            if data_weight > 0:
                pred = model(data_ic).squeeze(dim=-1)
                pred = pred * mollifier
                data_loss = myloss(pred, y)

            a = data_ic[..., 0]
            f_loss = darcy_loss(pred, a)

            loss = data_weight * data_loss + f_weight * f_loss
            loss.backward()
            optimizer.step()

            loss_dict['train_loss'] += loss.item() * y.shape[0]
            loss_dict['f_loss'] += f_loss.item() * y.shape[0]
            loss_dict['data_loss'] += data_loss.item() * y.shape[0]

        scheduler.step()
        train_loss_val = loss_dict['train_loss'] / len(train_loader.dataset)
        f_loss_val = loss_dict['f_loss'] / len(train_loader.dataset)
        data_loss_val = loss_dict['data_loss'] / len(train_loader.dataset)

        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch: {e}, train loss: {train_loss_val:.5f}, '
                    f'f_loss: {f_loss_val:.5f}, '
                    f'data loss: {data_loss_val:.5f}'
                )
            )
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    print('Done!')

def train_2d_burger(model,
                    train_loader, v,
                    optimizer, scheduler,
                    config,
                    rank=0, log=False,
                    project='PINO-2d-default',
                    group='default',
                    tags=['default'],
                    use_tqdm=True):

    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    ic_weight = config['train']['ic_loss']
    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    for e in pbar:
        model.train()
        train_pino = 0.0
        data_l2 = 0.0
        train_loss = 0.0

        for i, (x, y) in enumerate(train_loader):
            x, y = x, y
            out = model(x).reshape(y.shape)
            data_loss = myloss(out, y)

            loss_u, loss_f = PINO_loss(out, x[:, 0, :, 0], v)
            total_loss = loss_u * ic_weight + loss_f * f_weight + data_loss * data_weight
            
            optimizer.clear_grad()
            total_loss.backward()
            optimizer.step()

            data_l2 += data_loss.item()
            train_pino += loss_f.item()
            train_loss += total_loss.item()
            
        scheduler.step()
        data_l2 /= len(train_loader)
        train_pino /= len(train_loader)
        train_loss /= len(train_loader)
        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch {e}, train loss: {train_loss:.5f} '
                    f'train f error: {train_pino:.5f}; '
                    f'data l2 error: {data_l2:.5f}'
                )
            )

        if e % 100 == 0:
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                            model, optimizer)
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    print('Done!')