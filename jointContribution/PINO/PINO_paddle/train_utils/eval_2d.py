from tqdm import tqdm
import numpy as np

import paddle

from .losses import LpLoss, darcy_loss, PINO_loss

import matplotlib.pyplot as plt

def eval_darcy(model,
               dataloader,
               config,
               use_tqdm=True):
    model.eval()
    myloss = LpLoss(size_average=True)
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader

    mesh = dataloader.dataset.mesh
    mollifier = paddle.sin(np.pi * mesh[..., 0]) * paddle.sin(np.pi * mesh[..., 1]) * 0.001
    f_val = []
    test_err = []
    i = 0
    fig, ax = plt.subplots(3,3)
    with paddle.no_grad():
        for x, y in pbar:

            pred = model(x).reshape(y.shape)
            pred = pred * mollifier
            if i < 3:
                ax[2][i].imshow(pred[0, :, :])
                ax[2][i].set_title('prediction')
                ax[1][i].imshow(y[0, :, :])
                ax[1][i].set_title('ground truth')
                ax[0][i].imshow(x[0, :, :, 0])
                ax[0][i].set_title('input')

                for k in range(3):
                    ax[k][i].set_xlabel('x')
                    ax[k][i].set_ylabel('y')
            if i==3:
                plt.tight_layout()
                plt.savefig('result.png')
            i+=1
            data_loss = myloss(pred, y)
            a = x[..., 0]
            f_loss = darcy_loss(pred, a)

            test_err.append(data_loss.item())
            f_val.append(f_loss.item())
            if use_tqdm:
                pbar.set_description(
                    (
                        f'Equation error: {f_loss.item():.5f}, test l2 error: {data_loss.item()}'
                    )
                )
    mean_f_err = np.mean(f_val)
    std_f_err = np.std(f_val, ddof=1) / np.sqrt(len(f_val))

    mean_err = np.mean(test_err)
    std_err = np.std(test_err, ddof=1) / np.sqrt(len(test_err))

    print(f'==Averaged relative L2 error mean: {mean_err}, std error: {std_err}==\n'
          f'==Averaged equation error mean: {mean_f_err}, std error: {std_f_err}==')

def eval_burgers(model,
                 dataloader,
                 v,
                 config,
                 use_tqdm=True):
    model.eval()
    myloss = LpLoss(size_average=True)
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader

    test_err = []
    f_err = []
    i = 0
    fig, ax = plt.subplots(2,3)
    for x, y in pbar:
        x, y = x, y
        out = model(x).reshape(y.shape)
        data_loss = myloss(out, y)
        if i<3:
            ax[0][i].imshow(out[0, :, :])
            ax[0][i].set_xlabel('x')
            ax[0][i].set_ylabel('t')
            ax[0][i].set_title('prediction')
            ax[1][i].imshow(y[0, :, :])
            ax[1][i].set_xlabel('x')
            ax[1][i].set_ylabel('t')
            ax[1][i].set_title('ground truth')
        if i==3:
            plt.tight_layout()
            plt.savefig('result.png')
        i+=1
        loss_u, f_loss = PINO_loss(out, x[:, 0, :, 0], v)
        test_err.append(data_loss.item())
        f_err.append(f_loss.item())

    mean_f_err = np.mean(f_err)
    std_f_err = np.std(f_err, ddof=1) / np.sqrt(len(f_err))

    mean_err = np.mean(test_err)
    std_err = np.std(test_err, ddof=1) / np.sqrt(len(test_err))

    print(f'==Averaged relative L2 error mean: {mean_err}, std error: {std_err}==\n'
          f'==Averaged equation error mean: {mean_f_err}, std error: {std_f_err}==')

