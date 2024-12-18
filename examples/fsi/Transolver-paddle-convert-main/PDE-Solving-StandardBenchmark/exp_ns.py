import sys
# sys.path.append('../../utils')
from utils import paddle_aux
import os
import paddle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import scipy.io as scio
import numpy as np
from tqdm import *
from utils.testloss import TestLoss
from model_dict import get_model
parser = argparse.ArgumentParser('Training Transformer')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-05)
parser.add_argument('--model', type=str, default='Transolver_Structured_Mesh_2D')
parser.add_argument('--n-hidden', type=int, default=256, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=8, help='layers')
parser.add_argument('--n-heads', type=int, default=8)
parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--downsample', type=int, default=1)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--unified_pos', type=int, default=1)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--slice_num', type=int, default=32)
parser.add_argument('--eval', type=int, default=1)
parser.add_argument('--save_name', type=str, default='ns_Transolver')
parser.add_argument('--data_path', type=str, default='data/fno')
args = parser.parse_args()
n_gpu = paddle.device.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and paddle.device.cuda.device_count() >= 1
device = str(f'cuda:{args.gpu}' if use_cuda else 'cpu').replace('cuda', 'gpu')
data_path = (args.data_path +
    '/NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat')
ntrain = 1000
ntest = 200
T_in = 10
T = 10
step = 1
eval = args.eval
save_name = args.save_name


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not not parameter.stop_gradient:
            continue
        params = parameter.size
        total_params += params
    print(f'Total Trainable Params: {total_params}')
    return total_params


def main():
    r = args.downsample
    h = int((64 - 1) / r + 1)
    data = scio.loadmat(data_path)
    print(tuple(data['u'].shape))
    train_a = data['u'][:ntrain, ::r, ::r, :T_in][:, :h, :h, :]
    train_a = train_a.reshape(tuple(train_a.shape)[0], -1, tuple(train_a.
        shape)[-1])
    train_a = paddle.to_tensor(data=train_a)
    train_u = data['u'][:ntrain, ::r, ::r, T_in:T + T_in][:, :h, :h, :]
    train_u = train_u.reshape(tuple(train_u.shape)[0], -1, tuple(train_u.
        shape)[-1])
    train_u = paddle.to_tensor(data=train_u)
    test_a = data['u'][-ntest:, ::r, ::r, :T_in][:, :h, :h, :]
    test_a = test_a.reshape(tuple(test_a.shape)[0], -1, tuple(test_a.shape)[-1]
        )
    test_a = paddle.to_tensor(data=test_a)
    test_u = data['u'][-ntest:, ::r, ::r, T_in:T + T_in][:, :h, :h, :]
    test_u = test_u.reshape(tuple(test_u.shape)[0], -1, tuple(test_u.shape)[-1]
        )
    test_u = paddle.to_tensor(data=test_u)
    x = np.linspace(0, 1, h)
    y = np.linspace(0, 1, h)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.flatten(), y.flatten()]
    pos = paddle.to_tensor(data=pos, dtype='float32').unsqueeze(axis=0)
    pos_train = pos.tile(repeat_times=[ntrain, 1, 1])
    pos_test = pos.tile(repeat_times=[ntest, 1, 1])
    train_loader = paddle.io.DataLoader(dataset=paddle.io.TensorDataset([
        pos_train, train_a, train_u]), batch_size=args.batch_size, shuffle=True
        )
    test_loader = paddle.io.DataLoader(dataset=paddle.io.TensorDataset([
        pos_test, test_a, test_u]), batch_size=args.batch_size, shuffle=False)
    print('Dataloading is over.')
    model = get_model(args).Model(space_dim=2, n_layers=args.n_layers,
        n_hidden=args.n_hidden, dropout=args.dropout, n_head=args.n_heads,
        Time_Input=False, mlp_ratio=args.mlp_ratio, fun_dim=T_in, out_dim=1,
        slice_num=args.slice_num, ref=args.ref, unified_pos=args.
        unified_pos, H=h, W=h).to(device)
    optimizer = paddle.optimizer.AdamW(parameters=model.parameters(),
        learning_rate=args.lr, weight_decay=args.weight_decay)
    print(args)
    print(model)
    count_parameters(model)
    tmp_lr = paddle.optimizer.lr.OneCycleLR(total_steps=len(train_loader) *
        args.epochs, max_learning_rate=args.lr)
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr
    myloss = TestLoss(size_average=False)
    if eval:
        model.set_state_dict(state_dict=paddle.load(path=str(
            './checkpoints/' + save_name + '.pt')))
        model.eval()
        showcase = 10
        id = 0
        if not os.path.exists('./results/' + save_name + '/'):
            os.makedirs('./results/' + save_name + '/')
        test_l2_full = 0
        with paddle.no_grad():
            for x, fx, yy in test_loader:
                id += 1
                x, fx, yy = x.to(device), fx.to(device), yy.to(device)
                bsz = tuple(x.shape)[0]
                for t in range(0, T, step):
                    im = model(x, fx=fx)
                    fx = paddle.concat(x=(fx[..., step:], im), axis=-1)
                    if t == 0:
                        pred = im
                    else:
                        pred = paddle.concat(x=(pred, im), axis=-1)
                if id < showcase:
                    print(id)
                    plt.figure()
                    plt.axis('off')
                    plt.imshow(im[0, :, 0].reshape(64, 64).detach().cpu().
                        numpy(), cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(-3, 3)
                    plt.savefig(os.path.join('./results/' + save_name + '/',
                        'case_' + str(id) + '_pred_' + str(20) + '.pdf'))
                    plt.close()
                    plt.figure()
                    plt.axis('off')
                    plt.imshow(yy[0, :, t].reshape(64, 64).detach().cpu().
                        numpy(), cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(-3, 3)
                    plt.savefig(os.path.join('./results/' + save_name + '/',
                        'case_' + str(id) + '_gt_' + str(20) + '.pdf'))
                    plt.close()
                    plt.figure()
                    plt.axis('off')
                    plt.imshow((im[0, :, 0].reshape(64, 64) - yy[0, :, t].
                        reshape(64, 64)).detach().cpu().numpy(), cmap=
                        'coolwarm')
                    plt.colorbar()
                    plt.clim(-2, 2)
                    plt.savefig(os.path.join('./results/' + save_name + '/',
                        'case_' + str(id) + '_error_' + str(20) + '.pdf'))
                    plt.close()
                test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(
                    bsz, -1)).item()
            print(test_l2_full / ntest)
    else:
        for ep in range(args.epochs):
            model.train()
            train_l2_step = 0
            train_l2_full = 0
            for x, fx, yy in train_loader:
                loss = 0
                x, fx, yy = x.to(device), fx.to(device), yy.to(device)
                bsz = tuple(x.shape)[0]
                for t in range(0, T, step):
                    y = yy[..., t:t + step]
                    im = model(x, fx=fx)
                    loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                    if t == 0:
                        pred = im
                    else:
                        pred = paddle.concat(x=(pred, im), axis=-1)
                    fx = paddle.concat(x=(fx[..., step:], y), axis=-1)
                train_l2_step += loss.item()
                train_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(
                    bsz, -1)).item()
                optimizer.clear_gradients(set_to_zero=False)
                loss.backward()
                if args.max_grad_norm is not None:
                    paddle.nn.utils.clip_grad_norm_(parameters=model.
                        parameters(), max_norm=args.max_grad_norm)
                optimizer.step()
                scheduler.step()
            test_l2_step = 0
            test_l2_full = 0
            model.eval()
            with paddle.no_grad():
                for x, fx, yy in test_loader:
                    loss = 0
                    x, fx, yy = x.to(device), fx.to(device), yy.to(device)
                    bsz = tuple(x.shape)[0]
                    for t in range(0, T, step):
                        y = yy[..., t:t + step]
                        im = model(x, fx=fx)
                        loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                        if t == 0:
                            pred = im
                        else:
                            pred = paddle.concat(x=(pred, im), axis=-1)
                        fx = paddle.concat(x=(fx[..., step:], im), axis=-1)
                    test_l2_step += loss.item()
                    test_l2_full += myloss(pred.reshape(bsz, -1), yy.
                        reshape(bsz, -1)).item()
            print(
                'Epoch {} , train_step_loss:{:.5f} , train_full_loss:{:.5f} , test_step_loss:{:.5f} , test_full_loss:{:.5f}'
                .format(ep, train_l2_step / ntrain / (T / step), 
                train_l2_full / ntrain, test_l2_step / ntest / (T / step), 
                test_l2_full / ntest))
            if ep % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                print('save model')
                paddle.save(obj=model.state_dict(), path=os.path.join(
                    './checkpoints', save_name + '.pt'))
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        print('save model')
        paddle.save(obj=model.state_dict(), path=os.path.join(
            './checkpoints', save_name + '.pt'))


if __name__ == '__main__':
    main()
