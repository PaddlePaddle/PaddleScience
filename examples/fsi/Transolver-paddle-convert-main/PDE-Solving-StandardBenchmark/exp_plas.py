import sys

# sys.path.append('../../utils')
from utils import paddle_aux
import os
import paddle
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('Training Transformer')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-05)
parser.add_argument('--model', type=str, default='Transolver_Structured_Mesh_2D')
parser.add_argument('--n-hidden', type=int, default=64, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=3, help='layers')
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--downsamplex', type=int, default=1)
parser.add_argument('--downsampley', type=int, default=1)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--slice_num', type=int, default=32)
parser.add_argument('--eval', type=int, default=1)
parser.add_argument('--save_name', type=str, default='plas_Transolver')
parser.add_argument('--data_path', type=str, default=
'data/fno/plas_N987_T20.mat')
args = parser.parse_args()
eval = args.eval
save_name = args.save_name
import numpy as np
import scipy.io as scio
from tqdm import *
from utils.testloss import TestLoss
from model_dict import get_model
from utils.normalizer import UnitTransformer

n_gpu = paddle.device.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and paddle.device.cuda.device_count() >= 1
device = f'gpu:{args.gpu}' if use_cuda else 'cpu'


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not not parameter.stop_gradient:
            continue
        params = parameter.size
        total_params += params
    print(f'Total Trainable Params: {total_params}')
    return total_params


def random_collate_fn(batch):
    shuffled_batch = []
    shuffled_u = None
    shuffled_t = None
    shuffled_a = None
    shuffled_pos = None
    for item in batch:
        pos = item[0]
        t = item[1]
        a = item[2]
        u = item[3]
        num_timesteps = t.shape[0]
        permuted_indices = paddle.randperm(n=num_timesteps)
        t = t[permuted_indices]
        u = u[..., permuted_indices]
        if shuffled_t is None:
            shuffled_pos = pos.unsqueeze(axis=0)
            shuffled_t = t.unsqueeze(axis=0)
            shuffled_u = u.unsqueeze(axis=0)
            shuffled_a = a.unsqueeze(axis=0)
        else:
            shuffled_pos = paddle.concat(x=(shuffled_pos, pos.unsqueeze(
                axis=0)), axis=0)
            shuffled_t = paddle.concat(x=(shuffled_t, t.unsqueeze(axis=0)),
                                       axis=0)
            shuffled_u = paddle.concat(x=(shuffled_u, u.unsqueeze(axis=0)),
                                       axis=0)
            shuffled_a = paddle.concat(x=(shuffled_a, a.unsqueeze(axis=0)),
                                       axis=0)
    shuffled_batch.append(shuffled_pos)
    shuffled_batch.append(shuffled_t)
    shuffled_batch.append(shuffled_a)
    shuffled_batch.append(shuffled_u)
    return shuffled_batch


def main():
    DATA_PATH = args.data_path
    N = 987
    ntrain = 900
    ntest = 80
    s1 = 101
    s2 = 31
    T = 20
    Deformation = 4
    r1 = 1
    r2 = 1
    s1 = int((s1 - 1) / r1 + 1)
    s2 = int((s2 - 1) / r2 + 1)
    data = scio.loadmat(DATA_PATH)
    input = paddle.to_tensor(data=data['input'], dtype='float32')
    output = paddle.to_tensor(data=data['output'], dtype='float32').transpose(
        perm=paddle_aux.transpose_aux_func(paddle.to_tensor(data=data[
            'output'], dtype='float32').ndim, -2, -1))
    print(tuple(input.shape), tuple(output.shape))
    x_train = input[:ntrain, ::r1][:, :s1].reshape(ntrain, s1, 1).tile(
        repeat_times=[1, 1, s2])
    x_train = x_train.reshape(ntrain, -1, 1)
    y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
    y_train = y_train.reshape(ntrain, -1, Deformation, T)
    x_test = input[-ntest:, ::r1][:, :s1].reshape(ntest, s1, 1).tile(
        repeat_times=[1, 1, s2])
    x_test = x_test.reshape(ntest, -1, 1)
    y_test = output[-ntest:, ::r1, ::r2][:, :s1, :s2]
    y_test = y_test.reshape(ntest, -1, Deformation, T)
    print(tuple(x_train.shape), tuple(y_train.shape))
    x_normalizer = UnitTransformer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    x_normalizer.to(device)
    x = np.linspace(0, 1, s1)
    y = np.linspace(0, 1, s2)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.flatten(), y.flatten()]
    pos = paddle.to_tensor(data=pos, dtype='float32').unsqueeze(axis=0)
    pos_train = pos.tile(repeat_times=[ntrain, 1, 1])
    pos_test = pos.tile(repeat_times=[ntest, 1, 1])
    print('Dataloading is over.')
    t = np.linspace(0, 1, T)
    t = paddle.to_tensor(data=t, dtype='float32').unsqueeze(axis=0)
    t_train = t.tile(repeat_times=[ntrain, 1])
    t_test = t.tile(repeat_times=[ntest, 1])
    train_loader = paddle.io.DataLoader(dataset=paddle.io.TensorDataset([
        pos_train, t_train, x_train, y_train]), batch_size=args.batch_size,
        shuffle=True, collate_fn=random_collate_fn)
    test_loader = paddle.io.DataLoader(dataset=paddle.io.TensorDataset([
        pos_test, t_test, x_test, y_test]), batch_size=args.batch_size,
        shuffle=False)
    print('Dataloading is over.')
    model = get_model(args).Model(space_dim=2, n_hidden=args.n_hidden,
                                  n_layers=args.n_layers, Time_Input=True, n_head=args.n_heads,
                                  fun_dim=1, out_dim=Deformation, mlp_ratio=args.mlp_ratio, slice_num
                                  =args.slice_num, unified_pos=args.unified_pos, H=s1, W=s2).to(device)
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
        if not os.path.exists('./results/' + save_name + '/'):
            os.makedirs('./results/' + save_name + '/')
        test_l2_step = 0
        test_l2_full = 0
        showcase = 10
        id = 0
        with paddle.no_grad():
            for x, tim, fx, yy in test_loader:
                id += 1
                loss = 0
                x, fx, tim, yy = x.to(device), fx.to(device), tim.to(device
                                                                     ), yy.to(device)
                bsz = tuple(x.shape)[0]
                for t in range(T):
                    y = yy[..., t:t + 1]
                    input_T = tim[:, t:t + 1].reshape(bsz, 1)
                    im = model(x, fx, T=input_T)
                    loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                    if t == 0:
                        pred = im.unsqueeze(axis=-1)
                    else:
                        pred = paddle.concat(x=(pred, im.unsqueeze(axis=-1)
                                                ), axis=-1)
                if id < showcase:
                    print(id)
                    truth = y[0].reshape(101, 31, 4).squeeze().detach().cpu(
                    ).numpy()
                    pred_vis = im[0].reshape(101, 31, 4).squeeze().detach(
                    ).cpu().numpy()
                    truth_du = np.linalg.norm(truth[:, :, 2:], axis=-1)
                    pred_du = np.linalg.norm(pred_vis[:, :, 2:], axis=-1)
                    plt.axis('off')
                    plt.scatter(truth[:, :, 0], truth[:, :, 1], 10,
                                truth_du[:, :], cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(0, 6)
                    plt.savefig(os.path.join('./results/' + save_name + '/',
                                             'gt_' + str(id) + '.pdf'), bbox_inches='tight',
                                pad_inches=0)
                    plt.close()
                    plt.axis('off')
                    plt.scatter(pred_vis[:, :, 0], pred_vis[:, :, 1], 10,
                                pred_du[:, :], cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(0, 6)
                    plt.savefig(os.path.join('./results/' + save_name + '/',
                                             'pred_' + str(id) + '.pdf'), bbox_inches='tight',
                                pad_inches=0)
                    plt.close()
                    plt.axis('off')
                    plt.scatter(truth[:, :, 0], truth[:, :, 1], 10, pred_du
                    [:, :] - truth_du[:, :], cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(-0.2, 0.2)
                    plt.savefig(os.path.join('./results/' + save_name + '/',
                                             'error_' + str(id) + '.pdf'), bbox_inches='tight',
                                pad_inches=0)
                    plt.close()
                test_l2_step += loss.item()
                test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(
                    bsz, -1)).item()
        print('test_step_loss:{:.5f} , test_full_loss:{:.5f}'.format(
            test_l2_step / ntest / T, test_l2_full / ntest))
    else:
        for ep in range(args.epochs):
            model.train()
            train_l2_step = 0
            for x, tim, fx, yy in train_loader:
                x, fx, tim, yy = x.to(device), fx.to(device), tim.to(device
                                                                     ), yy.to(device)
                bsz = tuple(x.shape)[0]
                for t in range(T):
                    y = yy[..., t:t + 1]
                    input_T = tim[:, t:t + 1].reshape(bsz, 1)
                    im = model(x, fx, T=input_T)
                    loss = myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                    train_l2_step += loss.item()
                    optimizer.clear_gradients(set_to_zero=False)
                    loss.backward()
                    if args.max_grad_norm is not None:
                        paddle.nn.utils.clip_grad_norm_(parameters=model.
                                                        parameters(), max_norm=args.max_grad_norm)
                    optimizer.step()
                scheduler.step()
            model.eval()
            test_l2_step = 0
            test_l2_full = 0
            with paddle.no_grad():
                for x, tim, fx, yy in test_loader:
                    loss = 0
                    x, fx, tim, yy = x.to(device), fx.to(device), tim.to(device
                                                                         ), yy.to(device)
                    bsz = tuple(x.shape)[0]
                    for t in range(T):
                        y = yy[..., t:t + 1]
                        input_T = tim[:, t:t + 1].reshape(bsz, 1)
                        im = model(x, fx, T=input_T)
                        loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                        if t == 0:
                            pred = im.unsqueeze(axis=-1)
                        else:
                            pred = paddle.concat(x=(pred, im.unsqueeze(axis
                                                                       =-1)), axis=-1)
                    test_l2_step += loss.item()
                    test_l2_full += myloss(pred.reshape(bsz, -1), yy.
                                           reshape(bsz, -1)).item()
            print(
                'Epoch {} , train_step_loss:{:.5f} , test_step_loss:{:.5f} , test_full_loss:{:.5f}'
                    .format(ep, train_l2_step / ntrain / T, test_l2_step /
                            ntest / T, test_l2_full / ntest))
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
