import sys
import os
import paddle
import paddle.nn.functional as F
import argparse
import numpy as np
import scipy.io as scio
from tqdm import *
from utils.testloss import TestLoss
from einops import rearrange
from model_dict import get_model
from utils.normalizer import UnitTransformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser('Training Transolver')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-05)
parser.add_argument('--model', type=str, default='Transolver_Structured_Mesh_2D')
parser.add_argument('--n-hidden', type=int, default=128, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=8, help='layers')
parser.add_argument('--n-heads', type=int, default=8)
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')
parser.add_argument('--max_grad_norm', type=float, default=0.1)
parser.add_argument('--downsample', type=int, default=5)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--unified_pos', type=int, default=1)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--slice_num', type=int, default=64)
parser.add_argument('--eval', type=int, default=1)
parser.add_argument('--save_name', type=str, default='darcy_UniPDE')
parser.add_argument('--data_path', type=str, default='data/fno')
args = parser.parse_args()
n_gpu = paddle.device.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and paddle.device.cuda.device_count() >= 1
device = str(f'cuda:{args.gpu}' if use_cuda else 'cpu').replace('cuda', 'gpu')
train_path = args.data_path + '/piececonst_r421_N1024_smooth1.mat'
test_path = args.data_path + '/piececonst_r421_N1024_smooth2.mat'
ntrain = args.ntrain
ntest = 200
epochs = 500
eval = args.eval
save_name = args.save_name

paddle.disable_signal_handler()

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not not parameter.stop_gradient:
            continue
        params = parameter.size
        total_params += params
    print(f'Total Trainable Params: {total_params}')
    return total_params


def central_diff(x: paddle.Tensor, h, resolution):
    x = rearrange(x, 'b (h w) c -> b h w c', h=resolution, w=resolution)
    x = F.pad(x, pad=(1, 1, 1, 1), mode='constant', value=0)
    grad_x = (x[:, 1:-1, 2:, :] - x[:, 1:-1, :-2, :]) / (2 * h)
    grad_y = (x[:, 2:, 1:-1, :] - x[:, :-2, 1:-1, :]) / (2 * h)
    return grad_x, grad_y


def main():
    r = args.downsample
    h = int((421 - 1) / r + 1)
    s = h
    dx = 1.0 / s
    train_data = scio.loadmat(train_path)
    x_train = train_data['coeff'][:ntrain, ::r, ::r][:, :s, :s]
    x_train = x_train.reshape(ntrain, -1)
    x_train = paddle.to_tensor(data=x_train).astype(dtype='float32')
    y_train = train_data['sol'][:ntrain, ::r, ::r][:, :s, :s]
    y_train = y_train.reshape(ntrain, -1)
    y_train = paddle.to_tensor(data=y_train)
    test_data = scio.loadmat(test_path)
    x_test = test_data['coeff'][:ntest, ::r, ::r][:, :s, :s]
    x_test = x_test.reshape(ntest, -1)
    x_test = paddle.to_tensor(data=x_test).astype(dtype='float32')
    y_test = test_data['sol'][:ntest, ::r, ::r][:, :s, :s]
    y_test = y_test.reshape(ntest, -1)
    y_test = paddle.to_tensor(data=y_test)
    x_normalizer = UnitTransformer(x_train)
    y_normalizer = UnitTransformer(y_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    y_train = y_normalizer.encode(y_train)
    x_normalizer.to(device)
    y_normalizer.to(device)
    x = np.linspace(0, 1, s)
    y = np.linspace(0, 1, s)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.flatten(), y.flatten()]
    pos = paddle.to_tensor(data=pos, dtype='float32').unsqueeze(axis=0)
    pos_train = pos.tile(repeat_times=[ntrain, 1, 1])
    pos_test = pos.tile(repeat_times=[ntest, 1, 1])
    print('Dataloading is over.')
    train_loader = paddle.io.DataLoader(dataset=paddle.io.TensorDataset([
        pos_train, x_train, y_train]), batch_size=args.batch_size, shuffle=True
        )
    test_loader = paddle.io.DataLoader(dataset=paddle.io.TensorDataset([
        pos_test, x_test, y_test]), batch_size=args.batch_size, shuffle=False)
    model = get_model(args).Model(space_dim=2, n_layers=args.n_layers,
        n_hidden=args.n_hidden, dropout=args.dropout, n_head=args.n_heads,
        Time_Input=False, mlp_ratio=args.mlp_ratio, fun_dim=1, out_dim=1,
        slice_num=args.slice_num, ref=args.ref, unified_pos=args.
        unified_pos, H=s, W=s).to(device)
    optimizer = paddle.optimizer.AdamW(parameters=model.parameters(),
        learning_rate=args.lr, weight_decay=args.weight_decay)
    print(args)
    print(model)
    count_parameters(model)
    tmp_lr = paddle.optimizer.lr.OneCycleLR(total_steps=len(train_loader) *
        epochs, max_learning_rate=args.lr)
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr
    myloss = TestLoss(size_average=False)
    de_x = TestLoss(size_average=False)
    de_y = TestLoss(size_average=False)
    if eval:
        print('model evaluation')
        print(s, s)
        model.set_state_dict(state_dict=paddle.load(path=str(
            './checkpoints/' + save_name + '.pt')))
        model.eval()
        showcase = 10
        id = 0
        if not os.path.exists('./results/' + save_name + '/'):
            os.makedirs('./results/' + save_name + '/')
        with paddle.no_grad():
            rel_err = 0.0
            with paddle.no_grad():
                for x, fx, y in test_loader:
                    id += 1
                    x, fx, y = x.to(device), fx.to(device), y.to(device)
                    out = model(x, fx=fx.unsqueeze(axis=-1)).squeeze(axis=-1)
                    out = y_normalizer.decode(out)
                    tl = myloss(out, y).item()
                    rel_err += tl
                    if id < showcase:
                        print(id)
                        plt.figure()
                        plt.axis('off')
                        plt.imshow(out[0, :].reshape(85, 85).detach().cpu()
                            .numpy(), cmap='coolwarm')
                        plt.colorbar()
                        plt.savefig(os.path.join('./results/' + save_name +
                            '/', 'case_' + str(id) + '_pred.pdf'))
                        plt.close()
                        plt.figure()
                        plt.axis('off')
                        plt.imshow(y[0, :].reshape(85, 85).detach().cpu().
                            numpy(), cmap='coolwarm')
                        plt.colorbar()
                        plt.savefig(os.path.join('./results/' + save_name +
                            '/', 'case_' + str(id) + '_gt.pdf'))
                        plt.close()
                        plt.figure()
                        plt.axis('off')
                        plt.imshow((y[0, :] - out[0, :]).reshape(85, 85).
                            detach().cpu().numpy(), cmap='coolwarm')
                        plt.colorbar()
                        plt.clim(-0.0005, 0.0005)
                        plt.savefig(os.path.join('./results/' + save_name +
                            '/', 'case_' + str(id) + '_error.pdf'))
                        plt.close()
                        plt.figure()
                        plt.axis('off')
                        plt.imshow(fx[0, :].unsqueeze(axis=-1).reshape(85, 
                            85).detach().cpu().numpy(), cmap='coolwarm')
                        plt.colorbar()
                        plt.savefig(os.path.join('./results/' + save_name +
                            '/', 'case_' + str(id) + '_input.pdf'))
                        plt.close()
            rel_err /= ntest
            print('rel_err:{}'.format(rel_err))
    else:
        for ep in range(args.epochs):
            model.train()
            train_loss = 0
            reg = 0
            for x, fx, y in train_loader:
                x, fx, y = x.to(device), fx.to(device), y.to(device)
                optimizer.clear_gradients(set_to_zero=False)
                out = model(x, fx=fx.unsqueeze(axis=-1)).squeeze(axis=-1)
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
                l2loss = myloss(out, y)
                out = rearrange(out.unsqueeze(axis=-1),
                    'b (h w) c -> b c h w', h=s)
                out = out[..., 1:-1, 1:-1].contiguous()
                out = F.pad(out, pad=(1, 1, 1, 1), mode='constant', value=0)
                out = rearrange(out, 'b c h w -> b (h w) c')
                gt_grad_x, gt_grad_y = central_diff(y.unsqueeze(axis=-1), dx, s
                    )
                pred_grad_x, pred_grad_y = central_diff(out, dx, s)
                deriv_loss = de_x(pred_grad_x, gt_grad_x) + de_y(pred_grad_y,
                    gt_grad_y)
                loss = 0.1 * deriv_loss + l2loss
                loss.backward()
                if args.max_grad_norm is not None:
                    paddle.nn.utils.clip_grad_norm_(parameters=model.
                        parameters(), max_norm=args.max_grad_norm)
                optimizer.step()
                train_loss += l2loss.item()
                reg += deriv_loss.item()
                scheduler.step()
            train_loss /= ntrain
            reg /= ntrain
            print('Epoch {} Reg : {:.5f} Train loss : {:.5f}'.format(ep,
                reg, train_loss))
            model.eval()
            rel_err = 0.0
            id = 0
            with paddle.no_grad():
                for x, fx, y in test_loader:
                    id += 1
                    if id == 2:
                        vis = True
                    else:
                        vis = False
                    x, fx, y = x.to(device), fx.to(device), y.to(device)
                    out = model(x, fx=fx.unsqueeze(axis=-1)).squeeze(axis=-1)
                    out = y_normalizer.decode(out)
                    tl = myloss(out, y).item()
                    rel_err += tl
            rel_err /= ntest
            print('rel_err:{}'.format(rel_err))
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
