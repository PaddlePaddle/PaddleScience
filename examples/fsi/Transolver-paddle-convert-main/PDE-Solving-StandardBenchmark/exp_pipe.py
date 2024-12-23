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
parser.add_argument('--n-hidden', type=int, default=128, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=8, help='layers')
parser.add_argument('--n-heads', type=int, default=8)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')
parser.add_argument('--max_grad_norm', type=float, default=0.1)
parser.add_argument('--downsamplex', type=int, default=1)
parser.add_argument('--downsampley', type=int, default=1)
parser.add_argument('--mlp_ratio', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--slice_num', type=int, default=64)
parser.add_argument('--eval', type=int, default=1)
parser.add_argument('--save_name', type=str, default='pipe_Transolver')
parser.add_argument('--data_path', type=str, default='data/fno/pipe')
args = parser.parse_args()
eval = args.eval
save_name = args.save_name
import numpy as np
from tqdm import *
from utils.testloss import TestLoss
from model_dict import get_model
from utils.normalizer import UnitTransformer
n_gpu = paddle.device.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and paddle.device.cuda.device_count() >= 1
device = str(f'cuda:{args.gpu}' if use_cuda else 'cpu').replace('cuda', 'gpu')


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
    INPUT_X = args.data_path + '/Pipe_X.npy'
    INPUT_Y = args.data_path + '/Pipe_Y.npy'
    OUTPUT_Sigma = args.data_path + '/Pipe_Q.npy'
    ntrain = 1000
    ntest = 200
    N = 1200
    r1 = args.downsamplex
    r2 = args.downsampley
    s1 = int((129 - 1) / r1 + 1)
    s2 = int((129 - 1) / r2 + 1)
    inputX = np.load(INPUT_X)
    inputX = paddle.to_tensor(data=inputX, dtype='float32')
    inputY = np.load(INPUT_Y)
    inputY = paddle.to_tensor(data=inputY, dtype='float32')
    input = paddle.stack(x=[inputX, inputY], axis=-1)
    output = np.load(OUTPUT_Sigma)[:, 0]
    output = paddle.to_tensor(data=output, dtype='float32')
    print(tuple(input.shape), tuple(output.shape))
    x_train = input[:N][:ntrain, ::r1, ::r2][:, :s1, :s2]
    y_train = output[:N][:ntrain, ::r1, ::r2][:, :s1, :s2]
    x_test = input[:N][-ntest:, ::r1, ::r2][:, :s1, :s2]
    y_test = output[:N][-ntest:, ::r1, ::r2][:, :s1, :s2]
    x_train = x_train.reshape(ntrain, -1, 2)
    x_test = x_test.reshape(ntest, -1, 2)
    y_train = y_train.reshape(ntrain, -1)
    y_test = y_test.reshape(ntest, -1)
    x_normalizer = UnitTransformer(x_train)
    y_normalizer = UnitTransformer(y_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    y_train = y_normalizer.encode(y_train)
    x_normalizer.to(device)
    y_normalizer.to(device)
    train_loader = paddle.io.DataLoader(dataset=paddle.io.TensorDataset([
        x_train, x_train, y_train]), batch_size=args.batch_size, shuffle=True)
    test_loader = paddle.io.DataLoader(dataset=paddle.io.TensorDataset([
        x_test, x_test, y_test]), batch_size=args.batch_size, shuffle=False)
    print('Dataloading is over.')
    model = get_model(args).Model(space_dim=2, n_layers=args.n_layers,
        n_hidden=args.n_hidden, dropout=args.dropout, n_head=args.n_heads,
        Time_Input=False, mlp_ratio=args.mlp_ratio, fun_dim=0, out_dim=1,
        slice_num=args.slice_num, ref=args.ref, unified_pos=args.
        unified_pos, H=s1, W=s2).to(device)
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
        paddle.save(obj=model.state_dict(), path=os.path.join(
            './checkpoints', save_name + '_resave' + '.pt'))
        model.eval()
        if not os.path.exists('./results/' + save_name + '/'):
            os.makedirs('./results/' + save_name + '/')
        rel_err = 0.0
        showcase = 10
        id = 0
        with paddle.no_grad():
            for pos, fx, y in test_loader:
                id += 1
                x, fx, y = pos.to(device), fx.to(device), y.to(device)
                out = model(x, None).squeeze(axis=-1)
                out = y_normalizer.decode(out)
                tl = myloss(out, y).item()
                rel_err += tl
                if id < showcase:
                    print(id)
                    plt.axis('off')
                    plt.pcolormesh(x[0, :, 0].reshape(129, 129).detach().
                        cpu().numpy(), x[0, :, 1].reshape(129, 129).detach(
                        ).cpu().numpy(), np.zeros([129, 129]), shading=
                        'auto', edgecolors='black', linewidths=0.1)
                    plt.colorbar()
                    plt.savefig(os.path.join('./results/' + save_name + '/',
                        'input_' + str(id) + '.pdf'), bbox_inches='tight',
                        pad_inches=0)
                    plt.close()
                    plt.axis('off')
                    plt.pcolormesh(x[0, :, 0].reshape(129, 129).detach().
                        cpu().numpy(), x[0, :, 1].reshape(129, 129).detach(
                        ).cpu().numpy(), out[0, :].reshape(129, 129).detach
                        ().cpu().numpy(), shading='auto', cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(0, 0.3)
                    plt.savefig(os.path.join('./results/' + save_name + '/',
                        'pred_' + str(id) + '.pdf'), bbox_inches='tight',
                        pad_inches=0)
                    plt.close()
                    plt.axis('off')
                    plt.pcolormesh(x[0, :, 0].reshape(129, 129).detach().
                        cpu().numpy(), x[0, :, 1].reshape(129, 129).detach(
                        ).cpu().numpy(), y[0, :].reshape(129, 129).detach()
                        .cpu().numpy(), shading='auto', cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(0, 0.3)
                    plt.savefig(os.path.join('./results/' + save_name + '/',
                        'gt_' + str(id) + '.pdf'), bbox_inches='tight',
                        pad_inches=0)
                    plt.close()
                    plt.axis('off')
                    plt.pcolormesh(x[0, :, 0].reshape(129, 129).detach().
                        cpu().numpy(), x[0, :, 1].reshape(129, 129).detach(
                        ).cpu().numpy(), out[0, :].reshape(129, 129).detach
                        ().cpu().numpy() - y[0, :].reshape(129, 129).detach
                        ().cpu().numpy(), shading='auto', cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(-0.02, 0.02)
                    plt.savefig(os.path.join('./results/' + save_name + '/',
                        'error_' + str(id) + '.pdf'), bbox_inches='tight',
                        pad_inches=0)
                    plt.close()
        rel_err /= ntest
        print('rel_err:{}'.format(rel_err))
    else:
        for ep in range(args.epochs):
            model.train()
            train_loss = 0
            for pos, fx, y in train_loader:
                x, fx, y = pos.to(device), fx.to(device), y.to(device)
                optimizer.clear_gradients(set_to_zero=False)
                out = model(x, None).squeeze(axis=-1)
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
                loss = myloss(out, y)
                loss.backward()
                if args.max_grad_norm is not None:
                    paddle.nn.utils.clip_grad_norm_(parameters=model.
                        parameters(), max_norm=args.max_grad_norm)
                optimizer.step()
                train_loss += loss.item()
                scheduler.step()
            train_loss = train_loss / ntrain
            print('Epoch {} Train loss : {:.5f}'.format(ep, train_loss))
            model.eval()
            rel_err = 0.0
            with paddle.no_grad():
                for pos, fx, y in test_loader:
                    x, fx, y = pos.to(device), fx.to(device), y.to(device)
                    out = model(x, None).squeeze(axis=-1)
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
