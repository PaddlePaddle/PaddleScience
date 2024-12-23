import os
import paddle
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import *
from utils.testloss import TestLoss
from model_dict import get_model
from utils.normalizer import UnitTransformer
parser = argparse.ArgumentParser('Training Transformer')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-05)
parser.add_argument('--model', type=str, default='Transolver_Irregular_Mesh')
parser.add_argument('--n-hidden', type=int, default=128, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=8, help='layers')
parser.add_argument('--n-heads', type=int, default=8)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')
parser.add_argument('--max_grad_norm', type=float, default=0.1)
parser.add_argument('--downsample', type=int, default=5)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--slice_num', type=int, default=64)
parser.add_argument('--eval', type=int, default=1)
parser.add_argument('--save_name', type=str, default='elas_Transolver')
parser.add_argument('--data_path', type=str, default='data/fno')
args = parser.parse_args()
eval = args.eval
save_name = args.save_name

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
    ntrain = args.ntrain
    ntest = 200
    PATH_Sigma = (args.data_path +
        '/elasticity/Meshes/Random_UnitCell_sigma_10.npy')
    PATH_XY = args.data_path + '/elasticity/Meshes/Random_UnitCell_XY_10.npy'
    input_s = np.load(PATH_Sigma)
    input_s = paddle.to_tensor(data=input_s, dtype='float32').transpose(perm
        =[1, 0])
    input_xy = np.load(PATH_XY)
    input_xy = paddle.to_tensor(data=input_xy, dtype='float32').transpose(perm
        =[2, 0, 1])
    train_s = input_s[:ntrain]
    test_s = input_s[-ntest:]
    train_xy = input_xy[:ntrain]
    test_xy = input_xy[-ntest:]
    print(tuple(input_s.shape), tuple(input_xy.shape))
    y_normalizer = UnitTransformer(train_s)
    train_s = y_normalizer.encode(train_s)
    y_normalizer.to(device)
    train_loader = paddle.io.DataLoader(dataset=paddle.io.TensorDataset([
        train_xy, train_xy, train_s]), batch_size=args.batch_size, shuffle=True
        )
    test_loader = paddle.io.DataLoader(dataset=paddle.io.TensorDataset([
        test_xy, test_xy, test_s]), batch_size=args.batch_size, shuffle=False)
    print('Dataloading is over.')
    model = get_model(args).Model(space_dim=2, n_layers=args.n_layers,
        n_hidden=args.n_hidden, dropout=args.dropout, n_head=args.n_heads,
        Time_Input=False, mlp_ratio=args.mlp_ratio, fun_dim=0, out_dim=1,
        slice_num=args.slice_num, ref=args.ref, unified_pos=args.unified_pos
        ).to(device)
    optimizer = paddle.optimizer.AdamW(parameters=model.parameters(),
        learning_rate=args.lr, weight_decay=args.weight_decay)
    print(args)
    print(model)
    count_parameters(model)
    tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(T_max=args.epochs,
        learning_rate=optimizer.get_lr())
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr
    myloss = TestLoss(size_average=False)
    if eval:
        model.set_state_dict(state_dict=paddle.load(path=str(
            './checkpoints/' + save_name + '.pt')))
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
                    plt.scatter(x=fx[0, :, 0].detach().cpu().numpy(), y=fx[
                        0, :, 1].detach().cpu().numpy(), c=y[0, :].detach()
                        .cpu().numpy(), cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(0, 1000)
                    plt.savefig(os.path.join('./results/' + save_name + '/',
                        'gt_' + str(id) + '.pdf'), bbox_inches='tight',
                        pad_inches=0)
                    plt.close()
                    plt.axis('off')
                    plt.scatter(x=fx[0, :, 0].detach().cpu().numpy(), y=fx[
                        0, :, 1].detach().cpu().numpy(), c=out[0, :].detach
                        ().cpu().numpy(), cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(0, 1000)
                    plt.savefig(os.path.join('./results/' + save_name + '/',
                        'pred_' + str(id) + '.pdf'), bbox_inches='tight',
                        pad_inches=0)
                    plt.close()
                    plt.axis('off')
                    plt.scatter(x=fx[0, :, 0].detach().cpu().numpy(), y=fx[
                        0, :, 1].detach().cpu().numpy(), c=(y[0, :] - out[0,
                        :]).detach().cpu().numpy(), cmap='coolwarm')
                    plt.clim(-8, 8)
                    plt.colorbar()
                    plt.savefig(os.path.join('./results/' + save_name + '/',
                        'error_' + str(id) + '.pdf'), bbox_inches='tight',
                        pad_inches=0)
                    plt.close()
        rel_err /= ntest
        print('rel_err : {}'.format(rel_err))
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
            print('rel_err : {}'.format(rel_err))
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
