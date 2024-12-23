import os
import paddle
import train
import argparse
from dataset.load_dataset import load_train_val_fold
from dataset.dataset import GraphDataset
from models.Transolver import Model

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=
'data/PDE_data/mlcfd_data/training_data')
parser.add_argument('--save_dir', default=
'data/PDE_data/mlcfd_data/preprocessed_data')
parser.add_argument('--fold_id', default=0, type=int)
parser.add_argument('--gpu', default=3, type=int)
parser.add_argument('--val_iter', default=10, type=int)
parser.add_argument('--cfd_config_dir', default='cfd/cfd_params.yaml')
parser.add_argument('--cfd_model', default='Transolver')
parser.add_argument('--cfd_mesh', action='store_true')
parser.add_argument('--r', default=0.2, type=float)
parser.add_argument('--weight', default=0.5, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--batch_size', default=1, type=float)
parser.add_argument('--nb_epochs', default=200, type=float)
parser.add_argument('--preprocessed', default=1, type=int)
args = parser.parse_args()
print(args)
hparams = {'lr': args.lr, 'batch_size': args.batch_size, 'nb_epochs': args.
    nb_epochs}
n_gpu = paddle.device.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and paddle.device.cuda.device_count() >= 1
device = str(f'cuda:{args.gpu}' if use_cuda else 'cpu').replace('cuda', 'gpu')
print(device)
train_data, val_data, coef_norm = load_train_val_fold(args, preprocessed=
args.preprocessed)
train_ds = GraphDataset(train_data, use_cfd_mesh=args.cfd_mesh, r=args.r)
val_ds = GraphDataset(val_data, use_cfd_mesh=args.cfd_mesh, r=args.r)
if args.cfd_model == 'Transolver':
    model = Model(n_hidden=256, n_layers=8, space_dim=7, fun_dim=0, n_head=
    8, mlp_ratio=2, out_dim=4, slice_num=32, unified_pos=0).to(device)
path = (
    f'metrics/{args.cfd_model}/{args.fold_id}/{args.nb_epochs}_{args.weight}')
if not os.path.exists(path):
    os.makedirs(path)
model = train.main(device, train_ds, val_ds, model, hparams, path, val_iter
=args.val_iter, reg=args.weight, coef_norm=coef_norm)
