import paddle
import os
import yaml, json
import utils.metrics as metrics
from dataset.dataset import Dataset
import argparse
import numpy as np
from models.Transolver import Transolver

parser = argparse.ArgumentParser()
parser.add_argument('--my_path', default='/data/path', type=str)
parser.add_argument('--save_path', default='./', type=str)
parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')

args = parser.parse_args()

n_gpu = paddle.device.cuda.device_count()
use_cuda = n_gpu > 0 and 0 <= args.gpu < n_gpu
device = f'gpu:{args.gpu}' if use_cuda else 'cpu'

print(device)
if use_cuda:
    print('Using GPU')
else:
    print('Using CPU')

mod = Transolver(
    n_hidden=256,
    n_layers=8,
    space_dim=7,
    fun_dim=0,
    n_head=8,
    mlp_ratio=2,
    out_dim=4,
    slice_num=32,
    unified_pos=1,
    device=device
).to(device)

data_root_dir = args.my_path
ckpt_root_dir = args.save_path
tasks = ['full']

for task in tasks:
    print('Generating results for task ' + task + '...')
    s = task + '_test' if task != 'scarce' else 'full_test'
    s_train = task + '_train'
    data_dir = os.path.join(data_root_dir, 'Dataset')
    with open(os.path.join(data_dir, 'manifest.json'), 'r') as f:
        manifest = json.load(f)

    manifest_train = manifest[s_train]
    n = int(0.1 * len(manifest_train))

    train_dataset = manifest_train[:-n]

    _, coef_norm = Dataset(train_dataset, norm=True, sample=None, my_path=
        data_dir)
    model_names = ['Transolver']
    models = []
    hparams = []
    for model in model_names:
        model_path = os.path.join(ckpt_root_dir, 'metrics', task, model, "Transolver.pdparams")

        mod.set_state_dict(paddle.load(model_path))

        # mod = [m.to(device) for m in mod]

        models.append(mod)

        with open('params.yaml', 'r') as f:
            hparam = yaml.safe_load(f)[model]
            hparams.append(hparam)
    results_dir = os.path.join(ckpt_root_dir, 'scores', task)

    coefs = metrics.Results_test(device, models, hparams, coef_norm,
        data_dir, results_dir, n_test=3, criterion='MSE', s=s)
    # print(coefs)
    np.save(os.path.join(results_dir, 'true_coefs'), coefs[0])
    np.save(os.path.join(results_dir, 'pred_coefs_mean'), coefs[1])
    np.save(os.path.join(results_dir, 'pred_coefs_std'), coefs[2])
    for n, file in enumerate(coefs[3]):
        np.save(os.path.join(results_dir, 'true_surf_coefs_' + str(n)), file)
    for n, file in enumerate(coefs[4]):
        np.save(os.path.join(results_dir, 'surf_coefs_' + str(n)), file)
    np.save(os.path.join(results_dir, 'true_bls'), coefs[5])
    np.save(os.path.join(results_dir, 'bls'), coefs[6])
