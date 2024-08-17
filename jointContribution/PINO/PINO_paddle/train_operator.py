import yaml
from argparse import ArgumentParser
import math

import paddle
from paddle.io import DataLoader
from paddle.optimizer.lr import MultiStepDecay

from solver.random_fields import GaussianRF
from train_utils import Adam
from train_utils.datasets import NSLoader, online_loader, DarcyFlow, DarcyCombo
from train_utils.train_3d import mixed_train
from train_utils.train_2d import train_2d_operator
from models import FNO3d, FNO2d

def train_3d(args, config):
    data_config = config['data']

    # prepare dataloader for training with data
    if 'datapath2' in data_config:
        loader = NSLoader(datapath1=data_config['datapath'], datapath2=data_config['datapath2'],
                          nx=data_config['nx'], nt=data_config['nt'],
                          sub=data_config['sub'], sub_t=data_config['sub_t'],
                          N=data_config['total_num'],
                          t_interval=data_config['time_interval'])
    else:
        loader = NSLoader(datapath1=data_config['datapath'],
                          nx=data_config['nx'], nt=data_config['nt'],
                          sub=data_config['sub'], sub_t=data_config['sub_t'],
                          N=data_config['total_num'],
                          t_interval=data_config['time_interval'])

    train_loader = loader.make_loader(data_config['n_sample'],
                                      batch_size=config['train']['batchsize'],
                                      start=data_config['offset'],
                                      train=data_config['shuffle'])
    # prepare dataloader for training with only equations
    gr_sampler = GaussianRF(2, data_config['S2'], 2 * math.pi, alpha=2.5, tau=7)
    a_loader = online_loader(gr_sampler,
                             S=data_config['S2'],
                             T=data_config['T2'],
                             time_scale=data_config['time_interval'],
                             batchsize=config['train']['batchsize'])
    # create model
    model = FNO3d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  modes3=config['model']['modes3'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'], 
                  act=config['model']['act'])
    # Load from checkpoint
    if 'ckpt' in config['train']:
        ckpt_path = config['train']['ckpt']
        ckpt = paddle.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)
    # create optimizer and learning rate scheduler
    scheduler = MultiStepDecay(learning_rate=config['train']['base_lr'],
                               milestones=config['train']['milestones'],
                               gamma=config['train']['scheduler_gamma'])
    optimizer = Adam(learning_rate=scheduler, parameters=model.parameters())
    mixed_train(model,
                train_loader,
                loader.S, loader.T,
                a_loader,
                data_config['S2'], data_config['T2'],
                optimizer,
                scheduler,
                config,
                log=args.log,
                project=config['log']['project'],
                group=config['log']['group'])

def train_2d(args, config):
    data_config = config['data']

    dataset = DarcyCombo(datapath=data_config['datapath'], 
                         nx=data_config['nx'], 
                         sub=data_config['sub'], 
                         pde_sub=data_config['pde_sub'], 
                         num=data_config['n_sample'], 
                         offset=data_config['offset'])
    train_loader = DataLoader(dataset, batch_size=config['train']['batchsize'], shuffle=True)
    model = FNO2d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'],
                  act=config['model']['act'], 
                  pad_ratio=config['model']['pad_ratio'])
    # Load from checkpoint
    if 'ckpt' in config['train']:
        ckpt_path = config['train']['ckpt']
        ckpt = paddle.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)

    scheduler = MultiStepDecay(learning_rate=config['train']['base_lr'],
                               milestones=config['train']['milestones'],
                               gamma=config['train']['scheduler_gamma'])
    optimizer = Adam(learning_rate=scheduler, parameters=model.parameters())

    train_2d_operator(model,
                      train_loader,
                      optimizer, scheduler,
                      config, rank=0, log=args.log,
                      project=config['log']['project'],
                      group=config['log']['group'])

if __name__ == '__main__':
    # parse options
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    if 'name' in config['data'] and config['data']['name'] == 'Darcy':
        train_2d(args, config)
    else:
        train_3d(args, config)
