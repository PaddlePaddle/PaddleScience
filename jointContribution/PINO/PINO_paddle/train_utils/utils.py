import os
import numpy as np
import paddle

def vor2vel(w, L=2 * np.pi):
    '''
    Convert vorticity into velocity
    Args:
        w: vorticity with shape (batchsize, num_x, num_y, num_t)

    Returns:
        ux, uy with the same shape
    '''
    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)
    nt = w.size(3)
    device = w.device
    w = w.reshape(batchsize, nx, ny, nt)

    w_h = paddle.fft.fft2(w, axes=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = paddle.cat((paddle.arange(start=0, end=k_max, step=1, device=device),
                     paddle.arange(start=-k_max, end=0, step=1, device=device)), 0) \
        .reshape(N, 1).repeat(1, N).reshape(1, N, N, 1)
    k_y = paddle.cat((paddle.arange(start=0, end=k_max, step=1, device=device),
                     paddle.arange(start=-k_max, end=0, step=1, device=device)), 0) \
        .reshape(1, N).repeat(N, 1).reshape(1, N, N, 1)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    ux_h = 2 * np.pi / L * 1j * k_y * f_h
    uy_h = -2 * np.pi / L * 1j * k_x * f_h

    ux = paddle.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
    uy = paddle.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
    return ux, uy

def get_sample(N, T, s, p, q):
    # sample p nodes from Initial Condition, p nodes from Boundary Condition, q nodes from Interior

    # sample IC
    index_ic = paddle.randint(s, size=(N, p))
    sample_ic_t = paddle.zeros(N, p)
    sample_ic_x = index_ic/s

    # sample BC
    sample_bc = paddle.rand(size=(N, p//2))
    sample_bc_t =  paddle.cat([sample_bc, sample_bc],dim=1)
    sample_bc_x = paddle.cat([paddle.zeros(N, p//2), paddle.ones(N, p//2)],dim=1)

    sample_i_t = -paddle.cos(paddle.rand(size=(N, q))*np.pi/2) + 1
    sample_i_x = paddle.rand(size=(N,q))

    sample_t = paddle.cat([sample_ic_t, sample_bc_t, sample_i_t], dim=1).cuda()
    sample_t.requires_grad = True
    sample_x = paddle.cat([sample_ic_x, sample_bc_x, sample_i_x], dim=1).cuda()
    sample_x.requires_grad = True
    sample = paddle.stack([sample_t, sample_x], dim=-1).reshape(N, (p+p+q), 2)
    return sample, sample_t, sample_x, index_ic.long()

def get_grid(N, T, s):
    gridt = paddle.tensor(np.linspace(0, 1, T), dtype=paddle.float).reshape(1, T, 1).repeat(N, 1, s).cuda()
    gridt.requires_grad = True
    gridx = paddle.tensor(np.linspace(0, 1, s+1)[:-1], dtype=paddle.float).reshape(1, 1, s).repeat(N, T, 1).cuda()
    gridx.requires_grad = True
    grid = paddle.stack([gridt, gridx], dim=-1).reshape(N, T*s, 2)
    return grid, gridt, gridx

def get_2dgrid(S):
    '''
    get array of points on 2d grid in (0,1)^2
    Args:
        S: resolution

    Returns:
        points: flattened grid, ndarray (N, 2)
    '''
    xarr = np.linspace(0, 1, S)
    yarr = np.linspace(0, 1, S)
    xx, yy = np.meshgrid(xarr, yarr, indexing='ij')
    points = np.stack([xx.ravel(), yy.ravel()], axis=0).T
    return points

def paddle2dgrid(num_x, num_y, bot=(0,0), top=(1,1)):
    x_bot, y_bot = bot
    x_top, y_top = top
    x_arr = paddle.linspace(x_bot, x_top, num=num_x)
    y_arr = paddle.linspace(y_bot, y_top, num=num_y)
    xx, yy = paddle.meshgrid(x_arr, y_arr, indexing='ij')
    mesh = paddle.stack([xx, yy], dim=2)
    return mesh

def get_grid3d(S, T, time_scale=1.0, device='cpu'):
    gridx = paddle.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=paddle.float, device=device)
    gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
    gridy = paddle.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=paddle.float, device=device)
    gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
    gridt = paddle.tensor(np.linspace(0, 1 * time_scale, T), dtype=paddle.float, device=device)
    gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])
    return gridx, gridy, gridt

def convert_ic(u0, N, S, T, time_scale=1.0):
    u0 = u0.reshape(N, S, S, 1, 1).repeat([1, 1, 1, T, 1])
    gridx, gridy, gridt = get_grid3d(S, T, time_scale=time_scale, device=u0.device)
    a_data = paddle.cat((gridx.repeat([N, 1, 1, 1, 1]), gridy.repeat([N, 1, 1, 1, 1]),
                        gridt.repeat([N, 1, 1, 1, 1]), u0), dim=-1)
    return a_data

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def set_grad(tensors, flag=True):
    for p in tensors:
        p.requires_grad = flag

def zero_grad(params):
    '''
    set grad field to 0
    '''
    if isinstance(params, paddle.Tensor):
        if params.grad is not None:
            params.grad.zero_()
    else:
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

def count_params(net):
    count = 0
    for p in net.parameters():
        count += p.numel()
    return count

def save_checkpoint(path, name, model, optimizer=None):
    ckpt_dir = 'checkpoints/%s/' % path
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    try:
        model_state_dict = model.module.state_dict()
    except AttributeError:
        model_state_dict = model.state_dict()

    if optimizer is not None:
        optim_dict = optimizer.state_dict()
    else:
        optim_dict = 0.0

    paddle.save({
        'model': model_state_dict,
        'optim': optim_dict
    }, ckpt_dir + name)
    print('Checkpoint is saved at %s' % ckpt_dir + name)

def save_ckpt(path, model, optimizer=None, scheduler=None):
    model_state = model.state_dict()
    if optimizer:
        optim_state = optimizer.state_dict()
    else:
        optim_state = None
    
    if scheduler:
        scheduler_state = scheduler.state_dict()
    else:
        scheduler_state = None
    paddle.save({
        'model': model_state, 
        'optim': optim_state, 
        'scheduler': scheduler_state
    }, path)
    print(f'Checkpoint is saved to {path}')

def dict2str(log_dict):
    res = ''
    for key, value in log_dict.items():
        res += f'{key}: {value}|'
    return res