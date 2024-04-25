import paddle
import ppsci
import logging
import hydra
from omegaconf import DictConfig
from functions import *
from utils import *
import os
import time
import shutil
from tensorboardX import SummaryWriter
from ppsci.arch.ddpm import DDPM
from ppsci.arch import DDPM
def train(cfg):
    #initilize
    args = cfg.train
    config = cfg
    args.log_path = os.path.join(args.exp, 'logs', args.doc)
    tb_path = os.path.join(args.exp, 'tensorboard', args.doc)
    if os.path.exists(args.log_path):
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input('Folder already exists. Overwrite? (Y/N)')
            if response.upper() == 'Y':
                overwrite = True
        if overwrite:
            shutil.rmtree(args.log_path)
            shutil.rmtree(tb_path)
            os.makedirs(args.log_path)
            if os.path.exists(tb_path):
                shutil.rmtree(tb_path)
        else:
            print('Folder exists. Program halted.')
            sys.exit(0)
    else:
        os.makedirs(args.log_path)
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.log_path,
        'stdout.txt'))
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)
    device = str('cuda').replace('cuda', 'gpu'
        ) if paddle.device.cuda.device_count() >= 1 else str('cpu').replace(
        'cuda', 'gpu')
    logging.info('Using device: {}'.format(device))
    # new_config.place = device
    paddle.seed(seed=args.seed)
    np.random.seed(args.seed)

    #prepare
    logging.info('Writing log file to {}'.format(args.log_path))
    logging.info('Exp instance id = {}'.format(os.getpid()))
    logging.info('Exp comment = {}'.format(args.comment))
   
    #train
    model_var_type = config.model.var_type
    betas = get_beta_schedule_train(beta_schedule=config.diffusion.
        beta_schedule, beta_start=config.diffusion.beta_start, beta_end
        =config.diffusion.beta_end, num_diffusion_timesteps=config.
        diffusion.num_diffusion_timesteps)
    betas = paddle.to_tensor(data=betas).astype(dtype=
        'float32').to(device)
    num_timesteps = betas.shape[0]
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)
    alphas_cumprod_prev = paddle.concat(x=[paddle.ones(shape=[1]).to(
        device), alphas_cumprod[:-1]], axis=0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 -
        alphas_cumprod)
    if model_var_type == 'fixedlarge':
       logvar = betas.log()
    elif model_var_type == 'fixedsmall':
        logvar = posterior_variance.clip(min=1e-20).log()
    if os.path.exists(config.data.stat_path):
        print('Loading dataset statistics from {}'.format(config.data.
            stat_path))
        train_data = KMFlowTensorDataset(config.data.data_dir,
            stat_path=config.data.stat_path)
    else:
        print('No dataset statistics found. Computing statistics...')
        train_data = KMFlowTensorDataset(config.data.data_dir)
        train_data.save_data_stats(config.data.stat_path)
    x_offset, x_scale = train_data.stat['mean'], train_data.stat['scale']
    try:
        train_loader = paddle.io.DataLoader(dataset=train_data, batch_size=config.training.batch_size, shuffle=True, num_workers=config.data.num_workers)
    except Exception as e:
        print("An error occurred while loading data:", e)
    model = ppsci.arch.DDPM(config=config)
    num_params = sum(p.size for p in model.parameters() if not p.
        stop_gradient)
    print(num_params)
    model = model.to('gpu')
    optimizer = get_optimizer(config, model.parameters())
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(model)
    else:
        ema_helper = None
    start_epoch, step = 0, 0
    if args.resume_training:
        states = paddle.load(path=os.path.join(args.log_path,
            'ckpt.pth'))
        model.set_state_dict(state_dict=states[0])
        states[1]['param_groups'][0]['eps'] = config.optim.eps
        optimizer.set_state_dict(state_dict=states[1])
        start_epoch = states[2]
        step = states[3]
        if config.model.ema:
            ema_helper.set_state_dict(state_dict=states[4])
    writer = SummaryWriter()
    num_iter = 0
    log_freq = 100
    print('Starting training...')
    for epoch in range(start_epoch, config.training.n_epochs):
        data_start = time.time()
        data_time = 0
        epoch_loss = []
        for i, x in enumerate(train_loader):
            n = x.shape[0]
            data_time += time.time() - data_start
            model.train()
            step += 1
            x = x.to(device)
            e = paddle.randn(shape=x.shape, dtype=x.dtype)
            b = betas
            t = paddle.randint(low=0, high=num_timesteps, shape=(n //
                2 + 1,)).to(device)
            t = paddle.concat(x=[t, num_timesteps - t - 1], axis=0)[:n
                ]
            loss = loss_registry[config.model.type](model, x, t, e, b,
                x_offset.item(), x_scale.item())
            epoch_loss.append(loss.item())
            # tb_logger.add_scalar('loss', loss, global_step=step)
            if num_iter % log_freq == 0:
                logging.info(
                    f'step: {step}, loss: {loss.item()}, data time: {data_time / (i + 1)}'
                    )
            writer.add_scalar('loss', loss.item(), step)
            writer.add_scalar('data_time', data_time / (i + 1), step)
            """Class Method: *.zero_grad, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>                optimizer.zero_grad()
            optimizer.clear_grad()
            loss.backward()
            try:
                paddle.nn.utils.clip_grad_norm_(parameters=model.
                    parameters(), max_norm=config.optim.grad_clip)
            except Exception:
                pass
            optimizer.step()
            if config.model.ema:
                ema_helper.update(model)
            if step % config.training.snapshot_freq == 0 or step == 1:
                states = [model.state_dict(), optimizer.state_dict(),
                    epoch, step]
                if config.model.ema:
                    states.append(ema_helper.state_dict())
                paddle.save(obj=states, path=os.path.join(args.
                    log_path, 'ckpt_{}.pdparams'.format(step)))
                paddle.save(obj=states, path=os.path.join(args.
                    log_path, 'ckpt.pdparams'))
            data_start = time.time()
            num_iter = num_iter + 1
        print('==========================================================')
        print('Epoch: {}/{}, Loss: {}'.format(epoch, config.
            training.n_epochs, np.mean(epoch_loss)))
    print('Finished training')
    logging.info(
        f'step: {step}, loss: {loss.item()}, data time: {data_time / (i + 1)}'
        )
    paddle.save(obj=states, path=os.path.join(args.log_path,
        'ckpt_{}.pdparams'.format(step)))
    paddle.save(obj=states, path=os.path.join(args.log_path,
        'ckpt.pdparams'))
    print('Model saved at: ', args.log_path + 'ckpt_{}.pdparams'.format
        (step))
    writer.export_scalars_to_json('./runs/all_scalars.json')
    writer.close()
    

def evaluate(cfg):
    #initilize
    args = cfg.eval
    config = cfg
    os.makedirs(config.log_dir, exist_ok=True)
    if config.model.type == 'conditional':
        dir_name = 'recons_{}_t{}_r{}_w{}'.format(config.data.data_kw, args
            .t, args.reverse_steps, config.sampling.guidance_weight)
    else:
        dir_name = 'recons_{}_t{}_r{}_lam{}'.format(config.data.data_kw,
            args.t, args.reverse_steps, config.sampling.lambda_)
    if config.model.type == 'conditional':
        print('Use residual gradient guidance during sampling')
        dir_name = 'guided_' + dir_name
    elif config.sampling.lambda_ > 0:
        print('Use residual gradient penalty during sampling')
        dir_name = 'pi_' + dir_name
    else:
        print('Not use physical gradient during sampling')
    log_dir = os.path.join(config.log_dir, dir_name)
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('LOG')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, 'logging_info'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    image_sample_dir = log_dir
    device = 'gpu' if paddle.device.cuda.device_count() >= 1 else 'cpu'
    config.place = str(paddle.CUDAPlace(0))
    model_var_type = config.model.var_type
    betas = get_beta_schedule(beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end, num_diffusion_timesteps=
            config.diffusion.num_diffusion_timesteps)
    betas = paddle.to_tensor(data=betas).astype(dtype='float32').to(device)
    num_timesteps = betas.shape[0]
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 -
        alphas_cumprod)
    if model_var_type == 'fixedlarge':
        logvar = np.log(np.append(posterior_variance[1], betas[1:]))
    elif model_var_type == 'fixedsmall':
        logvar = np.log(np.maximum(posterior_variance, 1e-20))

    
    #restuct
    logger.info('Doing sparse reconstruction task')
    logger.info('Loading model')
    if config.model.type == 'conditional':
        print('Using conditional model')
        model = ppsci.arch.DDPM(config=config)
    else:
        print('Using unconditional model')
        raise NotImplementedError('not supported')
    model.set_state_dict(state_dict=paddle.load(path=config.model.ckpt_path))
    print(device)
    # self.device = 'gpu'
    model.to('gpu')
    # model.set_device('gpu')
    logger.info('Model loaded')
    model.eval()
    logger.info('Preparing data')
    ref_data, blur_data, data_mean, data_std = load_recons_data(
        config.data.data_dir, config.data.sample_data_dir, 
        config.data.data_kw, smoothing=config.data.smoothing,
        smoothing_scale=config.data.smoothing_scale)
    scaler = StdScaler(data_mean, data_std)
    logger.info('Start sampling')
    testset = paddle.io.TensorDataset([blur_data, ref_data])
# >>>>>>        test_loader = torch.utils.data.DataLoader(testset, batch_size=self.
#             config.sampling.batch_size, shuffle=False, num_workers=self.
#             config.data.num_workers)
    test_loader = paddle.io.DataLoader(testset, batch_size=config.
        sampling.batch_size, shuffle=False, num_workers=config.data.
        num_workers, use_shared_memory=False)
    l2_loss_all = np.zeros((ref_data.shape[0], args.repeat_run,
        args.sample_step))
    residual_loss_all = np.zeros((ref_data.shape[0], args.
        repeat_run, args.sample_step))
    for batch_index, (blur_data, data) in enumerate(test_loader):
        logger.info('Batch: {} / Total batch {}'.format(batch_index, len(
            test_loader)))
        x0 = blur_data.to(device)
        gt = data.to(device)
        logger.info('Preparing reference image')
        logger.info('Dumping visualization...')
        sample_folder = 'sample_batch{}'.format(batch_index)
        ensure_dir(os.path.join(image_sample_dir, sample_folder))
        sample_img_filename = 'input_image.png'
        path_to_dump = os.path.join(image_sample_dir,
            sample_folder, sample_img_filename)
        x0_masked = x0.clone()
        make_image_grid(slice2sequence(x0_masked), path_to_dump)
        sample_img_filename = 'reference_image.png'
        path_to_dump = os.path.join(image_sample_dir,
            sample_folder, sample_img_filename)
        make_image_grid(slice2sequence(gt), path_to_dump)
        if config.sampling.dump_arr:
            np.save(os.path.join(image_sample_dir, sample_folder,
                'input_arr.npy'), slice2sequence(x0).cpu().numpy())
            np.save(os.path.join(image_sample_dir, sample_folder,
                'reference_arr.npy'), slice2sequence(data).cpu().numpy())
        l2_loss_init = l2_loss(x0, gt)
        logger.info('L2 loss init: {}'.format(l2_loss_init))
        gt_residual = voriticity_residual(gt)[1].detach()
        init_residual = voriticity_residual(x0)[1].detach()
        logger.info('Residual init: {}'.format(init_residual))
        logger.info('Residual reference: {}'.format(gt_residual))
        x0 = scaler(x0)
        xinit = x0.clone()
        if config.sampling.log_loss:
            l2_loss_fn = lambda x: l2_loss(scaler.inverse(x).to('gpu'), gt)
            equation_loss_fn = lambda x: voriticity_residual(scaler.
                inverse(x), calc_grad=False)
            logger1 = MetricLogger({'l2 loss': l2_loss_fn,
                'residual loss': equation_loss_fn})
        for repeat in range(args.repeat_run):
            logger.info(f'Run No.{repeat}:')
            x0 = xinit.clone()
            for it in range(args.sample_step):
                e = paddle.randn(shape=x0.shape, dtype=x0.dtype)
                total_noise_levels = int(args.t * 0.7 ** it)
                a = (1 - betas).cumprod(dim=0)
                x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 -
                    a[total_noise_levels - 1]).sqrt()
                if config.model.type == 'conditional':
                    physical_gradient_func = lambda x: voriticity_residual(
                        scaler.inverse(x))[0] / scaler.scale()
                elif config.sampling.lambda_ > 0:
                    physical_gradient_func = lambda x: voriticity_residual(
                        scaler.inverse(x))[0] / scaler.scale(
                        ) * config.sampling.lambda_
                num_of_reverse_steps = int(args.reverse_steps * 
                    0.7 ** it)
                betas = betas.to(device)
                skip = total_noise_levels // num_of_reverse_steps
                seq = range(0, total_noise_levels, skip)
                if config.model.type == 'conditional':
                    xs, _ = guided_ddim_steps(x, seq, model, betas, w=config.sampling.guidance_weight, dx_func=physical_gradient_func, cache=False, logger=logger1)
                elif config.sampling.lambda_ > 0:
                    xs, _ = ddim_steps(x, seq, model, betas, dx_func=
                        physical_gradient_func, cache=False, logger=logger1)
                else:
                    xs, _ = ddim_steps(x, seq, model, betas, cache=
                        False, logger=logger1)
                x = xs[-1]
                x0 = xs[-1]
                l2_loss_f = l2_loss(scaler.inverse(x.clone()).to('gpu'), gt)
                logger.info('L2 loss it{}: {}'.format(it, l2_loss_f))
                residual_loss_f = voriticity_residual(scaler.inverse(x.
                    clone()), calc_grad=False).detach()
                logger.info('Residual it{}: {}'.format(it, residual_loss_f))
                l2_loss_all[batch_index * x.shape[0]:(batch_index + 1) *
                    x.shape[0], (repeat), (it)] = l2_loss_f.item()
                residual_loss_all[batch_index * x.shape[0]:(batch_index +
                    1) * x.shape[0], (repeat), (it)
                    ] = residual_loss_f.item()
                if config.sampling.dump_arr:
                    np.save(os.path.join(image_sample_dir,
                        sample_folder,
                        f'sample_arr_run_{repeat}_it{it}.npy'),
                        slice2sequence(scaler.inverse(x)).cpu().numpy())
                if config.sampling.log_loss:
                    logger1.log(os.path.join(image_sample_dir,
                        sample_folder), f'run_{repeat}_it{it}')
                    logger1.reset()
        logger.info('Finished batch {}'.format(batch_index))
        logger.info('========================================================'
            )
    logger.info('Finished sampling')
    logger.info(f'mean l2 loss: {l2_loss_all[..., -1].mean()}')
    logger.info(f'std l2 loss: {l2_loss_all[..., -1].std(axis=1).mean()}')
    logger.info(f'mean residual loss: {residual_loss_all[..., -1].mean()}')
    logger.info(
        f'std residual loss: {residual_loss_all[..., -1].std(axis=1).mean()}'
        )


@hydra.main(
    version_base=None, config_path="conf/", config_name="kmflow_re1000_rs256_sparse_recons_conditional.yaml"
)
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()









