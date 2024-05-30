import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import paddle
from denoise_step import ddim_steps
from denoise_step import guided_ddim_steps
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image

sys.path.append("home/aistudio/work/PaddleScience/ppsci/arch")
sys.path.append("/home/aistudio/external-libraries")


def normalize_array(x):
    x_min = np.amin(x)
    x_max = np.amax(x)
    y = (x - x_min) / (x_max - x_min)
    return y, x_min, x_max


def unnormalize_array(y, x_min, x_max):
    return y * (x_max - x_min) + x_min


def data_blurring(fno_data_sample):
    ds_size = 16
    resample_method = Image.NEAREST
    x_array, x_min, x_max = normalize_array(fno_data_sample.numpy())
    im = Image.fromarray((x_array * 255).astype(np.uint8))
    im_ds = im.resize((ds_size, ds_size))
    im_us = im_ds.resize((im.width, im.height), resample=resample_method)
    x_array_blur = np.asarray(im_us)
    x_array_blur = x_array_blur.astype(np.float32) / 255.0
    x_array_blur = unnormalize_array(x_array_blur, x_min, x_max)
    return paddle.to_tensor(data=x_array_blur)


class MetricLogger(object):
    def __init__(self, metric_fn_dict):
        self.metric_fn_dict = metric_fn_dict
        self.metric_dict = {}
        self.reset()

    def reset(self):
        for key in self.metric_fn_dict.keys():
            self.metric_dict[key] = []

    @paddle.no_grad()
    def update(self, **kwargs):
        for key in self.metric_fn_dict.keys():
            self.metric_dict[key].append(self.metric_fn_dict[key](**kwargs))

    def get(self):
        return self.metric_dict.copy()

    def log(self, outdir, postfix=""):
        # 准备一个新字典，用于存储可以被pickle序列化的对象
        serializable_metric_dict = {}

        def to_serializable(value):
            """将值转换为可序列化的格式"""
            if isinstance(value, paddle.Tensor):
                # 将Tensor转换为NumPy数组
                return value.numpy()
            elif isinstance(value, dict):
                # 递归处理字典中的值
                return {k: to_serializable(v) for k, v in value.items()}
            elif isinstance(value, list):
                # 递归处理列表中的值
                return [to_serializable(v) for v in value]
            else:
                # 其他类型的值直接返回
                return value

        # 遍历metric_dict并转换所有值
        for key, value in self.metric_dict.items():
            serializable_metric_dict[key] = to_serializable(value)

        # 使用修改后的字典进行序列化
        with open(os.path.join(outdir, f"metric_log_{postfix}.pkl"), "wb") as f:
            pickle.dump(serializable_metric_dict, f)


def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def load_flow_data(path, stat_path=None):
    data = np.load(path)
    print("Original data shape:", data.shape)
    data_mean, data_scale = np.mean(data[:-4]), np.std(data[:-4])
    print(f"Data range: mean: {data_mean} scale: {data_scale}")
    data = data[-4:, (...)].copy().astype(np.float32)
    data = paddle.to_tensor(data=data, dtype="float32")
    flattened_data = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1] - 2):
            flattened_data.append(data[(i), j : j + 3, (...)])
    flattened_data = paddle.stack(x=flattened_data, axis=0)
    print(f"data shape: {flattened_data.shape}")
    return flattened_data, data_mean.item(), data_scale.item()


def create_gaussian_kernel(kernel_size, sigma):
    # 创建一个二维高斯核
    ax = paddle.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = paddle.meshgrid(ax, ax)
    kernel = paddle.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / paddle.sum(kernel)
    return kernel


def gaussian_blur(image, kernel_size, sigma):
    # 创建高斯核
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)

    # 扩展维度以匹配卷积函数的权重格式 [Co, Ci, H, W]
    gaussian_kernel = gaussian_kernel.reshape([1, 1, kernel_size, kernel_size])
    gaussian_kernel = paddle.tile(
        gaussian_kernel, repeat_times=[image.shape[1], 1, 1, 1]
    )

    # 使用高斯核进行卷积
    # PaddlePaddle的conv2d需要输入的形状是[N, C, H, W]，其中N是批量大小，C是通道数
    # 因此，可能需要调整输入tensor的形状以符合这个要求
    padding = kernel_size // 2
    blurred_image = paddle.nn.functional.conv2d(
        image, weight=gaussian_kernel, stride=1, padding=padding, groups=image.shape[1]
    )
    return blurred_image


def load_recons_data(ref_path, sample_path, data_kw, smoothing, smoothing_scale):
    with np.load(sample_path, allow_pickle=True) as f:
        sampled_data = f[data_kw][-4:, (...)].copy().astype(np.float32)
    sampled_data = paddle.to_tensor(data=sampled_data, dtype="float32")
    ref_data = np.load(ref_path).astype(np.float32)
    data_mean, data_scale = np.mean(ref_data[:-4]), np.std(ref_data[:-4])
    ref_data = ref_data[-4:, (...)].copy().astype(np.float32)
    ref_data = paddle.to_tensor(data=ref_data, dtype="float32")
    flattened_sampled_data = []
    flattened_ref_data = []
    for i in range(ref_data.shape[0]):
        for j in range(ref_data.shape[1] - 2):
            flattened_ref_data.append(ref_data[(i), j : j + 3, (...)])
            flattened_sampled_data.append(sampled_data[(i), j : j + 3, (...)])
    flattened_ref_data = paddle.stack(x=flattened_ref_data, axis=0)
    flattened_sampled_data = paddle.stack(x=flattened_sampled_data, axis=0)
    if smoothing:
        arr = flattened_sampled_data
        ker_size = smoothing_scale
        arr = paddle.nn.functional.pad(
            arr,
            pad=(
                (ker_size - 1) // 2,
                (ker_size - 1) // 2,
                (ker_size - 1) // 2,
                (ker_size - 1) // 2,
            ),
            mode="circular",
        )
        arr = gaussian_blur(arr, ker_size, ker_size)
        flattened_sampled_data = arr[
            (...),
            (ker_size - 1) // 2 : -(ker_size - 1) // 2,
            (ker_size - 1) // 2 : -(ker_size - 1) // 2,
        ]
    print(f"data shape: {flattened_ref_data.shape}")
    return (
        flattened_ref_data,
        flattened_sampled_data,
        data_mean.item(),
        data_scale.item(),
    )


class MinMaxScaler(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, x):
        return x - self.min

    def inverse(self, x):
        return x * (self.max - self.min) + self.min

    def scale(self):
        return self.max - self.min


class StdScaler(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    def scale(self):
        return self.std


def nearest_blur_image(data, scale):
    blur_data = data[:, :, ::scale, ::scale]
    return blur_data


def gaussian_blur_image(data, scale):
    blur_data = paddle.visualization.transforms.GaussianBlur(
        kernel_size=scale, sigma=2 * scale + 1
    )(data)
    return blur_data


def random_square_hole_mask(data, hole_size):
    h, w = data.shape[2:]
    mask = paddle.zeros(shape=data.shape, dtype="int64").to(data.place)
    hole_x = np.random.randint(0, w - hole_size)
    hole_y = np.random.randint(0, h - hole_size)
    mask[(...), hole_y : hole_y + hole_size, hole_x : hole_x + hole_size] = 1
    return mask


def make_image_grid(images, out_path, ncols=10):
    t, h, w = images.shape
    images = images.detach().cpu().numpy()
    b = t // ncols
    fig = plt.figure(figsize=(8.0, 8.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(b, ncols))
    for ax, im_no in zip(grid, np.arange(b * ncols)):
        ax.imshow(images[(im_no), :, :], cmap="twilight", vmin=-23, vmax=23)
        ax.axis("off")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def slice2sequence(data):
    # 首先，选取第二个维度的特定切片
    selected_slice = data[:, 1:2, :, :]  # 选取F维度的第二个切片，保持形状为 [T, 1, H, W]

    # 然后，如果你想要合并T和F维度，可以使用reshape
    # 但在这个例子中，由于F维度只有一个元素，实际上你不需要合并，只需要去掉F这一维
    data = paddle.squeeze(selected_slice, axis=1)  # 结果形状为 [T, H, W]
    return data


def l1_loss(x, y):
    return paddle.mean(x=paddle.abs(x=x - y))


def l2_loss(x, y):
    return ((x - y) ** 2).mean(axis=(-1, -2)).sqrt().mean()


def float_to_complex(x):
    x = paddle.cast(x, dtype="float32")
    img_x = paddle.zeros_like(x)
    x = paddle.complex(x, img_x)
    return x


def voriticity_residual(w, re=1000.0, dt=1 / 32, calc_grad=True):
    w = w.clone()
    w.stop_gradient = not True
    nx = w.shape[2]
    w_h = paddle.fft.fft2(x=w[:, 1:-1], axes=[2, 3])
    k_max = nx // 2
    N = nx
    k_x = (
        paddle.concat(
            x=(
                paddle.arange(start=0, end=k_max, step=1),
                paddle.arange(start=-k_max, end=0, step=1),
            ),
            axis=0,
        )
        .reshape([N, 1])
        .tile(repeat_times=[1, N])
        .reshape([1, 1, N, N])
    )
    k_y = (
        paddle.concat(
            x=(
                paddle.arange(start=0, end=k_max, step=1),
                paddle.arange(start=-k_max, end=0, step=1),
            ),
            axis=0,
        )
        .reshape([1, N])
        .tile(repeat_times=[N, 1])
        .reshape([1, 1, N, N])
    )

    lap = k_x**2 + k_y**2

    k_x = float_to_complex(k_x)
    k_y = float_to_complex(k_y)

    lap[..., 0, 0] = 1.0

    lap = float_to_complex(lap)

    psi_h = w_h / lap
    u_h = 1.0j * k_y * psi_h
    v_h = -1.0j * k_x * psi_h
    wx_h = 1.0j * k_x * w_h
    wy_h = 1.0j * k_y * w_h
    wlap_h = -lap * w_h
    u = paddle.fft.irfft2(x=u_h[(...), :, : k_max + 1], axes=[2, 3])
    v = paddle.fft.irfft2(x=v_h[(...), :, : k_max + 1], axes=[2, 3])
    wx = paddle.fft.irfft2(x=wx_h[(...), :, : k_max + 1], axes=[2, 3])
    wy = paddle.fft.irfft2(x=wy_h[(...), :, : k_max + 1], axes=[2, 3])
    wlap = paddle.fft.irfft2(x=wlap_h[(...), :, : k_max + 1], axes=[2, 3])
    advection = u * wx + v * wy
    wt = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * dt)
    x = paddle.linspace(start=0, stop=2 * np.pi, num=nx + 1)
    x = x[0:-1]
    X, Y = paddle.meshgrid(x, x)
    f = -4 * paddle.cos(x=4 * Y)
    residual = wt + (advection - 1.0 / re * wlap + 0.1 * w[:, 1:-1]) - f
    residual_loss = (residual**2).mean()
    if calc_grad:
        dw = paddle.grad(outputs=residual_loss, inputs=w)[0]
        return dw, residual_loss
    else:
        return residual_loss


def l2_loss_fn(x, scaler, gt):
    return l2_loss(scaler.inverse(x).to("gpu"), gt)


def equation_loss_fn(x, scaler):
    return voriticity_residual(scaler.inverse(x), calc_grad=False)


def physical_gradient_func1(x, scaler):
    return voriticity_residual(scaler.inverse(x))[0] / scaler.scale()


def physical_gradient_func2(x, scaler, config):
    return (
        voriticity_residual(scaler.inverse(x))[0]
        / scaler.scale()
        * config.sampling.lambda_
    )


def eval_input(model, config):
    args = config.eval
    os.makedirs(config.log_dir, exist_ok=True)
    if config.model.type == "conditional":
        dir_name = "recons_{}_t{}_r{}_w{}".format(
            config.data.data_kw,
            args.t,
            args.reverse_steps,
            config.sampling.guidance_weight,
        )
    else:
        dir_name = "recons_{}_t{}_r{}_lam{}".format(
            config.data.data_kw, args.t, args.reverse_steps, config.sampling.lambda_
        )
    if config.model.type == "conditional":
        print("Use residual gradient guidance during sampling")
        dir_name = "guided_" + dir_name
    elif config.sampling.lambda_ > 0:
        print("Use residual gradient penalty during sampling")
        dir_name = "pi_" + dir_name
    else:
        print("Not use physical gradient during sampling")
    log_dir = os.path.join(config.log_dir, dir_name)
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("LOG")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler("%s/%s.txt" % (log_dir, "logging_info"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    image_sample_dir = log_dir
    device = "gpu" if paddle.device.cuda.device_count() >= 1 else "cpu"
    config.place = str(paddle.CUDAPlace(0))
    betas = get_beta_schedule(
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = paddle.to_tensor(data=betas).astype(dtype="float32").to(device)
    # alphas = 1.0 - betas

    # restuct
    logger.info("Doing sparse reconstruction task")
    logger.info("Loading model")
    model.to("gpu")
    logger.info("Model loaded")
    model.eval()
    logger.info("Preparing data")
    ref_data, blur_data, data_mean, data_std = load_recons_data(
        config.data.data_dir,
        config.data.sample_data_dir,
        config.data.data_kw,
        smoothing=config.data.smoothing,
        smoothing_scale=config.data.smoothing_scale,
    )
    scaler = StdScaler(data_mean, data_std)
    logger.info("Start sampling")
    testset = paddle.io.TensorDataset([blur_data, ref_data])
    test_loader = paddle.io.DataLoader(
        testset,
        batch_size=config.sampling.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        use_shared_memory=False,
    )
    l2_loss_all = np.zeros((ref_data.shape[0], args.repeat_run, args.sample_step))
    residual_loss_all = np.zeros((ref_data.shape[0], args.repeat_run, args.sample_step))
    for batch_index, (blur_data, data) in enumerate(test_loader):
        logger.info("Batch: {} / Total batch {}".format(batch_index, len(test_loader)))
        x0 = blur_data.to(device)
        gt = data.to(device)
        logger.info("Preparing reference image")
        logger.info("Dumping visualization...")
        sample_folder = "sample_batch{}".format(batch_index)
        ensure_dir(os.path.join(image_sample_dir, sample_folder))
        sample_img_filename = "input_image.png"
        path_to_dump = os.path.join(
            image_sample_dir, sample_folder, sample_img_filename
        )
        x0_masked = x0.clone()
        make_image_grid(slice2sequence(x0_masked), path_to_dump)
        sample_img_filename = "reference_image.png"
        path_to_dump = os.path.join(
            image_sample_dir, sample_folder, sample_img_filename
        )
        make_image_grid(slice2sequence(gt), path_to_dump)
        if config.sampling.dump_arr:
            np.save(
                os.path.join(image_sample_dir, sample_folder, "input_arr.npy"),
                slice2sequence(x0).cpu().numpy(),
            )
            np.save(
                os.path.join(image_sample_dir, sample_folder, "reference_arr.npy"),
                slice2sequence(data).cpu().numpy(),
            )
        l2_loss_init = l2_loss(x0, gt)
        logger.info("L2 loss init: {}".format(l2_loss_init))
        gt_residual = voriticity_residual(gt)[1].detach()
        init_residual = voriticity_residual(x0)[1].detach()
        logger.info("Residual init: {}".format(init_residual))
        logger.info("Residual reference: {}".format(gt_residual))
        x0 = scaler(x0)
        xinit = x0.clone()
        if config.sampling.log_loss:
            logger1 = MetricLogger(
                {"l2 loss": l2_loss_fn, "residual loss": equation_loss_fn}
            )
        for repeat in range(args.repeat_run):
            logger.info(f"Run No.{repeat}:")
            x0 = xinit.clone()
            for it in range(args.sample_step):
                e = paddle.randn(shape=x0.shape, dtype=x0.dtype)
                total_noise_levels = int(args.t * 0.7**it)
                a = (1 - betas).cumprod(dim=0)
                x = (
                    x0 * a[total_noise_levels - 1].sqrt()
                    + e * (1.0 - a[total_noise_levels - 1]).sqrt()
                )
                if config.model.type == "conditional":
                    physical_gradient_func = physical_gradient_func1(x, scaler)
                elif config.sampling.lambda_ > 0:
                    physical_gradient_func = physical_gradient_func2(x, scaler, config)
                num_of_reverse_steps = int(args.reverse_steps * 0.7**it)
                betas = betas.to(device)
                skip = total_noise_levels // num_of_reverse_steps
                seq = range(0, total_noise_levels, skip)
                if config.model.type == "conditional":
                    xs, _ = guided_ddim_steps(
                        x,
                        seq,
                        model,
                        betas,
                        w=config.sampling.guidance_weight,
                        dx_func=physical_gradient_func,
                        cache=False,
                        logger=logger1,
                    )
                elif config.sampling.lambda_ > 0:
                    xs, _ = ddim_steps(
                        x,
                        seq,
                        model,
                        betas,
                        dx_func=physical_gradient_func,
                        cache=False,
                        logger=logger1,
                    )
                else:
                    xs, _ = ddim_steps(
                        x, seq, model, betas, cache=False, logger=logger1
                    )
                x = xs[-1]
                x0 = xs[-1]
                l2_loss_f = l2_loss(scaler.inverse(x.clone()).to("gpu"), gt)
                logger.info("L2 loss it{}: {}".format(it, l2_loss_f))
                residual_loss_f = voriticity_residual(
                    scaler.inverse(x.clone()), calc_grad=False
                ).detach()
                logger.info("Residual it{}: {}".format(it, residual_loss_f))
                l2_loss_all[
                    batch_index * x.shape[0] : (batch_index + 1) * x.shape[0],
                    (repeat),
                    (it),
                ] = l2_loss_f.item()
                residual_loss_all[
                    batch_index * x.shape[0] : (batch_index + 1) * x.shape[0],
                    (repeat),
                    (it),
                ] = residual_loss_f.item()
                if config.sampling.dump_arr:
                    np.save(
                        os.path.join(
                            image_sample_dir,
                            sample_folder,
                            f"sample_arr_run_{repeat}_it{it}.npy",
                        ),
                        slice2sequence(scaler.inverse(x)).cpu().numpy(),
                    )
                if config.sampling.log_loss:
                    logger1.log(
                        os.path.join(image_sample_dir, sample_folder),
                        f"run_{repeat}_it{it}",
                    )
                    logger1.reset()
        logger.info("Finished batch {}".format(batch_index))
        logger.info("========================================================")
    logger.info("Finished sampling")
    logger.info(f"mean l2 loss: {l2_loss_all[..., -1].mean()}")
    logger.info(f"std l2 loss: {l2_loss_all[..., -1].std(axis=1).mean()}")
    logger.info(f"mean residual loss: {residual_loss_all[..., -1].mean()}")
    logger.info(f"std residual loss: {residual_loss_all[..., -1].std(axis=1).mean()}")
