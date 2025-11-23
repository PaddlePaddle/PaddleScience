import functools
import logging

import paddle

logger = logging.getLogger("base")


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init_Normal = paddle.nn.initializer.Normal(mean=0.0, std=std)
        init_Normal(m.weight.data)
        if m.bias is not None:
            m.bias.set_value(paddle.zeros_like(m.bias))
    elif classname.find("Linear") != -1:
        init_Normal = paddle.nn.initializer.Normal(mean=0.0, std=std)
        init_Normal(m.weight.data)
        if m.bias is not None:
            m.bias.set_value(paddle.zeros_like(m.bias))
    elif classname.find("BatchNorm2d") != -1:
        init_Normal = paddle.nn.initializer.Normal(mean=1.0, std=std)
        init_Normal(m.weight.data)
        init_Constant = paddle.nn.initializer.Constant(value=0.0)
        init_Constant(m.bias.data)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        init_KaimingNormal = paddle.nn.initializer.KaimingNormal(
            negative_slope=0, nonlinearity="leaky_relu"
        )
        init_KaimingNormal(m.weight.data)
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.set_value(paddle.zeros_like(m.bias))
    elif classname.find("Linear") != -1:
        init_KaimingNormal = paddle.nn.initializer.KaimingNormal(
            negative_slope=0, nonlinearity="leaky_relu"
        )
        init_KaimingNormal(m.weight.data)
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.set_value(paddle.zeros_like(m.bias))
    elif classname.find("BatchNorm2d") != -1:
        init_Constant = paddle.nn.initializer.Constant(value=1.0)
        init_Constant(m.weight.data)
        init_Constant = paddle.nn.initializer.Constant(value=0.0)
        init_Constant(m.bias.data)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init_Orthogonal = paddle.nn.initializer.Orthogonal(gain=1)
        init_Orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.set_value(paddle.zeros_like(m.bias))
    elif classname.find("Linear") != -1:
        init_Orthogonal = paddle.nn.initializer.Orthogonal(gain=1)
        init_Orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.set_value(paddle.zeros_like(m.bias))
    elif classname.find("BatchNorm2d") != -1:
        init_Constant = paddle.nn.initializer.Constant(value=1.0)
        init_Constant(m.weight.data)
        init_Constant = paddle.nn.initializer.Constant(value=0.0)
        init_Constant(m.bias.data)


def init_weights(net, init_type="kaiming", scale=1, std=0.02):
    logger.info("Initialization method [{:s}]".format(init_type))
    if init_type == "normal":
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == "kaiming":
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == "orthogonal":
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            "initialization method [{:s}] not implemented".format(init_type)
        )


def define_G(opt):
    model_opt = opt["model"]
    if model_opt["which_model_G"] == "ddpm":
        from .ddpm_modules import diffusion
        from .ddpm_modules import unet
    elif model_opt["which_model_G"] == "sr3":
        from .sr3_modules import diffusion
        from .sr3_modules import unet
    if (
        "norm_groups" not in model_opt["unet"]
        or model_opt["unet"]["norm_groups"] is None
    ):
        model_opt["unet"]["norm_groups"] = 32
    model = unet.UNet(
        in_channel=model_opt["unet"]["in_channel"],
        out_channel=model_opt["unet"]["out_channel"],
        norm_groups=model_opt["unet"]["norm_groups"],
        inner_channel=model_opt["unet"]["inner_channel"],
        channel_mults=model_opt["unet"]["channel_multiplier"],
        res_blocks=model_opt["unet"]["res_blocks"],
        dropout=model_opt["unet"]["dropout"],
    )
    netG = diffusion.GaussianDiffusion(
        model,
        image_H=model_opt["diffusion"]["image_H"],
        image_W=model_opt["diffusion"]["image_W"],
        channels=model_opt["diffusion"]["channels"],
        loss_type="l1",
        conditional=model_opt["diffusion"]["conditional"],
        schedule_opt=model_opt["beta_schedule"]["train"],
    )
    if opt["mode"] == "train":
        init_weights(netG, init_type="orthogonal")
    if opt["gpu_ids"] and opt["distributed"]:
        assert paddle.device.cuda.device_count() >= 1
        netG = paddle.DataParallel(layers=netG)
    return netG
