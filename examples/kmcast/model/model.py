import logging
import os
from collections import OrderedDict

import model.networks as networks
import paddle

from .base_model import BaseModel

logger = logging.getLogger("base")


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None
        self.set_loss()
        self.set_new_noise_schedule(
            opt["model"]["beta_schedule"]["train"], schedule_phase="train"
        )
        if self.opt["mode"] == "train":
            self.netG.train()
            if opt["model"]["finetune_norm"]:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.stop_gradient = not False
                    if k.find("transformer") >= 0:
                        v.stop_gradient = not True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            "Params [{:s}] initialized to 0 and will optimize.".format(
                                k
                            )
                        )
            else:
                optim_params = list(self.netG.parameters())
            self.optG = paddle.optimizer.Adam(
                parameters=optim_params,
                learning_rate=opt["train"]["optimizer"]["lr"],
                weight_decay=0.0,
            )
            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.clear_gradients(set_to_zero=False)
        l_pix = self.netG(self.data)
        b, c, h, w = tuple(self.data["HR"].shape)
        l_pix = l_pix.sum() / int(b * c * h * w)
        l_pix.backward()
        self.optG.step()
        self.log_dict["l_pix"] = l_pix.item()

    def test(self, continous=False):
        self.netG.eval()
        with paddle.no_grad():
            if isinstance(self.netG, paddle.DataParallel):
                self.SR = self.netG.module.super_resolution(self.data, continous)
            else:
                self.SR = self.netG.super_resolution(self.data, continous)
        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with paddle.no_grad():
            if isinstance(self.netG, paddle.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, paddle.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase="train"):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, paddle.DataParallel):
                self.netG.module.set_new_noise_schedule(schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict["SAM"] = self.SR.detach().astype(dtype="float32").cpu()
        else:
            out_dict["SR"] = self.SR.detach().astype(dtype="float32").cpu()
            out_dict["HR"] = self.data["HR"].detach().astype(dtype="float32").cpu()
            out_dict["LR"] = self.data["LR"].detach().astype(dtype="float32").cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, paddle.DataParallel):
            net_struc_str = "{} - {}".format(
                self.netG.__class__.__name__, self.netG.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.netG.__class__.__name__)
        logger.info(
            "Network G structure: {}, with parameters: {:,d}".format(net_struc_str, n)
        )
        # logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt["path"]["checkpoint"],
            "I{}_E{}_gen.pdparams".format(iter_step, epoch),
        )
        opt_path = os.path.join(
            self.opt["path"]["checkpoint"],
            "I{}_E{}_opt.pdparams".format(iter_step, epoch),
        )
        network = self.netG
        if isinstance(self.netG, paddle.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        paddle.save(obj=state_dict, path=gen_path)
        opt_state = {
            "epoch": epoch,
            "iter": iter_step,
            "scheduler": None,
            "optimizer": None,
        }
        opt_state["optimizer"] = self.optG.state_dict()
        paddle.save(obj=opt_state, path=opt_path)
        logger.info("Saved model in [{:s}] ...".format(gen_path))

    def load_network(self):
        load_path = self.opt["eval"]["pretrained_model_path"]
        if load_path is not None:
            logger.info("Loading pretrained model for G [{:s}] ...".format(load_path))
            gen_path = "{}_gen.pdparams".format(load_path)
            opt_path = "{}_opt.pdparams".format(load_path)
            network = self.netG
            if isinstance(self.netG, paddle.DataParallel):
                network = network.module

            network.set_state_dict(state_dict=paddle.load(path=str(gen_path)))
            if self.opt["mode"] == "train":
                opt = paddle.load(path=str(opt_path))
                self.optG.set_state_dict(state_dict=opt["optimizer"])
                self.begin_step = opt["iter"]
                self.begin_epoch = opt["epoch"]
