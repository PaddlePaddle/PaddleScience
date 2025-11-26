from ppsci.utils import logger


def create_model(cfg):
    from .model import DDPM as M

    m = M(cfg)
    logger.info("Model [{:s}] is created.".format(m.__class__.__name__))
    return m
