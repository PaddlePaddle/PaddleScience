"""earthformer CuboidTransformer on ENSO (examples/earthformer/conf/earthformer_enso_pretrain.yaml)."""
from pathlib import Path

import paddle
from cases import _common as C
from omegaconf import OmegaConf

import ppsci
from ppsci.arch import cuboid_transformer_decoder as D
from ppsci.arch import cuboid_transformer_encoder as E

EXAMPLE = "earthformer"
NOTE = (
    "synthetic sst_data/sst_target; MSE on sst_target only; MODEL kwargs from conf "
    "(checkpoint_level=2 recompute kept); lru_cache'd attention masks are cleared while "
    "tracing, else the infer program's mask Value leaks into the train program (segfault)"
)
CONFIG = (
    Path(__file__).resolve().parents[3]
    / "examples/earthformer/conf/earthformer_enso_pretrain.yaml"
)
BATCH_SIZE = 16
LEARNING_RATE = 1e-4


def build_model():
    return ppsci.arch.CuboidTransformer(**OmegaConf.load(CONFIG).MODEL)


def build_batch():
    return (
        {"sst_data": C.randn([BATCH_SIZE, 12, 24, 48, 1])},
        {"sst_target": C.randn([BATCH_SIZE, 14, 24, 48, 1])},
    )


def _clear_mask_cache():
    # compute_cuboid_*_attention_mask are lru_cache'd and return tensors, which are
    # program-local Values under to_static; dynamic mode keeps the cache as-is
    if not paddle.in_dynamic_mode():
        E.compute_cuboid_self_attention_mask.cache_clear()
        D.compute_cuboid_cross_attention_mask.cache_clear()


def forward(model, inputs):
    _clear_mask_cache()
    return model(inputs)


def compute_loss(model, inputs, labels):
    _clear_mask_cache()
    return C.data_loss(model(inputs), labels)
