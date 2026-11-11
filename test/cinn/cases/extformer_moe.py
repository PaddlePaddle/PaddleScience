"""extformer_moe ExtFormerMoECuboid on ENSO (examples/extformer_moe/conf/extformer_moe_enso_pretrain.yaml)."""
from pathlib import Path

import paddle
from cases import _common as C
from omegaconf import OmegaConf

import ppsci
from ppsci.arch import extformer_moe_cuboid_decoder as D
from ppsci.arch import extformer_moe_cuboid_encoder as E
from ppsci.arch import extformer_moe_utils as U

EXAMPLE = "extformer_moe"
NOTE = (
    "synthetic sst_data/sst_target (sst_target also fed as model input for the RnC loss); "
    "loss = MSE(sst_target) + aux_loss + rank_loss as in enso_metric.train_extformer_moe_func; "
    "MODEL/MOE/RNC kwargs from conf; the training-time gating noise "
    "(paddle.randn in GatingNet.forward) is replaced by fixed per-gate tensors from build_batch "
    "so top-k routing is identical in both modes; lru_cache'd attention masks are cleared "
    "while tracing, else the infer program's mask Value leaks into the train program; "
    "RnC FeatureSimilarity takes norm() of the zero diagonal feature differences, whose "
    "eager p_norm grad is 0/0, so the dynamic loss is NaN from step 2 while CINN (prim "
    "decomposed grad) stays finite: only the first train loss is comparable"
)
CONFIG = (
    Path(__file__).resolve().parents[3]
    / "examples/extformer_moe/conf/extformer_moe_enso_pretrain.yaml"
)
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
INPUT_SHAPE = (12, 24, 48, 1)
TARGET_SHAPE = (14, 24, 48, 1)

# gate key -> noise shape [B, T, H, W, num_experts], filled by build_model
_NOISE_SHAPES = {}
# state of the gating-noise feed: batch dict and key of the gate being traced/run
_NOISE = {"inputs": None, "key": None}
_orig_randn = paddle.randn
_orig_gate_forward = U.GatingNet.forward


def _fed_randn(shape, *args, **kwargs):
    key = _NOISE["key"]
    if key is None:
        return _orig_randn(shape, *args, **kwargs)
    return _NOISE["inputs"][key]


def _gate_forward(self, x, *args, **kwargs):
    use_fed = self.training and _NOISE["inputs"] is not None
    _NOISE["key"] = self._case_noise_key if use_fed else None
    out = _orig_gate_forward(self, x, *args, **kwargs)
    _NOISE["key"] = None
    return out


paddle.randn = _fed_randn  # falls through to the real randn outside a gate
U.GatingNet.forward = _gate_forward


def build_model():
    cfg = OmegaConf.load(CONFIG)
    model = ppsci.arch.ExtFormerMoECuboid(
        **cfg.MODEL,
        moe_config=OmegaConf.to_object(cfg.MOE),
        rnc_config=OmegaConf.to_object(cfg.RNC),
    )
    _NOISE_SHAPES.clear()
    for name, layer in model.named_sublayers():
        if isinstance(layer, U.GatingNet):
            key = "gate_noise." + name
            layer._case_noise_key = key
            _NOISE_SHAPES[key] = [
                BATCH_SIZE,
                *layer.input_shape[:-1],
                layer.num_experts,
            ]
    return model


def build_batch():
    target = C.randn([BATCH_SIZE, *TARGET_SHAPE])
    inputs = {"sst_data": C.randn([BATCH_SIZE, *INPUT_SHAPE]), "sst_target": target}
    for key, shape in _NOISE_SHAPES.items():
        inputs[key] = C.randn(shape)
    return inputs, {"sst_target": target}


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
    _NOISE["inputs"] = inputs
    out = model(inputs)
    _NOISE["inputs"] = None
    loss = C.mse(out["sst_target"], labels["sst_target"])
    if out["aux_loss"] is not None:
        loss = loss + out["aux_loss"]
    if out["rank_loss"] is not None:
        loss = loss + out["rank_loss"][0]
    return loss
