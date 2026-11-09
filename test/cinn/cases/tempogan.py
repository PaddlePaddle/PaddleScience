"""tempoGAN 2D smoke super-resolution (examples/tempoGAN/tempoGAN.py, conf/tempogan.yaml)."""
import paddle
import paddle.nn.functional as F
from cases import _common as C

import ppsci

EXAMPLE = "tempoGAN"
NOTE = (
    "ModelList(gen, spatial disc, tempo disc) with the example's input transforms; loss = "
    "sum of the 4 constraints (gen, gen_tempo, disc, disc_tempo), each running the full "
    "ModelList forward like the 3 Solvers do, under one Adam instead of 3 (no step lr); "
    "fp32 instead of AMP O2; low/high res 128/512 (implied by fc_channel), batch 8, tempo "
    "batch 8//3=2 x 3 frames; `stop_gradient = True` on gen output in transforms -> detach()"
)
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
WEIGHT_GEN = [5.0, 0.0, 1.0]
WEIGHT_GEN_LAYER = [-1.0e-5] * 5
WEIGHT_DISC = 1.0
LOW, HIGH = 128, 512

GEN = dict(
    input_keys=("input_gen",),
    output_keys=("output_gen",),
    in_channel=1,
    out_channels_tuple=[[2, 8, 8], [128, 128, 128], [32, 8, 8], [2, 1, 1]],
    kernel_sizes_tuple=[[[5, 5], [5, 5], [1, 1]]] * 4,
    strides_tuple=[[1, 1, 1]] * 4,
    use_bns_tuple=[[True] * 3] * 3 + [[False] * 3],
    acts_tuple=[["relu", None, None]] * 4,
)


def disc_cfg(s, in_channel):
    keys = [f"out0{s}_layer{i}" for i in range(4)] + [f"out_disc{s}_from_target"]
    keys += [f"out1{s}_layer{i}" for i in range(4)] + [f"out_disc{s}_from_gen"]
    return dict(
        input_keys=(f"input{s}_disc_from_target", f"input{s}_disc_from_gen"),
        output_keys=tuple(keys),
        in_channel=in_channel,
        out_channels=[32, 64, 128, 256],
        fc_channel=1048576,
        kernel_sizes=[[4, 4]] * 4,
        strides=[2, 2, 2, 1],
        use_bns=[False, True, True, True],
        acts=["leaky_relu"] * 4 + [None],
    )


# functions.py helpers, copied because importing it pulls in skimage/matplotlib/PIL
def interpolate(data, ratio, mode="nearest"):
    for _ in range(2):
        data = F.interpolate(
            data, [data.shape[-2] * ratio, data.shape[-1] * ratio], mode=mode
        )
    return data


def reshape_input(d):
    out = {}
    for k, v in d.items():
        N, Ch, H, W = v.shape
        out[k] = paddle.reshape(v, [N * Ch, 1, H, W])
    return out


def dereshape_input(d, Ch):
    for k, v in d.items():
        _, N, H, W = v.shape
        if N < Ch:  # the example warns and repeats the first sample
            d[k] = paddle.concat([v[:1]] * Ch, axis=1)
        else:
            d[k] = paddle.reshape(v[: int(N // Ch) * Ch], [-1, Ch, H, W])
    return d


def build_model():
    gen = ppsci.arch.Generator(**GEN)
    disc = ppsci.arch.Discriminator(**disc_cfg("", 2))
    tempo = ppsci.arch.Discriminator(**disc_cfg("_tempo", 3))

    def gen_in(_in):  # GenFuncs.transform_in
        d = reshape_input(_in)
        return {"input_gen": interpolate(d["density_low"], 2, "nearest")}

    def disc_in(_in):  # DiscFuncs.transform_in
        d = reshape_input(_in)
        low = interpolate(d["density_low"], 2, "nearest")
        fake = gen(d)["output_gen"].detach()
        return {
            "input_disc_from_target": paddle.concat([low, d["density_high"]], axis=1),
            "input_disc_from_gen": paddle.concat([low, fake], axis=1),
        }

    def tempo_in(_in):  # DiscFuncs.transform_in_tempo
        target = _in["density_high"]
        fake = gen(reshape_input(_in))["output_gen"].detach()
        return dereshape_input(
            {"input_tempo_disc_from_target": target, "input_tempo_disc_from_gen": fake},
            3,
        )

    gen.register_input_transform(gen_in)
    disc.register_input_transform(disc_in)
    tempo.register_input_transform(tempo_in)
    return ppsci.arch.ModelList((gen, disc, tempo))


def build_batch():
    nt = BATCH_SIZE // 3
    inputs = {
        "density_low": C.uniform([BATCH_SIZE, 1, LOW, LOW]),
        "density_high": C.uniform([BATCH_SIZE, 1, HIGH, HIGH]),
        "density_low_tempo": C.uniform([nt, 3, LOW, LOW]),
        "density_high_tempo": C.uniform([nt, 3, HIGH, HIGH]),
    }
    return inputs, {}


def split(inputs, tempo):
    s = "_tempo" if tempo else ""
    return {
        "density_low": inputs["density_low" + s],
        "density_high": inputs["density_high" + s],
    }


def run(model, x):  # Solver: ModelList forward + output_expr
    out = model(x)
    out["density_high"] = x["density_high"]
    return out


def forward(model, inputs):
    return model(split(inputs, False))


def bce(x, target):
    return F.binary_cross_entropy_with_logits(
        x, paddle.full_like(x, target), reduction="mean"
    )


def loss_gen(out):  # GenFuncs.loss_func_gen
    loss = F.l1_loss(out["output_gen"], out["density_high"]) * WEIGHT_GEN[0]
    loss += F.mse_loss(out["output_gen"], out["density_high"]) * WEIGHT_GEN[1]
    loss += bce(out["out_disc_from_gen"][-1], 1.0)
    keys = list(out.keys())
    layer = 0
    for i in range(1, len(WEIGHT_GEN_LAYER)):
        layer += (
            WEIGHT_GEN_LAYER[i]
            * F.mse_loss(out[keys[i]], out[keys[5 + i]], reduction="sum")
            / 2
        )
    return loss + layer * WEIGHT_GEN_LAYER[0]


def loss_gen_tempo(out):  # GenFuncs.loss_func_gen_tempo
    return bce(out["out_disc_tempo_from_gen"][-1], 1.0) * WEIGHT_GEN[2]


def loss_disc(out, s=""):  # DiscFuncs.loss_func / loss_func_tempo
    return bce(out[f"out_disc{s}_from_target"], 1.0) * WEIGHT_DISC + bce(
        out[f"out_disc{s}_from_gen"], 0.0
    )


def compute_loss(model, inputs, labels):
    x, xt = split(inputs, False), split(inputs, True)
    loss = loss_disc(run(model, x)) + loss_disc(run(model, xt), "_tempo")
    return loss + loss_gen(run(model, x)) + loss_gen_tempo(run(model, xt))
