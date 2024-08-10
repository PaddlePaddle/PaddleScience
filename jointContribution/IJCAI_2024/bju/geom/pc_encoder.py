import copy
from collections import OrderedDict

import paddle
from geom.models import ULIP_models as models


def load_geom_encoder(args, pretrained=True, frozen=False):
    ckpt = paddle.load(path=args.ulip_ckpt)
    state_dict = OrderedDict()
    for k, v in ckpt["state_dict"].items():
        state_dict[k.replace("module.", "")] = v
    print("=> creating model: {}".format(args.ulip_model))
    model = getattr(models, args.ulip_model)(args=args)
    if pretrained:
        model.set_state_dict(state_dict=state_dict, use_structured_name=True)
        print("=> loaded resume checkpoint '{}'".format(args.ulip_ckpt))
    else:
        print("=> new model without pretraining")
    point_encoder = copy.deepcopy(model.point_encoder)
    pc_projection = copy.deepcopy(model.pc_projection)
    del model
    if frozen:
        for params in point_encoder.parameters():
            params.stop_gradient = not False
        pc_projection.stop_gradient = not False
    return point_encoder, pc_projection
