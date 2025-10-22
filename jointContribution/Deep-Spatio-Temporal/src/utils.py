import paddle


def cal_loss(y_true, y_pred, name="wind_power"):
    y_true = (y_true + 1) / 2
    y_pred = (y_pred + 1) / 2
    diff = y_true - y_pred
    if name == "wind_speed":
        diff = diff * 40
    mae = []
    rmse = []
    for i in range(y_true.shape[1]):
        x = diff[:, i].detach()
        idx = ~paddle.isnan(x)
        l1_x = paddle.abs(x[idx]).mean().item()
        l2_x = (x[idx] ** 2).mean().item() ** 0.5

        mae.append(l1_x)
        rmse.append(l2_x)

    return mae, rmse


def get_weight_attr():
    return paddle.ParamAttr(initializer=paddle.nn.initializer.Normal(mean=0, std=0.1))


def get_bias_attr():
    return paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if not p.stop_gradient)
