import paddle
import torch


def save_checkpoint(checkpoint_path, model):
    """Save model and optimizer to checkpoint"""

    paddle.save(
        {"model_state": model},
        checkpoint_path,
    )


def torch2paddle():
    torch_path = "/home/aistudio/data_efficient_nopt/data/possion_64_inference/finetune_b01_m0_n8192.tar"
    paddle_path = "./data/pd_finetune_b01_m0_n8192.tar"

    torch_state_dict = torch.load(torch_path)["model_state"]
    # model.set_state_dict(checkpoint["model_state"])
    fc_names = ["classifier", "fc"]
    paddle_state_dict = {}
    import pdb

    pdb.set_trace()
    for k in torch_state_dict:
        if "num_batches_tracked" in k:  # 飞桨中无此参数，无需保存
            continue
        v = torch_state_dict[k].detach().cpu().numpy()
        flag = [i in k for i in fc_names]
        if any(flag) and "weight" in k:
            new_shape = [1, 0] + list(range(2, v.ndim))
            print(
                f"name: {k}, ori shape: {v.shape}, new shape: {v.transpose(new_shape).shape}"
            )
            v = v.transpose(new_shape)  # 转置 Linear 层的 weight 参数
        # 将 torch.nn.BatchNorm2d 的参数名称改成 paddle.nn.BatchNorm2D 对应的参数名称
        k = k.replace("running_var", "_variance")
        k = k.replace("running_mean", "_mean")
        k = k.replace("module.", "")
        # 添加到飞桨权重字典中
        # k = k
        print(f"k: {k}")
        paddle_state_dict[k] = v
    print(f"paddle_state_dict: {paddle_state_dict.keys()}")
    save_checkpoint(paddle_path, paddle_state_dict)


if __name__ == "__main__":
    torch2paddle()
