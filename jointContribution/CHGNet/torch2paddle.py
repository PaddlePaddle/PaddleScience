import numpy as np
import torch
import paddle


def torch2paddle():
    torch_path = r"C:\Users\huanye\Desktop\chgnet-main\chgnet\pretrained\0.3.0\chgnet_0.3.0_e29f68s314m37.pth.tar"
    paddle_path = r"C:\Users\huanye\Desktop\chgnet-main\chgnet\pretrained\0.3.0\chgnet_0.3.0_e29f68s314m37.pdparams"
    torch_state_dict = torch.load(torch_path)
    import pdb;pdb.set_trace()
    fc_names = ["classifier"]
    paddle_state_dict = {}
    for k in torch_state_dict:
        if "num_batches_tracked" in k:
            continue
        v = torch_state_dict[k].detach().cpu().numpy()
        flag = [i in k for i in fc_names]
        if any(flag) and "weight" in k:  # ignore bias
            new_shape = [1, 0] + list(range(2, v.ndim))
            print(
                f"name: {k}, ori shape: {v.shape}, new shape: {v.transpose(new_shape).shape}"
            )
            v = v.transpose(new_shape)
        k = k.replace("running_var", "_variance")
        k = k.replace("running_mean", "_mean")
        # if k not in model_state_dict:
        if False:
            print(k)
        else:
            paddle_state_dict[k] = v
    paddle.save(paddle_state_dict, paddle_path)


if __name__ == "__main__":
    torch2paddle()