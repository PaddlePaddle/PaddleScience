from pathlib import Path

import paddle
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

FC_NAMES = [
    "edge_encoder.mlp.0.linear",
    "graph_conv.0.gated_mlp_atom.g.0.linear",
    "graph_conv.0.gated_mlp_atom.g.1.linear",
    "graph_conv.0.gated_mlp_atom.sigma.0.linear",
    "graph_conv.0.gated_mlp_atom.sigma.1.linear",
    "graph_conv.0.edge_layer_atom.linear",
    "graph_conv.0.gated_mlp_edge.g.0.linear",
    "graph_conv.0.gated_mlp_edge.g.1.linear",
    "graph_conv.0.gated_mlp_edge.sigma.0.linear",
    "graph_conv.0.gated_mlp_edge.sigma.1.linear",
    "graph_conv.0.edge_layer_edge.linear",
    "graph_conv.0.three_body.atom_mlp.linear",
    "graph_conv.0.three_body.edge_gate_mlp.g.0.linear",
    "graph_conv.0.three_body.edge_gate_mlp.sigma.0.linear",
    "graph_conv.1.gated_mlp_atom.g.0.linear",
    "graph_conv.1.gated_mlp_atom.g.1.linear",
    "graph_conv.1.gated_mlp_atom.sigma.0.linear",
    "graph_conv.1.gated_mlp_atom.sigma.1.linear",
    "graph_conv.1.edge_layer_atom.linear",
    "graph_conv.1.gated_mlp_edge.g.0.linear",
    "graph_conv.1.gated_mlp_edge.g.1.linear",
    "graph_conv.1.gated_mlp_edge.sigma.0.linear",
    "graph_conv.1.gated_mlp_edge.sigma.1.linear",
    "graph_conv.1.edge_layer_edge.linear",
    "graph_conv.1.three_body.atom_mlp.linear",
    "graph_conv.1.three_body.edge_gate_mlp.g.0.linear",
    "graph_conv.1.three_body.edge_gate_mlp.sigma.0.linear",
    "graph_conv.2.gated_mlp_atom.g.0.linear",
    "graph_conv.2.gated_mlp_atom.g.1.linear",
    "graph_conv.2.gated_mlp_atom.sigma.0.linear",
    "graph_conv.2.gated_mlp_atom.sigma.1.linear",
    "graph_conv.2.edge_layer_atom.linear",
    "graph_conv.2.gated_mlp_edge.g.0.linear",
    "graph_conv.2.gated_mlp_edge.g.1.linear",
    "graph_conv.2.gated_mlp_edge.sigma.0.linear",
    "graph_conv.2.gated_mlp_edge.sigma.1.linear",
    "graph_conv.2.edge_layer_edge.linear",
    "graph_conv.2.three_body.atom_mlp.linear",
    "graph_conv.2.three_body.edge_gate_mlp.g.0.linear",
    "graph_conv.2.three_body.edge_gate_mlp.sigma.0.linear",
    "graph_conv.3.gated_mlp_atom.g.0.linear",
    "graph_conv.3.gated_mlp_atom.g.1.linear",
    "graph_conv.3.gated_mlp_atom.sigma.0.linear",
    "graph_conv.3.gated_mlp_atom.sigma.1.linear",
    "graph_conv.3.edge_layer_atom.linear",
    "graph_conv.3.gated_mlp_edge.g.0.linear",
    "graph_conv.3.gated_mlp_edge.g.1.linear",
    "graph_conv.3.gated_mlp_edge.sigma.0.linear",
    "graph_conv.3.gated_mlp_edge.sigma.1.linear",
    "graph_conv.3.edge_layer_edge.linear",
    "graph_conv.3.three_body.atom_mlp.linear",
    "graph_conv.3.three_body.edge_gate_mlp.g.0.linear",
    "graph_conv.3.three_body.edge_gate_mlp.sigma.0.linear",
    "final.g.0.linear",
    "final.g.1.linear",
    "final.g.2.linear",
    "final.sigma.0.linear",
    "final.sigma.1.linear",
    "final.sigma.2.linear",
    "atom_embedding.mlp.0.linear",
]


def mattersim_torch2paddle(load_path, save_path):
    torch_path = str(load_path)
    paddle_path = str(save_path)
    torch_checkpoint = torch.load(torch_path)
    torch_state_dict = torch_checkpoint["model"]

    # convert weights
    paddle_state_dict = {}
    for k in torch_state_dict:
        if "num_batches_tracked" in k:
            continue

        v = paddle.from_dlpack(torch_state_dict[k].detach())

        flag = [i in k for i in FC_NAMES]

        if any(flag) and "weight" in k:
            new_shape = [1, 0] + list(range(2, v.ndim))
            print(
                f"name: {k}, ori shape: {v.shape}, new shape: {v.transpose(new_shape).shape}"
            )
            v = v.transpose(new_shape)

        k = k.replace("running_var", "_variance")
        k = k.replace("running_mean", "_mean")

        paddle_state_dict[k] = v

    # convert ema
    torch_ema_state_dict = torch_checkpoint["ema"]
    paddle_ema_state_dict = {}
    paddle_ema_state_dict["decay"] = torch_ema_state_dict["decay"]
    paddle_ema_state_dict["num_updates"] = torch_ema_state_dict["num_updates"]

    paddle_ema_state_dict["shadow_params"] = []
    for ptw in torch_ema_state_dict["shadow_params"]:
        paddle_ema_state_dict["shadow_params"].append(paddle.from_dlpack(ptw.detach()))

    paddle_ema_state_dict["collected_params"] = []
    for ptw in torch_ema_state_dict["collected_params"]:
        paddle_ema_state_dict["collected_params"].append(
            paddle.from_dlpack(ptw.detach())
        )

    paddle_checkpoint = {
        "model": paddle_state_dict,
        "model_name": torch_checkpoint["model_name"],
        "model_args": torch_checkpoint["model_args"],
        "last_epoch": torch_checkpoint["last_epoch"],
        "validation_metrics": torch_checkpoint["validation_metrics"],
        "description": torch_checkpoint["description"],
        "ema": paddle_ema_state_dict,
    }

    paddle.save(paddle_checkpoint, paddle_path)


if __name__ == "__main__":
    print("Converting mattersim-v1.0.0-1M.pth to mattersim-v1.0.0-1M.pdparams")
    mattersim_torch2paddle(
        ROOT_DIR / "pretrained_models" / "mattersim-v1.0.0-5M.pth",
        ROOT_DIR / "pretrained_models" / "mattersim-v1.0.0-5M.pdparams",
    )

    print("Converting mattersim-v1.0.0-5M.pth to mattersim-v1.0.0-5M.pdparams")
    mattersim_torch2paddle(
        ROOT_DIR / "pretrained_models" / "mattersim-v1.0.0-5M.pth",
        ROOT_DIR / "pretrained_models" / "mattersim-v1.0.0-5M.pdparams",
    )
