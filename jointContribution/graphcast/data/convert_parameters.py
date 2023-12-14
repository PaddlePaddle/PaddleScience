import json
import os

import args
import graphcast
import numpy as np
import paddle


def convert(jax_parameters_path, paddle_parameters_path, mapping_csv, model):
    model = graphcast.GraphCastNet(config)
    state_dict = model.state_dict()
    state_dict_keys = state_dict.keys()
    print(state_dict_keys)
    for key in state_dict_keys:
        print(key, state_dict[key].shape)

    jax_data = np.load(jax_parameters_path)
    for param_name in jax_data.files:
        if jax_data[param_name].size == 1:
            print(param_name, "\t", jax_data[param_name])
        else:
            print(param_name, "\t", jax_data[param_name].shape)

    with open(mapping_csv, "r") as f:
        mapping = [line.strip().split(",") for line in f]
        for jax_key, paddle_key in mapping:
            state_dict[paddle_key].set_value(jax_data[jax_key])
    paddle.save(state_dict, paddle_parameters_path)


if __name__ == "__main__":
    params_names = os.listdir("params")
    config_jsons = [
        "config/GraphCast_operational.json",
        "config/GraphCast_small.json",
        "config/GraphCast.json",
    ]
    mapping_name = "graphcast-jax2paddle.csv"

    for params_name, config_json in zip(params_names, config_jsons):
        jax_parameters_path = os.path.join("params", params_name)
        paddle_parameters_path = os.path.join(
            "params",
            params_name.replace(".npz", ".pdparams").replace(" ", "-"),
        )

        with open(config_json, "r") as f:
            config = args.TrainingArguments(**json.load(f))
        convert(jax_parameters_path, paddle_parameters_path, mapping_name, config)
