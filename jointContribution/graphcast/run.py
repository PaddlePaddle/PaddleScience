import json
import os
import pickle

import args
import datasets
import graphcast
import graphtype
import numpy as np
import paddle

# isort: off
from graphtype import GraphGridMesh  # noqa: F401
from graphtype import TriangularMesh  # noqa: F401


def convert_parameters():
    def convert(
        jax_parameters_path,
        paddle_parameters_path,
        mapping_csv,
        model,
        output_size=False,
    ):
        model = graphcast.GraphCastNet(config)
        state_dict = model.state_dict()
        jax_data = np.load(jax_parameters_path)

        if output_size:
            for key in state_dict.keys():
                print(key, state_dict[key].shape)

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

    params_path = "data/params"
    mapping_path = "data/graphcast-jax2paddle.csv"

    params_names = [p for p in os.listdir(params_path) if ".npz" in p]
    config_jsons = {
        "resolution 0.25 - pressure levels 37": "config/GraphCast.json",
        "resolution 0.25 - pressure levels 13": "config/GraphCast_operational.json",
        "resolution 1.0 - pressure levels 13": "config/GraphCast_small.json",
    }

    for params_type, config_json in config_jsons.items():
        params_name = [n for n in params_names if params_type in n]
        if len(params_name) > 1:
            raise ValueError("More one parameter files")
        params_name = params_name[0]

        print(f"Start convert '{params_type}' parameters...")
        config_json = config_jsons[params_type]
        jax_parameters_path = os.path.join(params_path, params_name)
        paddle_parameters_path = os.path.join(
            params_path,
            params_name.replace(".npz", ".pdparams").replace(" ", "-"),
        )
        with open(config_json, "r") as f:
            config = args.TrainingArguments(**json.load(f))
        convert(jax_parameters_path, paddle_parameters_path, mapping_path, config)
        print(f"Convert {params_type} parameters finished.")


def make_graph_template():
    config_jsons = {
        "resolution 0.25 - pressure levels 37": "config/GraphCast.json",
        "resolution 0.25 - pressure levels 13": "config/GraphCast_operational.json",
        "resolution 1.0 - pressure levels 13": "config/GraphCast_small.json",
    }

    for model_type, config_json in config_jsons.items():
        print(
            f"Make graph template for {model_type} and "
            "Save into data/template_graph folder"
        )

        with open(config_json, "r") as f:
            config = args.TrainingArguments(**json.load(f))
        graph = GraphGridMesh(config=config)

        graph_template_path = os.path.join(
            "data/template_graph",
            f"{config.type}.pkl",
        )
        with open(graph_template_path, "wb") as f:
            pickle.dump(graph, f)


def test_datasets():
    with open("config/GraphCast_small.json", "r") as f:
        config = args.TrainingArguments(**json.load(f))
    era5dataset = datasets.ERA5Data(config=config, data_type="train")
    print(era5dataset)


def eval():
    with open("config/GraphCast_small.json", "r") as f:
        config = args.TrainingArguments(**json.load(f))
    dataset = datasets.ERA5Data(config=config, data_type="train")
    model = graphcast.GraphCastNet(config)
    model.set_state_dict(paddle.load(config.param_path))
    graph = model(graphtype.convert_np_to_tensor(dataset.input_data[0]))
    pred = dataset.denormalize(graph.grid_node_feat.numpy())
    pred = graph.grid_node_outputs_to_prediction(pred, dataset.targets_template)
    print(pred)


if __name__ == "__main__":
    convert_parameters()  # step.1
    make_graph_template()  # step.2
    test_datasets()  # step.3
    eval()  # step.4
