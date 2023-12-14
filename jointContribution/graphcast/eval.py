import json

import args
import datasets
import graphcast
import graphtype
import paddle

# isort: off
from graphtype import GraphGridMesh  # noqa: F401
from graphtype import TriangularMesh  # noqa: F401


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
    eval()
