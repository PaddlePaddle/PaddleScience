import time

import paddle
from dataset.Load_mesh import DatasetFactory
from NN.Transolver.SageTrans_importer_B import FVGN
from utils import get_param
from utils.get_param import get_hyperparam
from utils.Logger import Logger

params, git_info = get_param.params()
if params.load_index is None:
    params.load_index = "90"
device = str("cuda" if paddle.device.cuda.device_count() >= 1 else "cpu").replace(
    "cuda", "gpu"
)
print(f"Using device:{device}")
logger = Logger(
    get_hyperparam(params),
    use_csv=True,
    use_tensorboard=False,
    params=params,
    git_info=git_info,
    copy_code=False,
    saving_path="./Logger",
    loading_path="./Logger",
)
datasets_factory = DatasetFactory(params=params, device=device)
test_indices = list(range(0, 100))
test_dataset, test_loader = datasets_factory.create_testset(
    batch_size=1,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    valid_num=len(test_indices),
    subsampling=False,
    indices=test_indices,
)


model = FVGN(params)
fluid_model = model.to(device)
fluid_model.eval()
params.load_date_time, params.load_index = logger.load_state(
    model=fluid_model,
    optimizer=None,
    scheduler=None,
    datetime=logger.datetime,
    index=params.load_index,
    device=device,
)
params.load_index = params.load_index
print(f"loaded: {params.load_date_time}, {params.load_index}")
params.load_index = 0 if params.load_index is None else params.load_index
start = time.time()
with paddle.no_grad():
    epoc_val_loss = 0
    for batch_index, graph_cell in enumerate(test_loader):
        graph_cell = graph_cell[0]
        graph_cell = test_dataset.datapreprocessing(graph_cell, is_training=False)
        pred_node_valid = fluid_model(
            graph_cell=graph_cell, is_training=False, params=params
        )
        reversed_node_press = (
            pred_node_valid - 1e-08
        ) * graph_cell.press_std + graph_cell.press_mean
        logger.save_test_results(
            value=reversed_node_press.cpu().detach().squeeze().numpy(),
            num_id="".join(
                [chr(ascii_code) for ascii_code in graph_cell.origin_id.cpu().tolist()]
            )[2:],
        )
print(f"Generating answer completed completed in {time.time() - start:.2f} seconds")
