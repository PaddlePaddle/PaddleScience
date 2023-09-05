import numpy as np
import os
import pickle
import ppsci
from ppsci.utils import logger

def split_tensors(*tensors, ratio):
    assert len(tensors) > 0
    split1, split2 = [], []
    count = len(tensors[0])
    for tensor in tensors:
        assert len(tensor) == count
        split1.append(tensor[:int(len(tensor) * ratio)])
        split2.append(tensor[int(len(tensor) * ratio):])
    if len(tensors) == 1:
        split1, split2 = split1[0], split2[0]
    return split1, split2

# paddle.seed(999)

x = pickle.load(open(os.path.join("/home/my/Share/", "dataX.pkl"), "rb"))
y = pickle.load(open(os.path.join("/home/my/Share/", "dataY.pkl"), "rb"))

train_dataset, test_dataset = split_tensors(x, y, ratio=float(0.7))

test_x, test_y = test_dataset[:]
#x = np.array(x)
#x = x.transpose(0, 2, 3, 1)
#print(x.shape)

channels_weights = np.reshape(
        np.sqrt(np.mean(np.transpose(y, (0, 2, 3, 1)).reshape((981 * 172 * 79, 3)) ** 2, axis=0)),
        (1, -1, 1, 1))

BATCH_SIZE = 64
EPOCHS = 5000

def loss_expr(output_dict, *args):
  output=output_dict['x']
  y=args[0]['y']
  lossu = ((output[:, 0, :, :] - y[:, 0, :, :]) ** 2).reshape(
            (output.shape[0], 1, output.shape[2], output.shape[3]))
  lossv = ((output[:, 1, :, :] - y[:, 1, :, :]) ** 2).reshape(
      (output.shape[0], 1, output.shape[2], output.shape[3]))
  lossp = (((output[:, 2, :, :] - y[:, 2, :, :])).reshape(
      (output.shape[0], 1, output.shape[2], output.shape[3]))).abs()
  loss = (lossu + lossv + lossp) / channels_weights
  return loss.sum()

sup_constraint = ppsci.constraint.SupervisedConstraint(
    {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": {
                "x": test_x,
                "y": test_y
            },
            "label": {"y": test_y}
        },
        "batch_size": BATCH_SIZE,
        "sampler": {
            "name": "DistributedBatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
    },
    ppsci.loss.FunctionalLoss(loss_expr),
    name="sup_constraint",
)

constraint = {sup_constraint.name: sup_constraint}

kernel_size = 5
filters = [8, 16, 32, 32]
bn = False
wn = False

model = ppsci.arch.UNetEx(3, 3, filters=filters, kernel_size=kernel_size, batch_norm=bn, weight_norm=wn)

optimizer = ppsci.optimizer.Adam(0.001,weight_decay=0.005)(model)
solver = ppsci.solver.Solver(model,constraint=constraint,output_dir='.', optimizer=optimizer,epochs=EPOCHS)
solver.train()