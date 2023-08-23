#!/usr/bin/env python

import argparse
import os
import sys

import paddle
import pandas as pd
from paddle import nn
from src.datamgr import DataMgr
from src.datamgr import NRELDataMgr
from src.model import Seq2Seq
from src.trainer import Trainer
from src.utils import count_parameters

# from src.utils import init_weights


def main():
    parser = argparse.ArgumentParser(
        description="Deep Spatio Temporal Wind Forecasting"
    )
    parser.add_argument("--name", default="wind_power", type=str, help="model name")
    parser.add_argument("--epoch", default=300, type=int, help="max epochs")
    parser.add_argument("--batch_size", default=20000, type=int, help="batch size")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--k", default=5, type=int, help="number of spatio neighbors")
    parser.add_argument(
        "--n_turbines", default=200, type=int, help="number of turbines"
    )

    args = parser.parse_args()

    print("Running with following command line arguments: {}".format(args))

    BATCH_SIZE = args.batch_size
    EPOCHS = args.epoch
    LR = args.lr
    K = args.k
    SAVE_FILE = args.name

    if args.name == "wind_power":
        if not os.path.exists("./data/wind_power.csv"):
            sys.exit(
                "No data found!!!\n"
                "Download wind power data first (follow the instructions in readme)."
            )
        data_mgr = DataMgr(file_path="./data/" + args.name + ".csv", K=K)
    elif args.name == "wind_speed":
        if not os.path.exists("./data/wind_speed.csv"):
            sys.exit(
                "No data found!!!\n"
                "Download wind speed data first (follow the instructions in readme)."
            )
        data_mgr = NRELDataMgr(
            folder_path="./data/",
            file_path="wind_speed.csv",
            meta_path="wind_speed_meta.csv",
        )
    model = Seq2Seq(K=K, n_turbines=args.n_turbines)

    criterion = nn.MSELoss()
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=LR)
    print(f"model parameters:{count_parameters(model)}")
    # model.apply(init_weights) # change to Seq2Seq
    trainer = Trainer(
        model=model,
        data_mgr=data_mgr,
        optimizer=optimizer,
        criterion=criterion,
        SAVE_FILE=SAVE_FILE,
        BATCH_SIZE=BATCH_SIZE,
        name=args.name,
    )
    trainer.train(epochs=EPOCHS)
    loss, mae, rmse = trainer.report_test_error()
    outdf = pd.DataFrame({"MAE": mae, "RMSE": rmse}, index=[*range(1, 13)])
    outdf.to_csv("outputs/" + SAVE_FILE + "_metrics.csv")


if __name__ == "__main__":
    main()
    print("Done!")
