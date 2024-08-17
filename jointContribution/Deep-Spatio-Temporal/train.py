import argparse
import os
import sys

import paddle
import pandas as pd
from paddle import nn
from src import datamgr
from src import model
from src import trainer
from src import utils


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

    print(f"Running with following command line arguments: {args}")

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
        data_mgr = datamgr.DataMgr(
            file_path=os.path.join("./data", args.name + ".csv"), K=K
        )
    elif args.name == "wind_speed":
        if not os.path.exists("./data/wind_speed.csv"):
            sys.exit(
                "No data found!!!\n"
                "Download wind speed data first (follow the instructions in readme)."
            )
        data_mgr = datamgr.NRELDataMgr(
            folder_path="./data/",
            file_path="wind_speed.csv",
            meta_path="wind_speed_meta.csv",
        )
    model_obj = model.Seq2Seq(K=K, n_turbines=args.n_turbines)

    criterion = nn.MSELoss()
    optimizer = paddle.optimizer.Adam(
        parameters=model_obj.parameters(), learning_rate=LR
    )
    print(f"model parameters:{utils.count_parameters(model_obj)}")
    trainer_obj = trainer.Trainer(
        model=model_obj,
        data_mgr=data_mgr,
        optimizer=optimizer,
        criterion=criterion,
        SAVE_FILE=SAVE_FILE,
        BATCH_SIZE=BATCH_SIZE,
        name=args.name,
    )
    trainer_obj.train(epochs=EPOCHS)
    loss, mae, rmse = trainer_obj.report_test_error()
    outdf = pd.DataFrame({"MAE": mae, "RMSE": rmse}, index=[*range(1, 13)])
    outdf.to_csv(os.path.join("outputs", SAVE_FILE + "_metrics.csv"))


if __name__ == "__main__":
    main()
    print("Done!")
