import os

import numpy as np
import paddle
import tqdm
from paddle import io
from src import datamgr
from src import utils


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print("INFO: Early stopping")
                self.early_stop = True


class Trainer:
    def __init__(
        self,
        model,
        data_mgr,
        optimizer,
        criterion,
        SAVE_FILE,
        BATCH_SIZE,
        ENC_LEN=48,
        DEC_LEN=12,
        name="wind_power",
    ):
        self.model = model
        self.name = name
        if name == "wind_power":
            train_dataset = datamgr.wpDataset(
                data_mgr.train_data, ENC_LEN=ENC_LEN, DEC_LEN=DEC_LEN
            )
            val_dataset = datamgr.wpDataset(
                data_mgr.val_data, ENC_LEN=ENC_LEN, DEC_LEN=DEC_LEN
            )
            test_dataset = datamgr.wpDataset(
                data_mgr.test_data, ENC_LEN=ENC_LEN, DEC_LEN=DEC_LEN
            )
        else:
            train_dataset = datamgr.NRELwpDataset(
                data_mgr.train_data, ENC_LEN=ENC_LEN, DEC_LEN=DEC_LEN
            )
            val_dataset = datamgr.NRELwpDataset(
                data_mgr.val_data, ENC_LEN=ENC_LEN, DEC_LEN=DEC_LEN
            )
            test_dataset = datamgr.NRELwpDataset(
                data_mgr.test_data, ENC_LEN=ENC_LEN, DEC_LEN=DEC_LEN
            )
        train_dataloader = io.DataLoader(train_dataset, batch_size=BATCH_SIZE)
        val_dataloader = io.DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_dataloader = io.DataLoader(test_dataset, batch_size=BATCH_SIZE)

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.optimizer = optimizer
        self.criterion = criterion

        self.SAVE_FILE = SAVE_FILE

    def train(self, epochs):
        early_stopping = EarlyStopping()
        for epoch in range(epochs):
            print(" ")
            print(f"Epoch {epoch+1} of {epochs}")
            train_loss, train_mae, train_rmse = self.fit()
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train MAE: {np.array(train_mae).reshape(2,6)}")
            print(f"Train RMSE: {np.array(train_rmse).reshape(2,6)}")

            val_loss, val_mae, val_rmse = self.validate()
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val MAE: {np.array(val_mae).reshape(2,6)}")
            print(f"Val RMSE: {np.array(val_rmse).reshape(2,6)}")

            early_stopping(val_loss)
            print(f"Best Val Loss: {early_stopping.best_loss:.4f}")

            if early_stopping.early_stop:
                paddle.save(
                    self.model.state_dict(),
                    os.path.join("outputs", self.SAVE_FILE + ".pdparams"),
                )
                break
        else:
            paddle.save(
                self.model.state_dict(),
                os.path.join("outputs", self.SAVE_FILE + ".pdparams"),
            )

    def fit(self):
        print("Training")
        self.model.train()
        counter = 0
        running_loss = 0.0
        running_mae = [0.0] * 12
        running_rmse = [0.0] * 12
        prog_bar = tqdm.tqdm(
            enumerate(self.train_dataloader),
            total=int(len(self.train_dataset) / self.train_dataloader.batch_size),
        )
        for i, data in prog_bar:
            counter += 1
            self.optimizer.clear_grad()
            y_pred = self.model(data)
            y_true = data[2]
            y_true = y_true[:, 1:, 0]
            y_pred = y_pred.transpose((1, 0))
            mae, rmse = utils.cal_loss(y_true, y_pred, self.name)

            y_true, y_pred = self.rescale_output(y_true, y_pred)

            idx = ~paddle.isnan(y_true)
            loss = self.criterion(y_pred[idx], y_true[idx])
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            running_mae = [x + y for x, y in zip(running_mae, mae)]
            running_rmse = [x + y for x, y in zip(running_rmse, rmse)]

        train_loss = running_loss / counter
        train_mae = [x / counter for x in running_mae]
        train_rmse = [x / counter for x in running_rmse]

        return train_loss, train_mae, train_rmse

    def validate(self):
        print("Validating")
        self.model.eval()
        counter = 0
        running_loss = 0.0
        running_mae = [0.0] * 12
        running_rmse = [0.0] * 12

        prog_bar = tqdm.tqdm(
            enumerate(self.val_dataloader),
            total=int(len(self.val_dataset) / self.val_dataloader.batch_size),
        )
        with paddle.no_grad():
            for i, data in prog_bar:
                counter += 1
                y_pred = self.model(data)
                y_true = data[2]
                y_true = y_true[:, 1:, 0]
                y_pred = y_pred.transpose((1, 0))
                mae, rmse = utils.cal_loss(y_true, y_pred, self.name)

                y_true, y_pred = self.rescale_output(y_true, y_pred)

                idx = ~paddle.isnan(y_true)
                loss = self.criterion(y_pred[idx], y_true[idx])
                running_loss += loss.item()

                running_mae = [x + y for x, y in zip(running_mae, mae)]
                running_rmse = [x + y for x, y in zip(running_rmse, rmse)]

            val_loss = running_loss / counter
            val_mae = [x / counter for x in running_mae]
            val_rmse = [x / counter for x in running_rmse]

        return val_loss, val_mae, val_rmse

    def report_test_error(self):
        print("Calculating Test Error")
        self.model.eval()
        counter = 0
        running_loss = 0.0
        running_mae = [0.0] * 12
        running_rmse = [0.0] * 12

        prog_bar = tqdm.tqdm(
            enumerate(self.test_dataloader),
            total=int(len(self.test_dataset) / self.test_dataloader.batch_size),
        )
        with paddle.no_grad():
            for i, data in prog_bar:
                counter += 1
                y_pred = self.model(data)
                y_true = data[2]
                y_true = y_true[:, 1:, 0]
                y_pred = y_pred.transpose((1, 0))
                mae, rmse = utils.cal_loss(y_true, y_pred, self.name)

                y_true, y_pred = self.rescale_output(y_true, y_pred)
                idx = ~paddle.isnan(y_true)

                loss = self.criterion(y_pred[idx], y_true[idx])
                running_loss += loss.item()

                running_mae = [x + y for x, y in zip(running_mae, mae)]
                running_rmse = [x + y for x, y in zip(running_rmse, rmse)]

            test_loss = running_loss / counter
            test_mae = [x / counter for x in running_mae]
            test_rmse = [x / counter for x in running_rmse]

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test MAE: {np.array(test_mae).reshape(2,6)}")
        print(f"Test RMSE: {np.array(test_rmse).reshape(2,6)}")

        return test_loss, test_mae, test_rmse

    def rescale_output(self, y_true, y_pred):
        for i in range(12):
            y_true[:, i] = y_true[:, i] * np.sqrt(12 - i)
            y_pred[:, i] = y_pred[:, i] * np.sqrt(12 - i)

        return y_true, y_pred
