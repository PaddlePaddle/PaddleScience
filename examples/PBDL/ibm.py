# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import paddle
import pandas as pd

import ppsci
from ppsci.utils import logger

paddle.set_default_dtype(d="float32")
matplotlib.use("Agg")
device = str("cuda:0" if paddle.device.cuda.device_count() >= 1 else "cpu").replace(
    "cuda", "gpu"
)


class MyDataset(paddle.io.Dataset):
    def __init__(self, csv_file):
        data_pd = pd.read_csv(csv_file, encoding="gbk", header=None)
        self.data = np.array(data_pd)
        for i in range(tuple(self.data.shape)[-1]):
            data_min = np.min(self.data[:, (i)])
            data_max = np.max(self.data[:, (i)])
            if data_max != data_min:
                self.data[:, (i)] = (self.data[:, (i)] - data_min) / (
                    data_max - data_min
                ) * 2 - 1
        self.scaled_x1 = self.data[:, :48].copy()
        self.scaled_x2 = self.data[:, (48)].copy()
        self.scaled_y = self.data[:, (49)].copy()
        self.x1 = paddle.to_tensor(data=self.scaled_x1, dtype="float32")
        self.x2 = paddle.to_tensor(data=self.scaled_x2, dtype="float32").unsqueeze(
            axis=1
        )
        self.y = paddle.to_tensor(data=self.scaled_y, dtype="float32").unsqueeze(axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.y[idx]


def main(OUTPUT_DIR):

    csv_file = "forceInfo.csv"
    dataset = MyDataset(csv_file)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = paddle.io.random_split(
        dataset=dataset, lengths=[train_size, test_size]
    )
    train_loader = paddle.io.DataLoader(
        dataset=train_dataset, batch_size=train_size, shuffle=False
    )
    test_loader = paddle.io.DataLoader(
        dataset=test_dataset, batch_size=test_size, shuffle=False
    )

    class X1Net(paddle.nn.Layer):
        def __init__(self):
            super(X1Net, self).__init__()
            self.N = 512
            self.fc1 = paddle.nn.Linear(in_features=48, out_features=self.N)
            self.fc2 = paddle.nn.Linear(in_features=self.N, out_features=self.N)
            self.fc3 = paddle.nn.Linear(in_features=self.N, out_features=self.N)
            self.fc4 = paddle.nn.Linear(in_features=self.N, out_features=1)
            self.gelu = paddle.nn.GELU()

        def forward(self, x):
            x = self.gelu(self.fc1(x))
            x = self.gelu(self.fc2(x))
            x = self.gelu(self.fc3(x))
            x = self.fc4(x)
            return x

    class CombinedNet(paddle.nn.Layer):
        def __init__(self):
            super(CombinedNet, self).__init__()
            self.x1_net = X1Net()
            self.fc_combined = paddle.nn.Linear(
                in_features=1, out_features=1, bias_attr=False
            )

        def forward(self, x1, x2):
            x1_out = self.x1_net(x1)
            diff = x1_out - x2
            output = self.fc_combined(diff)
            return output

    class CombinedNet1(paddle.nn.Layer):
        def __init__(self):
            super(CombinedNet1, self).__init__()
            self.x1_net = X1Net()
            self.fc1 = paddle.nn.Linear(in_features=2, out_features=64, bias_attr=False)
            self.fc2 = paddle.nn.Linear(in_features=64, out_features=1, bias_attr=False)

        def forward(self, x1, x2):
            x1_out = self.x1_net(x1)
            combined = paddle.concat(x=(x1_out, x2), axis=1)
            x = paddle.nn.functional.relu(x=self.fc1(combined))
            output = self.fc2(x)
            return output

    model = CombinedNet().to(device)
    criterion = paddle.nn.MSELoss()
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=2e-06, weight_decay=0.0
    )
    checkpoint_path = f"{OUTPUT_DIR}/checkpoint.pth"

    def load_checkpoint(filepath, model, optimizer):
        if os.path.isfile(filepath):
            print(f"Loading checkpoint '{filepath}'")
            checkpoint = paddle.load(path=str(filepath))
            model.set_state_dict(state_dict=checkpoint["model_state_dict"])
            optimizer.set_state_dict(state_dict=checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"]
            loss = checkpoint["loss"]
            print(f"Checkpoint loaded. Last epoch: {epoch}, Loss: {loss}")
            return epoch, loss
        else:
            print(f"No checkpoint found at '{filepath}'")
            return 0, float("inf")

    def save_checkpoint(filepath, model, optimizer, epoch, loss, best_val_loss):
        if loss < best_val_loss:
            print(f"Saving checkpoint at epoch {epoch} with improved loss: {loss}")
            paddle.save(
                obj={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                path=filepath,
            )
        else:
            print(f"Checkpoint not saved at epoch {epoch}, loss did not improve.")

    resume_training = True
    if resume_training:
        start_epoch, _ = load_checkpoint(checkpoint_path, model, optimizer)
        print("Resuming training from checkpoint...")
    else:
        start_epoch, _ = 0, float("inf")
        print("Starting training from scratch...")
    for i, (x1_batch, x2_batch, y_batch) in enumerate(train_loader):
        x1_batch, x2_batch, y_batch = (
            x1_batch.to(device),
            x2_batch.to(device),
            y_batch.to(device),
        )
        break
    for i, (x1_test, x2_test, y_test) in enumerate(test_loader):
        x1_test, x2_test, y_test = (
            x1_test.to(device),
            x2_test.to(device),
            y_test.to(device),
        )
        break
    train_losses = []

    def train(
        model,
        train_loader,
        criterion,
        optimizer,
        num_epochs=100,
        checkpoint_interval=1000,
    ):
        model.train()
        best_val_loss = float("inf")
        for epoch in range(start_epoch, num_epochs):
            running_loss = 0.0
            outputs = model(x1_batch, x2_batch)
            loss = 1000000.0 * criterion(outputs, y_batch)
            running_loss += loss.item() * x1_batch.shape[0]
            optimizer.clear_gradients(set_to_zero=False)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % checkpoint_interval == 0:
                if loss.item() < best_val_loss:
                    save_checkpoint(
                        checkpoint_path,
                        model,
                        optimizer,
                        epoch + 1,
                        loss.item(),
                        best_val_loss,
                    )
                    best_val_loss = loss.item()
                    print(f"Model checkpoint saved at epoch {epoch + 1}, batch {i + 1}")
                else:
                    print(
                        f"Checkpoint not saved at epoch {epoch + 1}, loss did not improve."
                    )
            epoch_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_loss)
            if epoch % 1000 == 0:
                with open(f"{OUTPUT_DIR}/loss.dat", "a") as file0:
                    print(f"{epoch + 1}, {np.log10(epoch_loss):.15f}", file=file0)
                with paddle.no_grad():
                    prediction = outputs.detach().cpu().numpy()
                    target = y_batch.detach().cpu().numpy()
                    L2_error_training = np.sqrt(
                        np.linalg.norm(prediction - target) / np.linalg.norm(target)
                    )
                    with open(f"{OUTPUT_DIR}/TrainingLoss_L2.dat", "a") as file1:
                        print(f"{epoch + 1}, {L2_error_training:.15f}", file=file1)
                with paddle.no_grad():
                    test_prediction = model(x1_test, x2_test)
                    test_prediction = test_prediction.detach().cpu().numpy()
                    test_target = y_test.detach().cpu().numpy()
                    L2_error_testing = np.sqrt(
                        np.linalg.norm(test_prediction - test_target)
                        / np.linalg.norm(test_target)
                    )
                    with open(f"{OUTPUT_DIR}/TestingLoss_L2.dat", "a") as file2:
                        print(f"{epoch + 1}, {L2_error_testing:.15f}", file=file2)
                print(epoch, epoch_loss, L2_error_training, L2_error_testing)
            if epoch % 10000 == 0:
                plt.figure(figsize=(10, 5))
                plt.plot(
                    range(1, len(train_losses) + 1),
                    train_losses,
                    marker="o",
                    linestyle="-",
                    label="Training Loss",
                )
                plt.xlabel("Epoch")
                plt.ylabel("Loss (log scale)")
                plt.yscale("log")
                plt.title("Training and Validation Loss Over Epochs")
                plt.legend()
                plt.grid(True)
                plt.savefig(
                    f"{OUTPUT_DIR}/training_and_validation_loss_plot_epoch_{epoch+1}.png"
                )
                plt.close()

                plt.figure(figsize=(10, 5))
                plt.plot(target, prediction, marker=".", linestyle=None)
                plt.plot(test_target, test_prediction, marker=".", linestyle=None)
                plt.savefig(f"{OUTPUT_DIR}/predicted_Y_epoch_{epoch+1}.png")
                plt.close()

    train(
        model,
        train_loader,
        criterion,
        optimizer,
        num_epochs=5000000,
        checkpoint_interval=1000,
    )
    model_path = f"{OUTPUT_DIR}/trained_model.pth"
    paddle.save(obj=model.state_dict(), path=model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)
    # set output directory
    OUTPUT_DIR = "./output"
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")
    # run model
    main(OUTPUT_DIR)
        pass
