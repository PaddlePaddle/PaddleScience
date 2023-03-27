# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def loss_reader(loss, loss_name_list, f):
    """read losses from a log file and returns

    Args:
        loss (numpy array): return losses array
        loss_name_list (string∂): name list of losses
        f: files
    """
    counter = 0
    for line in lines:
        pattern = re.compile(r'\d+.\d+|\d+')  # 查找数字
        num_list = pattern.findall(line)
        # print(num_list)
        for i in range(len(loss_name_list)):
            loss[i][counter] = num_list[i + 1]
        line = f.readline()
        counter = counter + 1


def plot_trainloss(loss, loss_ref, name_list):
    """plot training loss only

    Args:
        loss (_type_): _description_
        loss_ref (_type_): _description_
        name_list (_type_): _description_
    """
    # fluctuition_cease_node = 75000
    n = loss.shape[1]
    n_ref = loss_ref.shape[1]
    epoch = np.arange(n)
    epoch_ref = np.arange(n_ref)
    fig, ax = plt.subplots(2, 3, figsize=(24, 16))
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            # col.axvline(fluctuition_cease_node, linestyle = '--', c = 'green')
            # col.axhline(loss[3 * i + j][fluctuition_cease_node], linestyle = '--', c = 'green')
            col.plot(epoch, loss[3 * i + j], 'b--', label='my log')
            if n >= n_ref:
                max_len = n_ref
            else:
                max_len = n
            col.plot(
                epoch_ref[0:max_len],
                loss_ref[3 * i + j][0:max_len],
                'r--',
                label='0307_4e5')
            col.set_yscale('log')
            col.set(xlim=(1, n), ylim=(1e-6, np.max(loss[3 * i + j])))
            col.legend()
            if col.get_subplotspec().is_last_row():
                col.set_xlabel('epoch')
            if col.get_subplotspec().is_first_col():
                col.set_ylabel('MSE loss value')
            col.set_title(name_list[3 * i + j] + " Diagram")

    fig.tight_layout()
    plt.savefig(out_path + '/output/total_loss.jpg')
    plt.close(fig)


def plot_loss_ratio(out_path, name_list, loss):
    """plot the ratio of loss by stacked histogram

    Args:
        out_path (_type_): _description_
        name_list (_type_): _description_
        loss (_type_): _description_
    """
    output_num = 10
    step = int(loss.shape[1] / output_num)
    list_loss = [[0] * output_num for i in range(m - 1)]
    for i in range(output_num):
        a = int(step * i)
        b = int(step * (i + 1))
        c = loss[1][a:b]  # total loss min value
        min_c = np.min(c)
        total_loss_min_index = np.where(c == min_c)[0][0] + step * i
        list_loss[0][i] = min_c
        for j in range(m - 2):
            list_loss[j + 1][i] = loss[j + 2][total_loss_min_index]

    color = ['yellow', 'black', 'green', 'blue']
    fig1, ax1 = plt.subplots(figsize=(30, 12))

    bottom = np.zeros((output_num, ))
    for i in range(1, 5):
        plt.bar(range(output_num), height=list_loss[i], bottom = bottom.tolist(), \
            color = color[i - 1], label = name_list[i + 1])
        bottom = bottom + np.array(list_loss[i])

    label_list = [
        str(step * i) + ' ~ ' + str(step * (i + 1)) for i in range(output_num)
    ]
    ax1.set_title("minum total loss: Compostition Analysis")
    ax1.set_xlabel("intervals")
    ax1.set_ylabel("loss")
    ax1.set_xticks(range(output_num))
    ax1.set_xticklabels(label_list)
    ax1.legend()
    fig1.tight_layout()
    plt.savefig(out_path + '/output/loss_ratio.jpg')
    plt.close(fig1)


def plot_train_validation_loss(file_err, out_path, n):
    """plot train and validation loss, where validation means quantitative mean nodes error

    Args:
        file_err (_type_): _description_
        out_path (_type_): _description_
        n (_type_): _description_
    """
    fig2, ax2 = plt.subplots(figsize=(30, 12))
    err_data = pd.read_csv(file_err)
    err_index = err_data['epoch'].tolist()
    err_u = err_data['error u'].tolist()
    err_v = err_data['error v'].tolist()
    err_w = err_data['error w'].tolist()
    err_p = err_data['error p'].tolist()

    loss_saved = np.zeros((len(err_index), ))
    ax2.plot(np.arange(n), loss[1], 'b--', label='train loss')
    ax2.plot(err_index, err_u, 'r--', label='validation loss : u')
    ax2.plot(err_index, err_v, 'y--', label='validation loss : v')
    ax2.plot(err_index, err_w, 'g--', label='validation loss : w')
    ax2.plot(err_index, err_p, 'c--', label='validation loss : p')
    ax2.axhline(err_u.pop(), linestyle='--', c='m')
    ax2.set(xlim=(0, n), ylim=(1e-6, np.max(loss[1])))
    ax2.set_title("Train & Validation: Loss Grow")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.set_yscale('log')
    ax2.legend()
    fig2.tight_layout()
    plt.savefig(out_path + '/output/train_validation_cmp.jpg')
    plt.close(fig2)


if __name__ == "__main__":
    # plot train data
    plt.style.use('_mpl-gallery')
    out_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    name_list = [
        'lr', 'total loss', 'equation loss', 'boundary loss',
        'Initial Condition loss', 'data loss'
    ]
    f = open(out_path + "/output/0307_4e5.txt")  # experiment loss file 
    f_ref = open(out_path + "/output/0314_99steps.txt")  # reference loss file 
    file_err = out_path + "/output/06_58_14_lbm_err_99.csv"  # lbm error file for validation
    m = len(name_list)

    # start read files
    lines = f.readlines()
    loss = np.zeros((m, len(lines)), dtype=np.float64)
    loss_reader(loss, name_list, f)
    f.close()

    lines = f_ref.readlines()
    loss_ref = np.zeros((m, len(lines)), dtype=np.float64)
    loss_reader(loss_ref, name_list, f_ref)
    f_ref.close()

    # ploting
    plot_trainloss(loss, loss_ref, name_list)
    plot_loss_ratio(out_path, name_list, loss)
    if os.path.exists(file_err) == True:
        plot_train_validation_loss(file_err, out_path, loss.shape[1])
    else:
        print(f"Warning : {file_err} is missing")
