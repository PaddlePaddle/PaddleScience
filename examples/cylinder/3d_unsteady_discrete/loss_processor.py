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

import matplotlib.pyplot as plt
import numpy as np
import re
import os


def loss_reader(loss, loss_name_list, f):
    """read losses from file and returns

    Args:
        loss (numpy array): return losses array
        loss_name_list (string∂): name list of losses
        f: files
    """
    counter = 0
    for line in lines: 
        pattern = re.compile(r'\d+.\d+|\d+')   # 查找数字
        num_list = pattern.findall(line)
        # print(num_list)
        for i in range(len(loss_name_list)):
            loss[i][counter] = num_list[i + 1]
        line = f.readline() 
        counter = counter + 1
    

# plot train data
plt.style.use('_mpl-gallery')
out_path = os.getcwd() + "/examples/cylinder/3d_unsteady_discrete"
name_list = ['lr', 'total loss', 'equation loss', 'boundary loss', 'Initial Condition loss', 'data loss']

# start read files
f = open(out_path + "/output/03_19_47_test_loss.txt")               # 返回一个文z件对象 
lines = f.readlines()
n = len(lines)
m = 6
loss = np.zeros((m, n), dtype = np.float64)
loss_reader(loss, name_list, f)
f.close()

f_ref = open(out_path + "/output/0307_4e5.txt")               # 返回一个文z件对象 
lines = f_ref.readlines()
n1 = len(lines)
m1 = 6
loss_ref = np.zeros((m1, n1), dtype = np.float64)
loss_reader(loss_ref, name_list, f_ref)

# plot
epoch = np.arange(n) 
epoch_ref = np.arange(n1) 
fig, ax = plt.subplots(2, 3, figsize=(24, 16))

for i, row in enumerate(ax):
    for j, col in enumerate(row):
        fluctuition_cease_node = 75000
        # col.axvline(fluctuition_cease_node, linestyle = '--', c = 'green')
        # col.axhline(loss[3 * i + j][fluctuition_cease_node], linestyle = '--', c = 'green')
        col.plot(epoch, loss[3 * i + j], 'b--', label = 'my log')
        if len(epoch) >= len(epoch_ref):
            max_len = len(epoch_ref)
        else:
            max_len = n
        col.plot(epoch_ref[0:max_len], loss_ref[3 * i + j][0:max_len], 'r--', label = '0307_4e5')
        col.set_yscale('log')
        col.set(xlim=(0, n), ylim=(0, np.max(loss[3 * i + j])))
        col.legend()
        if col.get_subplotspec().is_last_row():
            col.set_xlabel('epoch')
        if col.get_subplotspec().is_first_col():
            col.set_ylabel('MSE loss value')    
        col.set_title(name_list[3 * i + j] + " Diagram")

fig.tight_layout()
plt.savefig(out_path + '/output/total_loss.jpg')
plt.close(fig)


output_num = 10
step = int(n / output_num)

list_loss = [[0] * output_num for i in range(m - 1)]

for i in range(output_num):
    a = int(step * i)
    b = int(step * (i + 1))
    c = loss[1][a:b] # total loss min value
    min_c = np.min(c)
    total_loss_min_index = np.where(c == min_c)[0][0] + step * i
    list_loss[0][i] = min_c
    for j in range(m - 2):
        list_loss[j + 1][i] = loss[j + 2][total_loss_min_index]

color = ['yellow', 'black', 'green', 'blue']
fig1, ax1 = plt.subplots(figsize=(30, 12))

bottom = np.zeros((output_num,))
for i in range(1, 5):
    plt.bar(range(output_num), height=list_loss[i], bottom = bottom.tolist(), \
        color = color[i - 1], label = name_list[i + 1])
    bottom = bottom + np.array(list_loss[i])

label_list = [str(step * i) + ' ~ ' + str(step * (i + 1)) for i in range(output_num)]
ax1.set_title("minum total loss: Compostition Analysis")
ax1.set_xlabel("intervals")
ax1.set_ylabel("loss")
ax1.set_xticks(range(output_num))
ax1.set_xticklabels(label_list)
ax1.legend()
fig1.tight_layout()
plt.savefig(out_path + '/output/loss_ratio_ref.jpg')
plt.close(fig1)

