import random

import numpy as np
import paddle

random.seed(42)

dtype = paddle.get_default_dtype()
"""
该文件目的是：
1.接收到一组变量范围numpy矩阵以及样本需求个数，shape = (m,2)，输出样本numpy矩阵
执行ParameterArray函数即可
"""


def Partition(number_of_sample, limit_array):
    """
    为各变量的变量区间按样本数量进行划分，返回划分后的各变量区间矩阵
    :param number_of_sample: 需要输出的 样本数量
    :param limit_array: 所有变量范围组成的矩阵,为(m, 2)矩阵，m为变量个数，2代表上限和下限
    :return: 返回划分后的个变量区间矩阵（三维矩阵），三维矩阵每层对应于1个变量
    """
    coefficient_lower = np.zeros((number_of_sample, 2), dtype)
    coefficient_upper = np.zeros((number_of_sample, 2), dtype)
    for i in range(number_of_sample):
        coefficient_lower[i, 0] = 1 - i / number_of_sample
        coefficient_lower[i, 1] = i / number_of_sample
    for i in range(number_of_sample):
        coefficient_upper[i, 0] = 1 - (i + 1) / number_of_sample
        coefficient_upper[i, 1] = (i + 1) / number_of_sample

    partition_lower = coefficient_lower @ limit_array.T  # 变量区间下限
    partition_upper = coefficient_upper @ limit_array.T  # 变量区间上限

    partition_range = np.dstack(
        (partition_lower.T, partition_upper.T)
    )  # 得到各变量的区间划分，三维矩阵每层对应于1个变量
    return partition_range  # 返回区间划分上下限


def Representative(partition_range):
    """
    计算单个随机代表数的函数
    :param partition_range: 一个shape为 (m,N,2) 的三维矩阵，m为变量个数、n为样本个数、2代表区间上下限的两列
    :return: 返回由各变量分区后区间随机代表数组成的矩阵，每列代表一个变量
    """
    number_of_value = partition_range.shape[0]  # 获得变量个数
    numbers_of_row = partition_range.shape[1]  # 获得区间/分层个数
    coefficient_random = np.zeros(
        (number_of_value, numbers_of_row, 2), dtype
    )  # 创建随机系数矩阵
    representative_random = np.zeros((numbers_of_row, number_of_value), dtype)

    for m in range(number_of_value):
        for i in range(numbers_of_row):
            y = random.random()
            coefficient_random[m, i, 0] = 1 - y
            coefficient_random[m, i, 1] = y

    temp_arr = (
        partition_range * coefficient_random
    )  # 利用*乘实现公式计算（对应位置进行乘积计算），计算结果保存于临时矩阵 temp_arr 中 # [m,n,2]x[m,n,2]
    for j in range(number_of_value):  # 计算每个变量各区间内的随机代表数，行数为样本个数n，列数为变量个数m
        temp_random = (
            temp_arr[j, :, 0] + temp_arr[j, :, 1]
        )  # 算出每个变量各自的样本期望l+样本期望r,形状为[n,]
        representative_random[
            :, j
        ] = temp_random  # 把第j个变量的n个样本的期望之和加到representative_random上
    return representative_random  # 返回代表数向量


def Rearrange(arr_random):
    """
    打乱矩阵各列内的数据
    :param arr_random: 一个N行, m列的矩阵
    :return: 每列打乱后的矩阵
    """
    for i in range(arr_random.shape[1]):
        np.random.shuffle(arr_random[:, i])
    return arr_random


def ParameterArray(limitArray, sampleNumber):
    """
    根据输入的各变量的范围矩阵以及希望得到的样本数量，输出样本参数矩阵
    :param limitArray:变量上下限矩阵，shape为(m,2),m为变量个数
    :param sampleNumber:希望输出的 样本数量
    :return:样本参数矩阵
    """
    arr = Partition(sampleNumber, limitArray)
    parametersMatrix = Rearrange(Representative(arr))
    return parametersMatrix


"""以下为类创建"""


class DoE_LHS:
    # 拉丁超立方试验样本生成
    def __init__(self, name_value, bounds, N):
        self.name = name_value
        self.bounds = bounds
        self.result = None
        self.type = "LHS"
        self.ParameterArray = ParameterArray(bounds, N)
        self.N = N

    def get_sample(self):
        """
        将样本数据写入LHS.csv文件，文件保存至运行文件夹内
        """
        return self.ParameterArray
        # sample_data = pd.DataFrame(self.ParameterArray, columns=self.name)
        # sample_data.to_csv("LHS.csv")


"""以下为使用"""

# arr_limit = np.array([[-100, -100, -100, -1000, -1000, -1000, 0, 32, 8, 100],
#                       [100, 100, 100, 1000, 1000, 100, 10, 2000, 100, 500]]).T
# name_value = ["Fx", "Fy", "Fz", "Mx", "My", "Mz", "Pressure", "R", "nozzle_th", "nozzle_h"]  # 变量名
# q = DoE_LHS(N=100, bounds=arr_limit, name_value=name_value)
# q.write_to_csv() #样本结果写入csv文件
