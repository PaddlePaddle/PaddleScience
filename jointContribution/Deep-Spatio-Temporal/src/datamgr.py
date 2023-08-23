#!/usr/bin/env python

import numpy as np
import paddle
import pandas as pd
from paddle.io import Dataset
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors


class CRSData:
    def __init__(
        self,
        file_path="../data/Wind Spatio-Temporal Dataset2.csv",
        speed_scale=1.0,
        speed_lwrbd=0.0,
        speed_uprbd=25.0,
        power_scale=1.0,
        power_lwrbd=0.0,
        power_uprbd=1.0,
        K=5,
    ):

        self.csv_data = pd.read_csv(file_path, index_col=0, low_memory=False)

        self.spatio_info = self.csv_data.iloc[0:2, range(0, 200)].transpose()

        self.norm_spatio = 10.0 * preprocessing.scale(self.spatio_info, with_std=False)

        self.K = K

        self.speed_scale = speed_scale
        self.speed_lwrbd = speed_lwrbd
        self.speed_uprbd = speed_uprbd

        self.power_scale = power_scale
        self.power_lwrbd = power_lwrbd
        self.power_uprbd = power_uprbd

        temporal_data = self.csv_data.iloc[4:].astype("float").transpose()
        temporal_data.index = self.csv_data.iloc[3]
        self.temporal_data = self.correct_ts_index(temporal_data)
        self.temporal_data.columns = pd.to_datetime(self.temporal_data.columns)

        self.turbine_speed = self.temporal_data.iloc[range(0, 400, 2)]
        self.turbine_power = self.temporal_data.iloc[range(1, 400, 2)]
        self.mast_speed = self.temporal_data.iloc[[400, 402, 404]]
        self.mast_direction = self.temporal_data.iloc[[401, 403, 405]]

        self.scale_data(
            self.speed_scale,
            self.speed_lwrbd,
            self.speed_uprbd,
            self.power_scale,
            self.power_lwrbd,
            self.power_uprbd,
        )

        self._get_neighbors()

    def correct_ts_index(self, original_df):
        corrected_time_index = pd.date_range(
            original_df.columns[0], original_df.columns[-1], freq="1H"
        )
        # expand one hour
        corrected_time_index = corrected_time_index.union(
            [corrected_time_index[-1] + pd.Timedelta(1, unit="H")]
        )
        corrected_data = original_df.copy()
        corrected_data.columns = corrected_time_index
        return corrected_data

    def scale_data(
        self,
        speed_scale=1.0,
        speed_lwrbd=0.0,
        speed_uprbd=25.0,
        power_scale=1.0,
        power_lwrbd=0.0,
        power_uprbd=1.0,
    ):
        """
        Scale Data
        """

        self.norm_speed = (
            2
            * speed_scale
            * (self.turbine_speed - speed_lwrbd)
            / (speed_uprbd - speed_lwrbd)
            - speed_scale
        )
        self.norm_power = (
            2
            * power_scale
            * (self.turbine_power - power_lwrbd)
            / (power_uprbd - power_lwrbd)
            - power_scale
        )

    def _get_neighbors(self):
        nbrs = NearestNeighbors(n_neighbors=self.K, algorithm="auto").fit(
            self.norm_spatio
        )
        self.kNN_neighbors = nbrs.kneighbors(self.norm_spatio)[1]

        speed_mat = self.norm_speed.to_numpy()
        speed_diff = speed_mat[:, range(1, 30 * 24)] - speed_mat[:, range(30 * 24 - 1)]
        norm_speed_diff = preprocessing.normalize(speed_diff)

        speed_diff_sim = np.matmul(norm_speed_diff, norm_speed_diff.transpose())

        self.speed_diff_neighbors = (-speed_diff_sim).argsort()[:, : self.K]


class DataMgr(CRSData):
    def __init__(
        self,
        file_path="../../data/Wind Spatio-Temporal Dataset2.csv",
        train_len=60 * 24,
        val_len=30 * 24,
        time_len=365 * 24,
        ENC_LEN=48,
        DEC_LEN=12,
        K=5,
        similarity="spatio",
    ):

        super().__init__(file_path=file_path, K=K)

        self.ids = paddle.to_tensor(paddle.arange(200), dtype=paddle.float32).reshape(
            (200, 1)
        )
        if K <= 1:
            self.K = 1
            self.neighbors = paddle.to_tensor(paddle.arange(200)).reshape((200, 1))
        elif similarity == "spatio":
            self.K = K
            self.neighbors = self.kNN_neighbors
        elif similarity == "speed diff":
            self.K = K
            self.neighbors = self.speed_diff_neighbors
        else:
            raise NameError("K or similarity not correctly defined")

        self.speed_tensor = paddle.to_tensor(
            self.norm_speed.to_numpy(), dtype=paddle.float32
        ).unsqueeze(-1)
        self.power_tensor = paddle.to_tensor(
            self.norm_power.to_numpy(), dtype=paddle.float32
        ).unsqueeze(-1)

        self.time_features = (
            paddle.to_tensor(self.norm_speed.columns.hour.values, dtype=paddle.float32)
            .tile(repeat_times=(200, 1))
            .unsqueeze(-1)
        )
        self.spatio_tensor = (
            paddle.to_tensor(self.norm_spatio, dtype=paddle.float32)
            .unsqueeze(1)
            .tile(repeat_times=(1, 8760, 1))
        )

        knn_speed_list = []
        for idx in self.neighbors:
            t_select = paddle.index_select(
                self.speed_tensor, paddle.to_tensor(idx), axis=0
            )
            ndim = t_select.ndim
            perm = list(range(ndim))
            perm[0] = 2
            perm[2] = 0
            knn_speed_list.append(t_select.transpose(perm=perm))

        self.data = paddle.concat(knn_speed_list, axis=0)

        self.data = paddle.concat(
            (self.data, self.power_tensor, self.time_features, self.spatio_tensor),
            axis=2,
        )

        self.train_len = train_len
        self.time_len = time_len

        self.enc_len = ENC_LEN
        self.dec_len = DEC_LEN
        self.total_len = ENC_LEN + DEC_LEN

        self.train_data = self.data[:, :train_len, :]
        self.val_data = self.data[:, train_len : (train_len + val_len), :]
        self.test_data = self.data[:, (train_len + val_len) :, :]


class wpDataset(Dataset):
    def __init__(self, data, ENC_LEN=48, DEC_LEN=12, K=5):
        self.data = data
        self.enc_len = ENC_LEN
        self.total_len = ENC_LEN + DEC_LEN

    def __getitem__(self, index):
        tim = index // 200
        turbine = index % 200
        one_point = self.data[turbine, tim : (tim + self.total_len), :]
        x = paddle.index_select(
            one_point[: self.enc_len, :],
            paddle.to_tensor([*range(5), *range(6, 9)]),
            axis=1,
        )
        y = one_point[(self.enc_len - 1) : self.total_len, :]

        y = y[:, 5:9]
        return paddle.to_tensor(turbine, dtype=paddle.int64), x, y

    def __len__(self):
        return self.data.shape[0] * (self.data.shape[1] - self.total_len)


class NRELDataKneighbors:
    def __init__(
        self,
        folder_path="../data/",
        file_path="1h_wyoming_wind_speed_100m.csv",
        meta_path="1h_wyoming_meta.csv",
        K=9,
    ):
        self.K = K

        self.df_wind_speed = pd.read_csv(folder_path + file_path, index_col=0)
        self.df_meta = pd.read_csv(folder_path + meta_path, index_col=0)
        self.wind_speed_tensor = paddle.to_tensor(
            self.df_wind_speed.to_numpy(), dtype=paddle.float32
        )
        self.time_series = pd.to_datetime(
            [x.split("'")[1] for x in self.df_wind_speed.columns]
        )
        #         self._reformat_1d_to_2d()
        self._scale_data()
        self._get_neighbors()

    def _scale_data(self, lwr=0, upr=40, scale=1.0):
        self.wind_speed_tensor = (
            2 * scale * (self.wind_speed_tensor - lwr) / upr - scale
        )

    def _get_neighbors(self):
        nbrs = NearestNeighbors(n_neighbors=9, algorithm="auto").fit(
            self.df_meta.iloc[:, :2]
        )
        self.kNN_neighbors = nbrs.kneighbors(self.df_meta.iloc[:, :2])[1]
        self.norm_spatio = 10.0 * preprocessing.scale(
            self.df_meta.iloc[:, :2], with_std=False
        )


class NRELDataMgr(NRELDataKneighbors):
    def __init__(
        self,
        folder_path="../data/",
        file_path="1h_wyoming_wind_speed_100m.csv",
        meta_path="1h_wyoming_meta.csv",
        train_len=6 * 30 * 24,
        val_len=2 * 30 * 24,
        K=9,
    ):
        super().__init__(
            folder_path=folder_path, file_path=file_path, meta_path=meta_path, K=K
        )
        self.neighbors = self.kNN_neighbors
        self.wind_speed_tensor = self.wind_speed_tensor.unsqueeze(-1)
        self.spatio_tensor = paddle.to_tensor(
            self.norm_spatio, dtype=paddle.float32
        ).unsqueeze(1)
        self.spatio_tensor = self.spatio_tensor.tile(repeat_times=(1, 8784, 1))

        knn_speed_list = []
        for idx in self.neighbors:
            # knn_speed_list.append(self.wind_speed_tensor[idx, :, :].permute(2, 1, 0))
            select1 = paddle.index_select(
                self.wind_speed_tensor, paddle.to_tensor(idx), axis=0
            )
            knn_speed_list.append(select1.transpose((2, 1, 0)))
        self.data = paddle.concat(knn_speed_list, axis=0)

        self.data = paddle.concat((self.data, self.spatio_tensor), axis=2)

        self.hour = paddle.to_tensor(self.time_series.hour.values, dtype=paddle.float32)
        self.hour = self.hour.unsqueeze(0)
        self.hour = self.hour.unsqueeze(-1)
        self.hour = self.hour.tile(repeat_times=(100, 1, 1))

        self.data = paddle.concat((self.data, self.hour), axis=2)

        self.train_data = self.data[:, :train_len, :]
        self.val_data = self.data[:, train_len : (train_len + val_len), :]
        self.test_data = self.data[:, (train_len + val_len) :, :]


class NRELwpDataset(Dataset):
    def __init__(self, data, ENC_LEN=48, DEC_LEN=12, K=9):
        self.data = data
        self.enc_len = ENC_LEN
        self.total_len = ENC_LEN + DEC_LEN
        self.K = K

    def __getitem__(self, index):

        tim = index // 100
        turbine = index % 100
        one_point = self.data[turbine, tim : (tim + self.total_len), :]
        x = one_point[: self.enc_len, :]
        y = one_point[(self.enc_len - 1) : self.total_len, :]

        # y = y[:, [0, self.K, self.K + 1, self.K + 2]]
        y = paddle.index_select(
            y, paddle.to_tensor([0, self.K, self.K + 1, self.K + 2]), axis=1
        )
        return paddle.to_tensor(turbine, dtype=paddle.int64), x, y

    def __len__(self):
        return self.data.shape[0] * (self.data.shape[1] - self.total_len)
