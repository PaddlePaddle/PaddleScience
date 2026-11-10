import pandas as pd
from .timefeatures import time_features
import numpy as np
from paddle import nn
import paddle
def set_time(time_stamp):

    time_stamp = [pd.to_datetime(date_str, format='%Y/%m/%d/%H') for date_str in time_stamp]
    time_stamp = pd.DataFrame({'date': time_stamp})


    time_feature = time_features(time_stamp, timeenc=1, freq='h').astype(np.float32)
    time_feature = paddle.to_tensor(time_feature)
    
    return time_feature

class TimeFeatureEmbedding(nn.Layer):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x, seq_len):
        x = set_time(x)

        time_feature = self.embed(x)
        time_feature = time_feature.unsqueeze(1)
        time_feature = paddle.expand(time_feature ,[time_feature .shape[0], seq_len, time_feature .shape[2]])
        return time_feature