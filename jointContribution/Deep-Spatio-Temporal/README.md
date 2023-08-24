# Deep-Spatio-Temporal

Code for [Deep Spatio-Temporal Wind Power Forecasting](https://arxiv.org/abs/2109.14530)

## How to use

The model is validated on two datasets.

### Wind power forecasting

This dataset is from <https://aml.engr.tamu.edu/book-dswe/dswe-datasets/>. The data used here is Wind Spatio-Temporal Dataset2. Download data, put it into the './data' folder and rename it to 'wind_power.csv'. Then, run following

``` bash
python train.py --name wind_power --epoch 300 --batch_size 20000 --lr 0.001 --k 5 --n_turbines 200
```

### Wind speed forecasting

**运行 getNRELdata.py 需要在 <https://developer.nrel.gov/signup/> 注册，获取API Key，  
然后使用 `pip install h5pyd` 安装 h5pyd，使用 hsconfigure 命令配置（<https://github.com/HDFGroup/hsds_examples）>  
endpoint: <https://developer.nrel.gov/api/hsds>**

The model performance on wind speed forecasting is validated on NREL WIND dataset (<https://www.nrel.gov/wind/data-tools.html>). We select one wind farm with 100 turbines from Wyoming. To get data, first run

``` bash
python getNRELdata.py
```

Then run

``` bash
python train.py --name wind_speed --epoch 300 --batch_size 20000 --lr 0.001 --k 9 --n_turbines 100
```

## References

* Jiangyuan Li, Mohammadreza Armandpour. (2022) "Deep Spatio-Temporal Wind Power Forecasting". IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).
* <https://github.com/jiangyuan-li/Deep-Spatio-Temporal.git>
