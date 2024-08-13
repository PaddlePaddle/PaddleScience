
## dataset download

download dataset from urls:


    https://paddle-org.bj.bcebos.com/hrrr_h5_crop/2015.tar
    https://paddle-org.bj.bcebos.com/hrrr_h5_crop/2016.tar
    https://paddle-org.bj.bcebos.com/hrrr_h5_crop/2017.tar
    https://paddle-org.bj.bcebos.com/hrrr_h5_crop/2018.tar
    https://paddle-org.bj.bcebos.com/hrrr_h5_crop/2019.tar
    https://paddle-org.bj.bcebos.com/hrrr_h5_crop/2020.tar
    https://paddle-org.bj.bcebos.com/hrrr_h5_crop/2020.tar
    https://paddle-org.bj.bcebos.com/hrrr_h5_crop/2021.tar
    https://paddle-org.bj.bcebos.com/hrrr_h5_crop/2022.tar
    https://paddle-org.bj.bcebos.com/hrrr_h5_crop/mean_crop.npy
    https://paddle-org.bj.bcebos.com/hrrr_h5_crop/std_crop.npy
    https://paddle-org.bj.bcebos.com/hrrr_h5_crop/time_mean_crop.npy

dataset:
  ```
    hrrr_h5_crop/
      └─ train/  
          └─ 2015/  
          └─ 2016/  
          └─ 2017/
          └─ 2018/
          └─ 2019/
          └─ 2020/
          └─ 2021/
      └─ valid/  
          └─ 2022/
      └─ stat/  
          └─ mean_crop.npy
          └─ std_crop.npy
          └─ time_mean_crop.npy
  ```

## run

sh train.sh
