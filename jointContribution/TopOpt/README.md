# TopOpt

Neural networks for topology optimization

Paper link: [[ArXiv](https://arxiv.org/abs/1709.09578)]


## Highlights

- Proposed a deep learning based approach for speeding up the topology optimization methods solving layout problems by stating the problem as an image segmentation task
- Introduce convolutional encoder-decoder architecture (UNet) and the overall approach achieved high performance


## 参考

- <https://github.com/ISosnovik/nn4topopt>


## 数据集

整理原始数据集生成hd5格式数据集

``` shell
mkdir -p Dataset/PreparedData/

python prepare_h5datasets.py
```

## 训练模型

``` shell
python Codes/training_case1.py
```

## 指标结果

保存在eval_results.ipynb中
可以与源代码结果对比 <https://github.com/ISosnovik/nn4topopt/blob/master/results.ipynb>
