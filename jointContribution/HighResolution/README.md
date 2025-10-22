## Install deepali

``` sh
# get deepali(with paddle backend)
git clone https://github.com/PFCCLab/deepali.git
# add temporary path
export PYTHONPATH="/path_to_deepali/src/:$PYTHONPATH"
```

## Dataset

Download demo dataset:

``` sh
cd PaddleScience/jointContribution/HighResolution
# linux
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/HighResolution/patient001.zip
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/HighResolution/Hammersmith_myo2.zip
# windows
# curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/HighResolution/patient001.zip
# curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/HighResolution/Hammersmith_myo2.zip

# unzip
unzip patient001.zip -d data
unzip Hammersmith_myo2.zip
```

## Run

python main_ACDC.py
