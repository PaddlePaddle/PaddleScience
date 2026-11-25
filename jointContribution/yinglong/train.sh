export PYTHONPATH=$PWD
python -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,4' examples/fourcastnet_hrrr/train_pretrain.py
