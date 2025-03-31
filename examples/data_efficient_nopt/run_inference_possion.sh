#!/bin/bash

set -ex

CUDA_VISIBLE_DEVICES=0 python3.9 inference_fno_helmholtz_poisson.py \
--config config/inference_poisson.yaml \
--ckpt_path /home/aistudio/data_efficient_nopt/data/possion_64_inference/finetune_b01_m0_n8192.tar \
--num_demos 1
