#!/bin/bash

# yaml file
yaml_config=./config/operators_poisson.yaml
run_name="r0"
config="pois-64-pretrain-e1_20_m3"

# yaml_config=./config/operators_helmholtz.yaml
# run_name="r0"
# config="helm-64-pretrain-o1_20_m1"

# run command
cmd="python3 pretrain_basic.py --run_name $run_name --config $config --yaml_config $yaml_config"

$cmd
