#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J n_opt_helm
#SBATCH --mail-user=
#SBATCH --mail-type=all
#SBATCH -t 6:00:00
#SBATCH -A

# this run script assumes slurm job scheduler, but similar run cmd can be used elsewhere
# for DDP (slurm vars setting)
export MASTER_ADDR=$(hostname)

# number of gpus
ngpu=1

# yaml file
yaml_config=./config/operators_poisson.yaml
run_name="r0"
config="pois-64-pretrain-e1_20_m3"

# yaml_config=./config/operators_helmholtz.yaml
# run_name="r0"
# config="helm-64-pretrain-o1_20_m1"

# run command
# cmd="python train.py --yaml_config=$config_file --config=$config --run_num=$run_num --root_dir=$scratch"
cmd="python3.9 pretrain_basic.py --run_name $run_name --config $config --yaml_config $yaml_config"
# cmd="python finetune.py --yaml_config=$config_file  --config=$config --run_num=$run_num --root_dir=$scratch --weights=$scratch/expts/helm-64-pretrain-o1_20_m$m_id/r0/checkpoints/backbone.tar"

$cmd
# srun -l -n $ngpu --cpus-per-task=10 --gpus-per-node $ngpu bash -c "source export_DDP_vars.sh && $cmd"
# done
