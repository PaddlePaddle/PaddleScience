#!/usr/bin/env bash
export PDSC_DIR=$(cd "$( dirname ${BASH_SOURCE[0]})"; cd ..; pwd)
export TEST_DIR="${PDSC_DIR}"
source ${TEST_DIR}/test_tipc/common_func.sh

# Read txt in ./test_tipc/configs/
PREPARE_PARAM_FILE=$1
dataline=`cat $PREPARE_PARAM_FILE`
lines=(${dataline})
workdir=$(func_parser_value "${lines[65]}")
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Test training benchmark for a model.
function _set_params(){
    model_item=$(func_parser_value "${lines[1]}")           # (必选) 模型 item |fastscnn|segformer_b0| ocrnet_hrnetw48
    base_batch_size=$(func_parser_value "${lines[57]}")     # (必选) 如果是静态图单进程，则表示每张卡上的BS，需在训练时*卡数
    fp_item=$(func_parser_value "${lines[58]}")             # (必选) fp32|fp16
    epochs=$(func_parser_value "${lines[59]}")              # (必选) Epochs
    run_mode=$(func_parser_value "${lines[3]}")             # (必选) MP模型并行|DP数据并行|PP流水线并行|混合并行DP1-MP1-PP1|DP1-MP4-PP1
    device_num=$(func_parser_value "${lines[4]}")           # (必选) 使用的卡数量，N1C1|N1C8|N4C32 （4机32卡）

    backend="paddle"
    model_repo="PaddleScience"      # (必选) 模型套件的名字
    speed_unit="samples/sec"        # (必选)速度指标单位
    skip_steps=0                    # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="ips:"                  # (必选)解析日志，筛选出性能数据所在行的关键字
    convergence_key="loss:"         # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"

#   以下为通用执行命令，无特殊可不用修改
    model_name=${model_item}_bs${base_batch_size}_${fp_item}_${run_mode}  # (必填) 且格式不要改动,与竞品名称对齐
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}  # （必填） TRAIN_LOG_DIR  benchmark框架设置该参数为全局变量
    speed_log_path=${LOG_PATH_INDEX_DIR:-$(pwd)}

    train_log_file=${run_log_path}/${model_repo}_${model_name}_${device_num}_log
    speed_log_file=${speed_log_path}/${model_repo}_${model_name}_${device_num}_speed
    echo run_log_path: ${run_log_path}
}

function _analysis_log(){
    echo "train_log_file: ${train_log_file}"
    echo "speed_log_file: ${speed_log_file}"
    cmd="python "${BENCHMARK_ROOT}"/scripts/analysis.py --filename ${train_log_file} \
        --speed_log_file ${speed_log_file} \
        --model_name ${model_name} \
        --base_batch_size ${base_batch_size} \
        --run_mode ${run_mode} \
        --fp_item ${fp_item} \
        --keyword ${keyword} \
        --skip_steps ${skip_steps} \
        --device_num ${device_num} "
    echo ${cmd}
    eval $cmd
}

function _train(){
    echo "current CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, model_name=${model_name}, device_num=${device_num}, is profiling=${profiling}"
    cd ${workdir}
    train_cmd="python3.10 ${model_item}.py TRAIN.epochs=${epochs}"
    echo "train_cmd: ${train_cmd}"
    timeout 15m ${train_cmd} > ${train_log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
    else
        echo -e "${model_name}, SUCCESS"
    fi
    cd $(pwd)
}

_set_params $@
export frame_version=`python -c "import paddle;print(paddle.__version__)"`
export frame_commit=`python -c "import paddle;print(paddle.__git_commit__)"`
export model_branch=`git rev-parse HEAD`
echo "---------Paddle version = ${frame_version}"
echo "---------Paddle commit = ${frame_commit}"
echo "---------PaddleScience commit = ${model_branch}"

job_bt=`date '+%Y%m%d%H%M%S'`
_train
job_et=`date '+%Y%m%d%H%M%S'`
export model_run_time=$((${job_et}-${job_bt}))
_analysis_log
