#!/usr/bin/env bash

function _run(){
    ps -ef
    killall -9 python
    sleep 9
    ps -ef
    # export model_branch=`git symbolic-ref HEAD 2>/dev/null | cut -d"/" -f 3`
    # export model_commit=$(git log|head -n1|awk '{print $2}')
    str_tmp=$(echo `pip list|grep paddlepaddle-gpu|awk -F ' ' '{print $2}'`)
    export frame_version=${str_tmp%%.post*}
    export frame_commit=$(echo `python -c "import paddle;print(paddle.version.commit)"`)
    echo "---------frame_version is ${frame_version}"
    echo "---------Paddle commit is ${frame_commit}"
    echo "---------Model commit is ${model_commit}"
    echo "---------model_branch is ${model_branch}"

    job_bt=`date '+%Y%m%d%H%M%S'`
    _train
    job_et=`date '+%Y%m%d%H%M%S'`
    export model_run_time=$((${job_et}-${job_bt}))
    if [ "${separator}" == "" ]; then
        separator="None"
    fi
    analysis_options=""
    if [ "${position}" != "" ]; then
        analysis_options="${analysis_options} --position ${position}"
    fi
    if [ "${range}" != "" ]; then
        analysis_options="${analysis_options} --range ${range}"
    fi
    if [ "${model_mode}" != "" ]; then
        analysis_options="${analysis_options} --model_mode ${model_mode}"
    fi
    if [ "${speed_unit}" != "" ]; then
        analysis_options="${analysis_options} --speed_unit ${speed_unit}"
    fi
    if [ "${convergence_key}" != "" ]; then
        analysis_options="${analysis_options} --convergence_key ${convergence_key}"
    fi
    if [ ${profiling} = "false" ]; then     # 开启 profiling 时目前不使用该脚本进行日志解析
        python ${BENCHMARK_ROOT}/scripts/analysis.py \
            --filename ${log_file} \
            --log_with_profiler ${profiling_log_file:-"not found!"} \
            --profiler_path ${profiler_path:-"not found!"} \
            --speed_log_file ${speed_log_file} \
            --model_name ${model_name} \
            --base_batch_size ${base_batch_size} \
            --run_mode ${run_mode} \
            --fp_item ${fp_item} \
            --keyword ${keyword} \
            --skip_steps ${skip_steps} \
            --device_num ${device_num} \
            --separator "${separator}" ${analysis_options}
    fi
}
