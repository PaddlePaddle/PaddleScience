#!/usr/bin/env bash
######################## get params
model_repo=${1:-"model_repo"}
job_name=${2:-"job_name"}
run_model_config_type=${3:-"run_model_config_type"}
run_model_sh=${4:-"run_model_sh"}
########################
echo --------- 训练前启动开启统计资源消耗的进程
    # 统计gpu
    if [[ ${pscpu} == "" ]];then
        echo -e " --------- running on gpu --------- "
        rm -rf ${PDC_LOG}/benchmark_log/gpu_use.log
        nvidia-smi --id=0,1,2,3,4,5,6,7 --query-gpu=utilization.gpu,memory.used --format=csv -lms 100 > ${PDC_LOG}/benchmark_log/gpu_use.log 2>&1 &
        gpu_memory_pid=$!
    fi
    # 统计CPU
    time=$(date "+%Y-%m-%d %H:%M:%S")
    LAST_CPU_INFO=$(cat /proc/stat | grep -w cpu | awk '{print $2,$3,$4,$5,$6,$7,$8}')
    LAST_SYS_IDLE=$(echo $LAST_CPU_INFO | awk '{print $4}')
    LAST_TOTAL_CPU_T=$(echo $LAST_CPU_INFO | awk '{print $1+$2+$3+$4+$5+$6+$7}')
    echo gpu_memory_pid:${gpu_memory_pid}, time: ${time}
echo  --------- 模型开始训练
    cd ./${model_repo};
    ls; 
    if [[ ${run_model_config_type} == 'dynamic'* ]];then  
        echo ---------- 执行动态图模型, TIPC规范 ---------- ;
        tipc_model_txt=${run_model_sh}
        if [[ ${model_repo} =~ 'Speech' ]] || [[ ${model_repo} =~ 'NLP' ]]  ;then      # Speech 的执行目录在tests/下
            cd tests/;
        fi;
        echo ---------- ${job_name} prepare data ---------- ;
        bash test_tipc/prepare.sh ${tipc_model_txt} benchmark_train ;
        pip list;
        echo ---------- ${job_name} ${run_model_config_type} running ---------- ;
        bash test_tipc/benchmark_train.sh ${tipc_model_txt} benchmark_train ${run_model_config_type} ; 

    elif [[ ${run_model_config_type} == 'static'* ]];then  
        echo ---------- 执行静态图模型, 竞品规范 ---------- ;
        static_model_sh=${run_model_sh}
        if [[ ${model_repo} =~ 'PaddleSeg' ]];then      # PaddleSeg 的静态图执行目录在legacy/下
            cd legacy/;
        fi
        if [[ ${model_repo} =~ 'PaddleNLP' ]];then      # PaddleNLP 的静态图执行目录在tests/下
            cd tests/;
        fi
        ls;
        ls ${static_model_sh};
        echo ---------- ${job_name} ${run_model_config_type} running ---------- ;
        bash ${static_model_sh};
    else  
        echo ---------- 执行竞品模型 ---------- ;
        other_model_sh=${run_model_sh}
        echo ---------- ${job_name} ${run_model_config_type} running ---------- ;
        bash ${other_model_sh};
    fi;

echo " --------- 模型训练结束,计算资源消耗"
    # 运行结束后，计算CPU和gpu的资源消耗
    NEXT_CPU_INFO=$(cat /proc/stat | grep -w cpu | awk '{print $2,$3,$4,$5,$6,$7,$8}')
    NEXT_SYS_IDLE=$(echo $NEXT_CPU_INFO | awk '{print $4}')
    NEXT_TOTAL_CPU_T=$(echo $NEXT_CPU_INFO | awk '{print $1+$2+$3+$4+$5+$6+$7}')

    #系统空闲时间
    SYSTEM_IDLE=`echo ${NEXT_SYS_IDLE} ${LAST_SYS_IDLE} | awk '{print $1-$2}'`
    #CPU总时间
    TOTAL_TIME=`echo ${NEXT_TOTAL_CPU_T} ${LAST_TOTAL_CPU_T} | awk '{print $1-$2}'`
    echo "LAST_SYS_IDLE:" $LAST_SYS_IDLE
    echo "NEXT_SYS_IDLE:" $NEXT_SYS_IDLE
    echo "LAST_TOTAL_CPU_T:" $LAST_TOTAL_CPU_T
    echo "NEXT_TOTAL_CPU_T:" $NEXT_TOTAL_CPU_T
    echo "SYSTEM_IDLE:" $SYSTEM_IDLE
    echo "TOTAL_TIME: " $TOTAL_TIME
    AVG_CPU_USE=0
    if [ $TOTAL_TIME == 0 ];then  # 两次系统的总时间一致,说明CPU的使用的时间计划为0
        AVG_CPU_USE=0
    else
        AVG_CPU_USE=`echo ${SYSTEM_IDLE} ${TOTAL_TIME} | awk '{printf "%.2f", (1-$1/$2)*100}'`
    fi

    # 计算显存占用
    MAX_GPU_MEMORY_USE=0
    AVG_GPU_USE=0
    if [[ ${pscpu} == "" ]];then
        kill ${gpu_memory_pid}
        MAX_GPU_MEMORY_USE=`awk 'BEGIN {max = 0} {if(NR>1){if ($3 > max) max=$3}} END {print max}' ${PDC_LOG}/benchmark_log/gpu_use.log`
        AVG_GPU_USE=`awk '{if(NR>1 && $1 >0){time+=$1;count+=1}} END{if(count>0) avg=time/count; else avg=0; printf("%.2f\n" ,avg)}' ${PDC_LOG}/benchmark_log/gpu_use.log`
    fi
    echo "------------ 资源消耗: {AVG_CPU_USE: $AVG_CPU_USE %,MAX_GPU_MEMORY_USE: $MAX_GPU_MEMORY_USE MiB, AVG_GPU_USE: $AVG_GPU_USE %}"
echo ------------ 将资源消耗写进 speed 文件中
    # PY脚本读取speed再将资源消耗写进去
    python ${BENCHMARK_ROOT}/scripts/GetResourceUtilization.py --speed_log_file ${LOG_PATH_INDEX_DIR}/${job_name}_speed --AVG_CPU_USE ${AVG_CPU_USE}  --MAX_GPU_MEMORY_USE ${MAX_GPU_MEMORY_USE} --AVG_GPU_USE ${AVG_GPU_USE}

