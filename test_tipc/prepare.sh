# Set directories
export PDSC_DIR=$(cd "$( dirname ${BASH_SOURCE[0]})"; cd ..; pwd)
export TEST_DIR="${PDSC_DIR}"
export TIPC_TEST="ON" # open tipc log in solver.py 
export PYTHONPATH=${PDSC_DIR}

BENCHMARK_ROOT="${TEST_DIR}/test_tipc/tools"
echo -e "\n* [TEST_DIR] is now set : \n" ${TEST_DIR} "\n"
echo -e "\n* [BENCHMARK_ROOT] is now set : \n" ${BENCHMARK_ROOT} "\n"
echo -e "\n* [PYTHONPATH] is now set : \n" ${PYTHONPATH} "\n"

source ${TEST_DIR}/test_tipc/common_func.sh

# Read parameters from [/tipc/config/*/*.txt]
PREPARE_PARAM_FILE=$1
dataline=`cat $PREPARE_PARAM_FILE`
lines=(${dataline})
download_dataset=$(func_parser_value "${lines[61]}")
python=$(func_parser_value "${lines[2]}")
export pip=$(func_parser_value "${lines[62]}")
workdir=$(func_parser_value "${lines[63]}")
${pip} install --upgrade pip
${pip} install pybind11
${pip} install -r requirements.txt

if [ ${download_dataset} ] ; then
    cd ${PDSC_DIR}${workdir}
    ${python} ${PDSC_DIR}${download_dataset}
fi
