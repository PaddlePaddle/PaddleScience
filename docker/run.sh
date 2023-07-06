cd examples/bracket
wget https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_data.tar
tar -xvf bracket_data.tar

cd ..
docker build . -t paddlescience:latest

if [ -x "$(command -v podman)" ]; then
    nvidia-docker run -it -v $PWD:/work paddlescience
elif [ -x "$(command -v docker)" ]; then
    docker run --gpus all -it -v $PWD/:/work paddlescience
else
    echo "docker 和 nvidia-docker 都未安装,请先安装！"
fi