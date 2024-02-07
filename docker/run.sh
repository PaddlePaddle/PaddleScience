docker build . -t paddlescience:latest

if [ -x "$(command -v nvidia-docker)" ]; then
    nvidia-docker run --network=host -it paddlescience
elif [ -x "$(command -v docker)" ]; then
    docker run --gpus all --network=host -it paddlescience
else
    echo "Docker start failed, please install nvidia-docker or docker(>=19.03) first"
fi
