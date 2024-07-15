docker build . -t paddlescience:latest

if [ -x "$(command -v nvidia-docker)" ]; then
    nvidia-docker run --name paddlescience_container --network=host -it paddlescience
elif [ -x "$(command -v docker)" ]; then
    docker run --name paddlescience_container --gpus all --network=host -it paddlescience
else
    echo "Docker start failed, please install nvidia-docker or docker(>=19.03) first"
fi
