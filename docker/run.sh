docker build . -t paddlescience:latest

if [ -x "$(command -v nvidia-docker)" ]; then
    nvidia-docker run --network=host -it paddlescience
elif [ -x "$(command -v docker)" ]; then
    docker run --gpus all --network=host -it paddlescience
else
    echo "Please install docker or nvidia-docker first."
fi
