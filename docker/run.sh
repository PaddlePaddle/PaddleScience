docker build . -t paddlescience:latest

if [ -x "$(command -v nvidia-docker)" ]; then
    nvidia-docker run -it -v $PWD:/work paddlescience
elif [ -x "$(command -v docker)" ]; then
    docker run --gpus all -it -v $PWD:/work paddlescience
else
    echo "Please install docker or nvidia-docker first."
fi
