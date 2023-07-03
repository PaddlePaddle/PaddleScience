cd examples/bracket
wget https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_data.tar
tar -xvf bracket_data.tar

cd ..
docker build . -t paddlescience:latest
docker run --gpus all -it -v $pwd/:/work paddlescience
