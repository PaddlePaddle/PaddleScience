git clone https://github.com/PaddlePaddle/PaddleScience.git

cd PaddleScience && git checkout -b develop 6839369
export PYTHONPATH=$PWD:$PYTHONPATH
cd examples/bracket
wget https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_data.tar
tar -xvf bracket_data.tar 

cd ..
docker build . -t paddlescience:latest
docker run --gpus all -it -v ./:/work paddlescience
