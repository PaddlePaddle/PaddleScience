# IJCAI_Code

The paddle version of the top three in each track of the IJCAI 2024 competition.

Inference codes only now.

## Dataset

Please refer to the .ipynb files in each directory to download the data and set the corresponding parameters.

## Checkpoint

Donwload checkpoints:

``` sh
cd PaddleScience/jointContribution/IJCAI_2024
# linux
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/models/contrib/IJCAI_2024_ckpts.tar.gz
# windows
# curl https://paddle-org.bj.bcebos.com/paddlescience/models/contrib/IJCAI_2024_ckpts.tar.gz
```

Unzip the checkpoints and move them to the corresponding directory:

``` sh
tar -xvzf IJCAI_2024_ckpts.tar.gz

# aminos
mkdir -p ./aminos/Logger/states/
mv ./ckpts/aminos/90.pdparams ./aminos/Logger/states/90.pdparams

# tenfeng
mkdir -p ./results/
mv ./ckpts/tenfeng/checkpoint.pdparams ./tenfeng/results/checkpoint.pdparams

# leejt
mv ./ckpts/leejt/model.pdparams ./leejt/model.pdparams

# bju
mv ./ckpts/bju/geom/ckpt ./bju/geom/
mv ./ckpts/bju/pretrained_checkpoint.pdparams ./bju/pretrained_checkpoint.pdparams

# zhongzaicanyu
mv ./ckpts/zhongzaicanyu/pretrained_checkpoint.pdparams ./zhongzaicanyu/pretrained_checkpoint.pdparams
```

## Inference

First enter the corresponding directory. For example "aminos":

``` sh
cd aminos
```

Install requirements:

``` sh
pip install -r requirements.txt
```

Run Inference:

``` py
### aminos
python infer.py --dataset_dir "./Datasets" --load_index="90"

### tenfeng
python infer.py --epochs 69 --milestones 40 50 60 65 68 --gpu_id 0  --depth 5 --hidden_dim 256 --num_slices 32 --batch_size 4 --loss_type 'rl2' --submit --log_dir "./results" --training_data_dir "./Dataset/train_track_B_e" --testing_data_dir "./Dataset/Testset_track_B_e"

### leejt
python infer.py

### bju
python infer.py --train_data_dir "./Dataset/Trainset_track_B" --test_data_dir "./Dataset/Testset_track_B/Inference" --info_dir "./Dataset/Testset_track_B/Auxiliary" --ulip_ckpt "./geom/ckpt/checkpoint_pointbert.pdparams"

### zhongzaicanyu
python infer.py --data_dir "./Dataset/data_centroid_track_B_vtk" --test_data_dir "./Dataset/track_B_vtk" --save_dir "./Dataset/data_centroid_track_B_vtk_preprocessed_data"
```
