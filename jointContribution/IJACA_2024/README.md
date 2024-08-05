# IJACA_Code
The paddle version of the top three in each track of the IJACA 2024 competition.
Inference codes only now.

## Dataset(To be added)
Please refer to the .ipynb files in each directory to download the data and set the corresponding parameters.

## Checkpoint(To be added)

## Inference commands
### aminos
python infer.py --dataset_dir /your_path/Datasets

### tenfeng
python infer.py --epochs 69 --milestones 40 50 60 65 68 --gpu_id 0  --depth 5 --hidden_dim 256 --num_slices 32 --batch_size 4 --loss_type 'rl2' --submit --log_dir your_path --training_data_dir /your_path/Dataset/train_track_B_e --testing_data_dir /your_path/Dataset/Testset_track_B_e

### leejt
python infer.py

### bju
python infer.py

### zhongzaicanyu
python infer.py
