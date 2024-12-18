export CUDA_VISIBLE_DEVICES=3

python main.py \
--cfd_model=Transolver \
--gpu 3 \
--preprocessed 1 \
--data_dir data/PDE_data/mlcfd_data/training_data \
--save_dir data/PDE_data/mlcfd_data/preprocessed_data \
