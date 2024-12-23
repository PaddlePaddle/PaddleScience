export CUDA_VISIBLE_DEVICES=3

python main_evaluation.py \
--cfd_model=Transolver \
--data_dir data/PDE_data/mlcfd_data/training_data \
--save_dir data/PDE_data/mlcfd_data/preprocessed_data