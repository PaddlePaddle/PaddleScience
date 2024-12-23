export CUDA_VISIBLE_DEVICES=3

python exp_elas.py \
--gpu 3 \
--model Transolver_Irregular_Mesh \
--n-hidden 128 \
--n-heads 8 \
--n-layers 8 \
--lr 0.001 \
--max_grad_norm 0.1 \
--batch-size 1 \
--slice_num 64 \
--unified_pos 0 \
--ref 8 \
--eval 1 \
--save_name elas_Transolver
