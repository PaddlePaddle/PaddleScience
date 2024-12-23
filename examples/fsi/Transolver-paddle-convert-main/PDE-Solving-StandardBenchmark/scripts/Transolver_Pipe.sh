export CUDA_VISIBLE_DEVICES=1

python exp_pipe.py \
--gpu 1 \
--model Transolver_Structured_Mesh_2D \
--n-hidden 128 \
--n-heads 8 \
--n-layers 8 \
--mlp_ratio 2 \
--lr 0.001 \
--max_grad_norm 0.1 \
--batch-size 8 \
--slice_num 64 \
--unified_pos 0 \
--ref 8 \
--eval 1 \
--save_name pipe_Transolver

