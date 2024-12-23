export CUDA_VISIBLE_DEVICES=0

python exp_darcy.py \
--gpu 0 \
--model 1 \
--n-hidden 128 \
--n-heads 8 \
--n-layers 8 \
--lr 0.001 \
--max_grad_norm 0.1 \
--batch-size 4 \
--slice_num 64 \
--unified_pos 1 \
--ref 8 \
--eval 0 \
--downsample 5 \
--save_name darcy_UniPDE

