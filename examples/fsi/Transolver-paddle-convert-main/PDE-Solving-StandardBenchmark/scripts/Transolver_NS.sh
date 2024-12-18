export CUDA_VISIBLE_DEVICES=3

python exp_ns.py \
--gpu 3 \
--model Transolver_Structured_Mesh_2D \
--n-hidden 256 \
--n-heads 8 \
--n-layers 8 \
--lr 0.001 \
--batch-size 2 \
--slice_num 32 \
--unified_pos 1 \
--ref 8 \
--eval 0 \
--save_name ns_Transolver

