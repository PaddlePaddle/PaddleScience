export BATCH_SIZE=16

# Prediction experiments
# for CFD-GCN
mpirun -np $((BATCH_SIZE+1)) --oversubscribe python main.py --batch-size $BATCH_SIZE --gpus 1 -dw 1 --su2-config config/coarse.cfg --model cfd_gcn --hidden-size 512 --num-layers 6 --num-end-convs 3 --optim adam -lr 5e-4 --data-dir data/NACA0012_interpolate --coarse-mesh meshes/mesh_NACA0012_xcoarse.su2 -e cfd_gcn_interp > /dev/null
# for GCN
mpirun -np $((BATCH_SIZE+1)) --oversubscribe python main.py --batch-size $BATCH_SIZE --gpus 1 -dw 1 --model gcn --hidden-size 512 --num-layers 6 --optim adam -lr 5e-4 --data-dir data/NACA0012_interpolate/ -e gcn_interp

# Generalization experiments
# for CFD-GCN
mpirun -np $((BATCH_SIZE+1)) --oversubscribe python main.py --batch-size $BATCH_SIZE --gpus 1 -dw 1 --su2-config config/coarse.cfg --model cfd_gcn --hidden-size 512 --num-layers 6 --num-end-convs 3 --optim adam -lr 5e-4 --data-dir data/NACA0012_machsplit_noshock --coarse-mesh meshes/mesh_NACA0012_xcoarse.su2 -e cfd_gcn_gen > /dev/null
# for GCN
mpirun -np $((BATCH_SIZE+1)) --oversubscribe python main.py --batch-size $BATCH_SIZE --gpus 1 -dw 1 --model gcn --hidden-size 512 --num-layers 6 --optim adam -lr 5e-4 --data-dir data/NACA0012_machsplit_noshock/ -e gcn_gen
