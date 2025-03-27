#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=16
#SBATCH --mem 200G
#SBATCH -p optimal
#SBATCH -A optimal 
# nvidia-smi

module load anaconda/2022.10 
source activate /ailab/user/chenpengan/liuxinyuan/xinyuan_env

for sd in 1327 1337 1347; do
    for sp in split split2 split3; do
        for MMD in 0.0001; do
            python ../../CRISP/train_script.py \
            --config ../configs/sci.yaml \
            --split $sp \
            --seed $sd \
            --savedir ../results/sci/sci_${sp}_${sd} \
            --MMD $MMD 
        done;
    done;
done