#!/bin/bash
#SBATCH --job-name=train-aidodna
#SBATCH --partition=day
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1080ti:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=61G
#SBATCH --array=0
#SBATCH --output=logs/train105k-%A_%a.out
set -euo pipefail
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate dna-inversion

SLUGS=(aidodna)
FM=${SLUGS[$SLURM_ARRAY_TASK_ID]}
echo "[$(date '+%F %T')] $FM  7500/length = 105000 train"

python train.py -m \
    data="hg38_multilen_seqdisjoint/${FM}_multi_hg38_multilen_seqdisjoint_mean" \
    model=query_decoder \
    data.max_samples=7500 \
    train.max_val_samples=1000 \
    optim.epochs=15 \
    optim.num_workers=5 \
    hydra.sweep.dir="outputs/train_105k_${FM}" \
    hydra.sweep.subdir='${hydra.job.override_dirname}' \
    hydra.job.chdir=True
