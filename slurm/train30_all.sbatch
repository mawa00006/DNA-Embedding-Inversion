#!/bin/bash
#SBATCH --job-name=train30
#SBATCH --partition=day
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1080ti:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=61G
#SBATCH --array=0-11%6
#SBATCH --output=logs/train30-%A_%a.out
set -euo pipefail
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate dna-inversion

SLUGS=(ntv2 ntv1 aidodna seqlens hyenadna_large genalm evo2 grover caduceus mistraldna dnabert1 dnabert2)
FM=${SLUGS[$SLURM_ARRAY_TASK_ID]}
echo "[$(date '+%F %T')] $FM 30 epochs  7500/length = 105000 train"

python train.py -m \
    data="hg38_multilen_seqdisjoint/${FM}_multi_hg38_multilen_seqdisjoint_mean" \
    model=query_decoder \
    data.max_samples=7500 \
    train.max_val_samples=1000 \
    optim.epochs=30 \
    optim.num_workers=5 \
    hydra.sweep.dir="outputs/train_105k_30ep_${FM}" \
    hydra.sweep.subdir='${hydra.job.override_dirname}' \
    hydra.job.chdir=True
echo "[$(date '+%F %T')] done $FM"
