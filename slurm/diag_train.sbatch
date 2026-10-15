#!/bin/bash
#SBATCH --job-name=diag-train
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1080ti:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=61G
#SBATCH --output=logs/diag-%j.out
set -euo pipefail
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate dna-inversion

( while true; do
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader
    sleep 5
  done ) &
SAMPLER=$!

python train.py -m \
    data=hg38_multilen_seqdisjoint/hyenadna_large_multi_hg38_multilen_seqdisjoint_mean \
    model=query_decoder \
    data.subset_fraction=0.002 \
    optim.epochs=1 \
    optim.num_workers=5 \
    hydra.sweep.dir=outputs/diag_hyenadna_large \
    hydra.sweep.subdir='${hydra.job.override_dirname}' \
    hydra.job.chdir=True

kill $SAMPLER
