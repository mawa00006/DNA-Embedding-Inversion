#!/bin/bash
#SBATCH --job-name=probe-train
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1080ti:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=61G
#SBATCH --output=logs/probe-%j.out
set -euo pipefail
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate dna-inversion

python train.py -m \
    data=hg38_multilen_seqdisjoint/hyenadna_large_multi_hg38_multilen_seqdisjoint_mean \
    model=query_decoder \
    data.subset_fraction=0.01 \
    optim.epochs=1 \
    optim.num_workers=5 \
    hydra.sweep.dir=outputs/probe_hyenadna_large \
    hydra.sweep.subdir='${hydra.job.override_dirname}' \
    hydra.job.chdir=True
