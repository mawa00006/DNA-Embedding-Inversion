#!/bin/bash
#SBATCH --job-name=train-hyenadna
#SBATCH --partition=week
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1080ti:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=61G
#SBATCH --output=logs/train-%j.out
set -euo pipefail

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate dna-inversion

SCRIPT=train_hyenadna.sbatch
SWEEP_DIR=outputs/train_multilen_seqdisjoint_hyenadna_large
ROUND=${ROUND:-1}
MAX_ROUNDS=${MAX_ROUNDS:-6}

echo "[$(date '+%F %T')] round $ROUND/$MAX_ROUNDS on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The Hydra output dir is derived from the override string, so these overrides
# must be BYTE-IDENTICAL on every resubmission or the run starts from scratch in
# a new directory instead of resuming from checkpoint.pt.
python train.py -m \
    data=hg38_multilen_seqdisjoint/hyenadna_large_multi_hg38_multilen_seqdisjoint_mean \
    model=query_decoder \
    optim.epochs=15 \
    optim.num_workers=5 \
    hydra.sweep.dir="$SWEEP_DIR" \
    hydra.sweep.subdir='${hydra.job.override_dirname}' \
    hydra.job.chdir=True

# model.pt is written only on completion; checkpoint.pt is deleted at that point.
# If it is absent, the walltime cut training short -- resubmit to resume.
if find "$SWEEP_DIR" -name model.pt -print -quit 2>/dev/null | grep -q .; then
    echo "[$(date '+%F %T')] training complete"
elif [ "$ROUND" -lt "$MAX_ROUNDS" ]; then
    echo "[$(date '+%F %T')] incomplete -- resubmitting round $((ROUND + 1))"
    sbatch --export=ALL,ROUND=$((ROUND + 1)),MAX_ROUNDS=$MAX_ROUNDS "$SCRIPT"
else
    echo "[$(date '+%F %T')] incomplete and MAX_ROUNDS reached -- investigate"
fi
