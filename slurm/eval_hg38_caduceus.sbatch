#!/bin/bash
#SBATCH --job-name=eval-caduceus
#SBATCH --partition=day
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1080ti:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=61G
#SBATCH --array=0
#SBATCH --output=logs/eval-hg38-%A_%a.out
set -euo pipefail
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate dna-inversion

SLUGS=(caduceus)
FM=${SLUGS[$SLURM_ARRAY_TASK_ID]}

python evaluate.py \
    run_dir="outputs/train_105k_${FM}" \
    identification_only=false identification.enabled=false \
    hydra.run.dir="outputs/eval_105k_${FM}"
