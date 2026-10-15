#!/bin/bash
#SBATCH --job-name=evalood-evo2
#SBATCH --partition=day
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1080ti:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=61G
#SBATCH --array=0
#SBATCH --output=logs/eval-ood-%A_%a.out
set -euo pipefail
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate dna-inversion

SLUGS=(evo2)
FM=${SLUGS[$SLURM_ARRAY_TASK_ID]}

python evaluate.py \
    run_dir="outputs/train_105k_${FM}" \
    eval_ood_1000g=true \
    identification_only=false identification.enabled=false \
    hydra.run.dir="outputs_1000g/eval_105k_${FM}"
