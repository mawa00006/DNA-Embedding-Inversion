#!/bin/bash
#SBATCH --job-name=id-smoke
#SBATCH --partition=day
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=61G
#SBATCH --output=logs/id-smoke-%j.out
set -euo pipefail
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate dna-inversion
export HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1
export NUM_GPUS=1 MAX_JOBS=1
bash prepare_and_generate_identification_multilen.sh
