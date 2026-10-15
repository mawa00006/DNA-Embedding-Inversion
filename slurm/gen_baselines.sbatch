#!/bin/bash
#SBATCH --job-name=gen-base
#SBATCH --partition=week
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=45G
#SBATCH --array=0-13%7
#SBATCH --output=logs/gen-base-%A_%a.out
set -euo pipefail
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate dna-inversion
export HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1

FM=${FM:?set FM}
BS=${BS:-256}
DIR=data/hg38_multilen_seqdisjoint
LENGTHS=(10 15 20 25 30 35 40 45 50 60 70 80 90 100)
L=${LENGTHS[$SLURM_ARRAY_TASK_ID]}

echo "[$(date '+%F %T')] $FM L=$L bs=$BS $(hostname)"
python "generate/generate_${FM}_embeddings.py" \
    input_path="$DIR/hg38_multilen_seqdisjoint.h5" \
    seq_length="$L" \
    mean=true deduplicate_sequences=false \
    val_size=50000 test_size=50000 \
    batch_size="$BS" \
    train_output_path="$DIR/train_${FM}_${L}_hg38_multilen_seqdisjoint.h5" \
    val_output_path="$DIR/val_${FM}_${L}_hg38_multilen_seqdisjoint.h5" \
    test_output_path="$DIR/test_${FM}_${L}_hg38_multilen_seqdisjoint.h5" \
    update_config=false
