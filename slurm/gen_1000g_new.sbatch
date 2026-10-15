#!/bin/bash
#SBATCH --job-name=gen1000g-new
#SBATCH --partition=day
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1080ti:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=61G
#SBATCH --array=0-1
#SBATCH --output=logs/gen1000g-%A_%a.out
set -euo pipefail
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate dna-inversion
export HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1

SLUGS=(ntv2 dnabert1)
FM=${SLUGS[$SLURM_ARRAY_TASK_ID]}
D=data/1000g_multilen_seqdisjoint
SRC=$D/1000g_multilen_seqdisjoint.h5

for LEN in 10 15 20 25 30 35 40 45 50 60 70 80 90 100; do
  OUT=$D/test_${FM}_${LEN}_1000g_multilen_seqdisjoint.h5
  echo "[$(date '+%F %T')] $FM L=$LEN"
  python "generate/generate_${FM}_embeddings.py" \
      input_path="$SRC" seq_length="$LEN" \
      mean=true deduplicate_sequences=false \
      train_split=1.0 val_split=0.0 \
      train_output_path="$OUT" val_output_path="$OUT" test_output_path="$OUT" \
      batch_size=512 gen_chunk_size=512 update_config=false
done
echo "[$(date '+%F %T')] done $FM"
