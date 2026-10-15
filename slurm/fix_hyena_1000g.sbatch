#!/bin/bash
#SBATCH --job-name=fix-hyena-1000g
#SBATCH --partition=day
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1080ti:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=61G
#SBATCH --output=logs/fix-hyena-1000g-%j.out
set -euo pipefail
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate dna-inversion
export HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1

D=data/1000g_multilen_seqdisjoint
for LEN in 90 100; do
  OUT=$D/test_hyenadna_large_${LEN}_1000g_multilen_seqdisjoint.h5
  echo "[$(date '+%F %T')] L=$LEN"
  python generate/generate_hyenadna_large_embeddings.py \
      input_path=$D/1000g_multilen_seqdisjoint.h5 seq_length=$LEN \
      mean=true deduplicate_sequences=false train_split=1.0 val_split=0.0 \
      train_output_path="$OUT" val_output_path="$OUT" test_output_path="$OUT" \
      batch_size=512 gen_chunk_size=2000 update_config=false
done
