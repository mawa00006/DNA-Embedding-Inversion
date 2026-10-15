#!/bin/bash
#SBATCH --job-name=gen-caduceus
#SBATCH --partition=L40Sday
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=92G
#SBATCH --array=0-13%7
#SBATCH --output=logs/gen-%A_%a.out
set -euo pipefail
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate dna-caduceus
export HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1

FM=${FM:?set FM}
BS=${BS:-4096}
GC=${GC:-4096}
DIR=data/hg38_multilen_seqdisjoint

LENGTHS=(10 15 20 25 30 35 40 45 50 60 70 80 90 100)
declare -A TOTAL=( [10]=1000000  [15]=1000000  [20]=1000000 \
                   [25]=2000000  [30]=2000000  [35]=2000000 \
                   [40]=2000000  [45]=2000000  [50]=3000000 \
                   [60]=3000000  [70]=3000000  [80]=3000000 \
                   [90]=4000000  [100]=4000000 )

L=${LENGTHS[$SLURM_ARRAY_TASK_ID]}
N=${TOTAL[$L]}
echo "[$(date '+%F %T')] $FM L=$L N=$N bs=$BS host=$(hostname)"

python "generate/generate_${FM}_embeddings.py" \
    input_path="$DIR/hg38_multilen_seqdisjoint.h5" \
    seq_length="$L" num_sequences="$N" \
    mean=true deduplicate_sequences=false \
    val_size=50000 test_size=50000 \
    batch_size="$BS" gen_chunk_size="$GC" \
    train_output_path="$DIR/train_${FM}_${L}_hg38_multilen_seqdisjoint.h5" \
    val_output_path="$DIR/val_${FM}_${L}_hg38_multilen_seqdisjoint.h5" \
    test_output_path="$DIR/test_${FM}_${L}_hg38_multilen_seqdisjoint.h5" \
    update_config=false
echo "[$(date '+%F %T')] done $FM L=$L"
