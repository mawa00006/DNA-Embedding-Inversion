#!/bin/bash
set -euo pipefail

# Coordinate- and sequence-disjoint multi-length data path. This pipeline first
# creates one source HDF5 whose intervals are disjoint across every length/split
# and which contains no repeated exact sequence within a length.

MAX_JOBS=${MAX_JOBS:-3}
NUM_GPUS=${NUM_GPUS:-3}
SLEEP_TIME=${SLEEP_TIME:-5}
SEED=${SEED:-42}
DATA_DIR=${DATA_DIR:-data/hg38_multilen_seqdisjoint}
SOURCE_H5=${SOURCE_H5:-$DATA_DIR/hg38_multilen_seqdisjoint.h5}
HG38_FASTA=${HG38_FASTA:-data/hg38.fa}
LENGTHS=(10 15 20 25 30 35 40 45 50 60 70 80 90 100)
FMS=(dnabert2 evo2 ntv2)
VAL_SIZE=50000
TEST_SIZE=50000
JOB_IDX=0

declare -A TOTAL_SEQS=(
    [10]=1000000 [15]=1000000 [20]=1000000
    [25]=2000000 [30]=2000000 [35]=2000000 [40]=2000000 [45]=2000000
    [50]=3000000 [60]=3000000 [70]=3000000 [80]=3000000
    [90]=4000000 [100]=4000000
)

wait_for_jobs() {
    while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep "$SLEEP_TIME"
    done
}

source_complete() {
    python -c "import h5py,sys; p=sys.argv[1]; f=h5py.File(p,'r'); ok=(f.attrs.get('schema','') == 'dna_inversion_hg38_multilen_v2' and bool(f.attrs.get('generation_complete',False)) and bool(f.attrs.get('sequence_content_split_disjoint',False))); f.close(); sys.exit(0 if ok else 1)" "$1" 2>/dev/null
}

generation_complete() {
    python -c "import h5py,sys; from src.generation_io import generation_complete; f=h5py.File(sys.argv[2],'r'); expected=str(f['lengths/'+sys.argv[3]].attrs['ordered_sequence_sha256']); f.close(); sys.exit(0 if generation_complete(sys.argv[1], expected) else 1)" "$1" "$2" "$3" 2>/dev/null
}

if source_complete "$SOURCE_H5"; then
    echo "[Ready] Cleaned source corpus: $SOURCE_H5"
else
    if [ -e "$SOURCE_H5" ]; then
        echo "ERROR: $SOURCE_H5 exists but is incomplete. Move it aside first." >&2
        exit 1
    fi
    echo "===== Building coordinate- and sequence-disjoint hg38 corpus ====="
    python scripts/prepare_hg38_multilen.py \
        --fasta "$HG38_FASTA" \
        --output "$SOURCE_H5" \
        --seed "$SEED" \
        --val-size "$VAL_SIZE" \
        --test-size "$TEST_SIZE"
    source_complete "$SOURCE_H5"
fi

mkdir -p "$DATA_DIR"
for LEN in "${LENGTHS[@]}"; do
    TOTAL=${TOTAL_SEQS[$LEN]}
    for FM in "${FMS[@]}"; do
        TRAIN="$DATA_DIR/train_${FM}_${LEN}_hg38_multilen_seqdisjoint.h5"
        VAL="$DATA_DIR/val_${FM}_${LEN}_hg38_multilen_seqdisjoint.h5"
        TEST="$DATA_DIR/test_${FM}_${LEN}_hg38_multilen_seqdisjoint.h5"
        if generation_complete "$TRAIN" "$SOURCE_H5" "$LEN"; then
            echo "[Skip] $FM length $LEN embeddings are complete."
            continue
        fi
        GPU_ID=$((JOB_IDX % NUM_GPUS))
        wait_for_jobs
        echo "[$FM] Embedding length $LEN ($TOTAL unique windows) on GPU $GPU_ID..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python "generate/generate_${FM}_embeddings.py" \
            input_path="$SOURCE_H5" \
            seq_length="$LEN" \
            num_sequences="$TOTAL" \
            mean=true \
            deduplicate_sequences=false \
            val_size="$VAL_SIZE" \
            test_size="$TEST_SIZE" \
            train_output_path="$TRAIN" \
            val_output_path="$VAL" \
            test_output_path="$TEST" \
            update_config=false &
        JOB_IDX=$((JOB_IDX + 1))
    done
done

wait
echo "Sequence-disjoint multi-length embedding generation complete."
