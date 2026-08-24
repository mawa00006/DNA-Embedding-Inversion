#!/bin/bash
set -euo pipefail

# Real-individual OOD reconstruction sets for models trained by
# prepare_and_generate_multilen.sh.

MAX_JOBS=${MAX_JOBS:-3}
NUM_GPUS=${NUM_GPUS:-3}
SLEEP_TIME=${SLEEP_TIME:-5}
HG38_DATA_DIR=${HG38_DATA_DIR:-data/hg38_multilen_seqdisjoint}
DATA_DIR=${DATA_DIR:-data/1000g_multilen_seqdisjoint}
SOURCE_HG38=${SOURCE_HG38:-$HG38_DATA_DIR/hg38_multilen_seqdisjoint.h5}
SOURCE_1000G=${SOURCE_1000G:-$DATA_DIR/1000g_multilen_seqdisjoint.h5}
NUM_SEQUENCES=${NUM_SEQUENCES:-15000}
LENGTHS=(10 15 20 25 30 35 40 45 50 60 70 80 90 100)
FMS=(dnabert2 evo2 ntv2)
JOB_IDX=0

wait_for_jobs() {
    while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep "$SLEEP_TIME"
    done
}

generation_complete() {
    python -c "import h5py,sys; from src.generation_io import generation_complete; f=h5py.File(sys.argv[2],'r'); expected=str(f['lengths/'+sys.argv[3]].attrs['ordered_sequence_sha256']); f.close(); sys.exit(0 if generation_complete(sys.argv[1], expected) else 1)" "$1" "$2" "$3" 2>/dev/null
}

source_complete() {
    python -c "import h5py,sys; f=h5py.File(sys.argv[1],'r'); ok=(f.attrs.get('schema','') == 'dna_inversion_1000g_multilen_v2' and bool(f.attrs.get('generation_complete',False))); f.close(); sys.exit(0 if ok else 1)" "$1" 2>/dev/null
}

if source_complete "$SOURCE_1000G"; then
    echo "[Skip] 1000G source corpus is complete: $SOURCE_1000G"
else
    if [ -e "$SOURCE_1000G" ]; then
        echo "ERROR: $SOURCE_1000G exists but is incomplete. Move it aside first." >&2
        exit 1
    fi
    python scripts/prepare_1000g_multilen.py \
        --source "$SOURCE_HG38" \
        --output "$SOURCE_1000G" \
        --num-sequences "$NUM_SEQUENCES"
fi

mkdir -p "$DATA_DIR"
for LEN in "${LENGTHS[@]}"; do
    for FM in "${FMS[@]}"; do
        TEST="$DATA_DIR/test_${FM}_${LEN}_1000g_multilen_seqdisjoint.h5"
        if generation_complete "$TEST" "$SOURCE_1000G" "$LEN"; then
            echo "[Skip] $TEST is complete."
            continue
        fi
        GPU_ID=$((JOB_IDX % NUM_GPUS))
        wait_for_jobs
        echo "[$FM] Generating length-$LEN 1000G OOD embeddings on GPU $GPU_ID..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python "generate/generate_${FM}_embeddings.py" \
            input_path="$SOURCE_1000G" \
            seq_length="$LEN" \
            num_sequences="$NUM_SEQUENCES" \
            mean=true \
            deduplicate_sequences=false \
            train_split=1.0 \
            val_split=0.0 \
            train_output_path="$TEST" \
            val_output_path="$TEST" \
            test_output_path="$TEST" \
            update_config=false &
        JOB_IDX=$((JOB_IDX + 1))
    done
done

wait
echo "1000G multi-length OOD embedding generation complete."
