#!/bin/bash
# Prepare per-individual 1000G sequences and generate embeddings for the
# identification evaluation.
#
# Two stages:
#   Stage A (smoke): N=50, K=100, one length, one FM, one locus mode.
#   Stage B (full):  N=$COHORT_SIZE, K=$NUM_LOCI over LENGTHS x FMs x modes.
#
# Pass --full to skip the interactive prompt and run Stage B directly.

set -euo pipefail

REFERENCE="data/hg38.fa"
VCF_DIR="data/1000g_vcfs"
# The sequence-disjoint wrapper sets both variables below.
SOURCE_MULTILEN_H5=${SOURCE_MULTILEN_H5:-}
ID_VARIANT_PREFIX=${ID_VARIANT_PREFIX:-}

# Must match identification.seq_lengths in conf/evaluate.yaml (the lengths the
# identification eval scores). Generating other lengths just wastes compute.
LENGTHS=(25)
LOCUS_MODES=(random snp_enriched)
# All three foundation models are scored for identification. The dataset has
# N_cohort x K_loci x 2 haplotypes rows, so cost scales with N*K, not K alone.
# At N=1000, K=4096, 2 hap = ~8.2M mean vectors/mode: DNABERT-2 (768-d) ~26 GB,
# NTv2 (1024-d) ~34 GB, Evo2 (4096-d) ~131 GB per mode. Generation is resumable
# (existing H5s are skipped), so stage Evo2 separately if disk is tight.
FMS=(ntv2 dnabert2 evo2)
COHORT_SIZE=1000
# Total loci per individual. K=4096 gives a long k-axis (4x the old 1000) so
# identification_model can fit (d, r) and extrapolate the k required for a target
# hit rate, while keeping the eval's (K, N, N) score tensor ~16 GB at N=1000.
NUM_LOCI=4096
SEED=42
MIN_LOCUS_SPACING=50000
MAF_MIN=0.05

# Smoke-test parameters.
SMOKE_LENGTH=50
SMOKE_FM=ntv2
SMOKE_MODE=random
SMOKE_N=50
SMOKE_K=100

# Parallelism for embedding generation.
MAX_JOBS=${MAX_JOBS:-10}
SLEEP_TIME=${SLEEP_TIME:-5}
NUM_GPUS=${NUM_GPUS:-2}

FULL=0
for arg in "$@"; do
    case "$arg" in
        --full) FULL=1 ;;
        *) echo "Unknown arg: $arg" >&2; exit 2 ;;
    esac
done

wait_for_jobs() {
    while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep "$SLEEP_TIME"
    done
}

# Helper: is a resumable identification H5 already COMPLETE? Reads only the H5
# completion marker (cheap), so a partial file returns non-zero and is resumed by
# the generator instead of being skipped wholesale. Absent file -> non-zero.
id_complete() {
    python -c "import sys; from src.generation_io import identification_complete,identification_csv_input_sha256; expected=identification_csv_input_sha256(sys.argv[2]); sys.exit(0 if identification_complete(sys.argv[1], expected) else 1)" "$1" "$2" 2>/dev/null
}

# bcftools sanity check.
if ! command -v bcftools >/dev/null 2>&1; then
    echo "ERROR: bcftools not on PATH. Activate the project conda env first." >&2
    exit 1
fi

# The locus-blind identification path aligns predicted sequences to hg38 with
# BWA-MEM, so the index must exist before the eval runs. Idempotent: skips if
# the index is already built.
bash scripts/build_bwa_index.sh

prepare_one() {
    local LEN=$1
    local N=$2
    local K=$3
    local MODE=$4
    local OUT_CSV="data/1000g/1000g_id_${ID_VARIANT_PREFIX}${MODE}_seq${LEN}_N${N}_K${K}.csv"
    if [ -f "$OUT_CSV" ]; then
        echo "[Skip] $OUT_CSV already exists."
        return
    fi
    echo "[Prepare] $OUT_CSV (mode=$MODE, L=$LEN, N=$N, K=$K)..."
    local SOURCE_ARGS=()
    if [ -n "$SOURCE_MULTILEN_H5" ]; then
        SOURCE_ARGS=(--source_multilen_h5 "$SOURCE_MULTILEN_H5")
    fi
    python scripts/prepare_identification.py \
        --reference "$REFERENCE" \
        --vcf_dir "$VCF_DIR" \
        --output_csv "$OUT_CSV" \
        --seq_length "$LEN" \
        --cohort_size "$N" \
        --num_loci "$K" \
        --locus_mode "$MODE" \
        --min_locus_spacing "$MIN_LOCUS_SPACING" \
        --maf_min "$MAF_MIN" \
        --seed "$SEED" \
        "${SOURCE_ARGS[@]}"
}

generate_one() {
    local FM=$1
    local LEN=$2
    local N=$3
    local K=$4
    local MODE=$5
    local IN_CSV="data/1000g/1000g_id_${ID_VARIANT_PREFIX}${MODE}_seq${LEN}_N${N}_K${K}.csv"
    if [ ! -f "$IN_CSV" ]; then
        echo "[Skip] $IN_CSV missing for FM=$FM"
        return
    fi
    local OUT_H5="data/1000g/test_${FM}_${LEN}_1000g_id_${ID_VARIANT_PREFIX}${MODE}.h5"
    if id_complete "$OUT_H5" "$IN_CSV"; then
        echo "[Skip] $OUT_H5 already complete."
        return
    fi
    local GEN_SCRIPT="generate/generate_${FM}_embeddings.py"
    if [ ! -f "$GEN_SCRIPT" ]; then
        echo "[Skip] Missing generator script $GEN_SCRIPT"
        return
    fi
    local GPU_ID=$(( ${JOB_IDX:-0} % NUM_GPUS ))
    echo "[$FM] Generating identification embeddings (L=$LEN, mode=$MODE) on GPU $GPU_ID..."
    wait_for_jobs
    CUDA_VISIBLE_DEVICES=$GPU_ID python "$GEN_SCRIPT" \
        input_path="$IN_CSV" \
        seq_length="$LEN" \
        num_sequences=99999999 \
        mean=true \
        identification_mode=true \
        update_config=false \
        train_output_path="data/1000g/dummy_${FM}_${LEN}_1000g_id_${ID_VARIANT_PREFIX}${MODE}.h5" \
        val_output_path="data/1000g/dummy_${FM}_${LEN}_1000g_id_${ID_VARIANT_PREFIX}${MODE}.h5" \
        test_output_path="$OUT_H5" &
    JOB_IDX=$(( ${JOB_IDX:-0} + 1 ))
}

run_smoke() {
    echo "===== Stage A: smoke test ====="
    echo "L=$SMOKE_LENGTH N=$SMOKE_N K=$SMOKE_K FM=$SMOKE_FM mode=$SMOKE_MODE"
    prepare_one "$SMOKE_LENGTH" "$SMOKE_N" "$SMOKE_K" "$SMOKE_MODE"
    JOB_IDX=0
    generate_one "$SMOKE_FM" "$SMOKE_LENGTH" "$SMOKE_N" "$SMOKE_K" "$SMOKE_MODE"
    wait
    echo "Smoke test complete."
}

run_full() {
    echo "===== Stage B: full identification dataset ====="
    echo "LENGTHS=${LENGTHS[*]}  FMs=${FMS[*]}  modes=${LOCUS_MODES[*]}"
    echo "N=$COHORT_SIZE  K=$NUM_LOCI"

    # Stage B prep (sequential CPU-bound bcftools work).
    for LEN in "${LENGTHS[@]}"; do
        for MODE in "${LOCUS_MODES[@]}"; do
            prepare_one "$LEN" "$COHORT_SIZE" "$NUM_LOCI" "$MODE"
        done
    done

    # Stage B embedding generation (parallel, round-robin GPUs).
    JOB_IDX=0
    for LEN in "${LENGTHS[@]}"; do
        for MODE in "${LOCUS_MODES[@]}"; do
            for FM in "${FMS[@]}"; do
                generate_one "$FM" "$LEN" "$COHORT_SIZE" "$NUM_LOCI" "$MODE"
            done
        done
    done
    wait
    echo "Full identification dataset complete."
}

run_smoke

if [ "$FULL" -eq 1 ]; then
    run_full
else
    echo ""
    echo "Smoke test finished. Re-run with --full to launch the full job:"
    echo "  $0 --full"
fi
