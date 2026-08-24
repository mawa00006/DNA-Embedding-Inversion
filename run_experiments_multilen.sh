#!/bin/bash
set -euo pipefail

# End-to-end benchmark for the coordinate- and sequence-disjoint release.

MAX_JOBS=${MAX_JOBS:-1}
NUM_GPUS=${NUM_GPUS:-1}
SLEEP_TIME=${SLEEP_TIME:-5}
MULTI_EPOCHS=${MULTI_EPOCHS:-20}
BWA_THREADS=${BWA_THREADS:-8}
ID_REPETITIONS=${ID_REPETITIONS:-50}
export MAX_JOBS NUM_GPUS SLEEP_TIME BWA_THREADS ID_REPETITIONS
FMS=(dnabert2 evo2 ntv2)
JOB_IDX=0

wait_for_jobs() {
    while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep "$SLEEP_TIME"
    done
}

echo "===== Data: coordinate- and sequence-disjoint hg38 ====="
bash prepare_and_generate_multilen.sh

echo "===== Data: real-individual 1000G OOD ====="
bash prepare_and_generate_1000g_multilen.sh

echo "===== Data: 1000G identification cohorts ====="
bash prepare_and_generate_identification_multilen.sh --full

echo "===== Train one multi-length Query Decoder per FM ====="
for FM in "${FMS[@]}"; do
    GPU_ID=$((JOB_IDX % NUM_GPUS))
    wait_for_jobs
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py -m \
        data="hg38_multilen_seqdisjoint/${FM}_multi_hg38_multilen_seqdisjoint_mean" \
        model=query_decoder \
        optim.epochs="$MULTI_EPOCHS" \
        hydra.sweep.dir="outputs/train_multilen_seqdisjoint_${FM}" \
        hydra.sweep.subdir=\${hydra.job.override_dirname} \
        hydra.job.chdir=True &
    JOB_IDX=$((JOB_IDX + 1))
done
wait

echo "===== Evaluate held-out hg38 and real-individual 1000G ====="
for FM in "${FMS[@]}"; do
    TRAIN_DIR="outputs/train_multilen_seqdisjoint_${FM}"
    GPU_ID=$((JOB_IDX % NUM_GPUS))
    wait_for_jobs
    CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py \
        run_dir="$TRAIN_DIR" \
        identification_only=false \
        identification.enabled=false \
        hydra.run.dir="outputs/eval_multilen_seqdisjoint_${FM}" &
    JOB_IDX=$((JOB_IDX + 1))

    GPU_ID=$((JOB_IDX % NUM_GPUS))
    wait_for_jobs
    CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py \
        run_dir="$TRAIN_DIR" \
        eval_ood_1000g=true \
        identification_only=false \
        identification.enabled=false \
        hydra.run.dir="outputs_1000g/eval_multilen_seqdisjoint_${FM}" &
    JOB_IDX=$((JOB_IDX + 1))
done
wait

echo "===== Evaluate cohort identification ====="
for FM in "${FMS[@]}"; do
    GPU_ID=$((JOB_IDX % NUM_GPUS))
    wait_for_jobs
    CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py \
        run_dir="outputs/train_multilen_seqdisjoint_${FM}" \
        identification_only=true \
        identification.h5_pattern="data/1000g/test_{fm}_{L}_1000g_id_multilen_seqdisjoint_{mode}.h5" \
        identification.num_target_repetitions="$ID_REPETITIONS" \
        identification.locus_blind.bwa_threads="$BWA_THREADS" \
        identification.locus_blind.workdir_root="outputs/locus_blind_workdirs_multilen_seqdisjoint" \
        hydra.run.dir="outputs/identification_multilen_seqdisjoint_${FM}" &
    JOB_IDX=$((JOB_IDX + 1))
done
wait

python plot_cross_dataset.py \
    --eval-dirs outputs/eval_multilen_seqdisjoint_dnabert2 outputs/eval_multilen_seqdisjoint_evo2 outputs/eval_multilen_seqdisjoint_ntv2 \
    --inversion-model query_decoder \
    --output-dir outputs/cross_dataset_comparison_multilen_seqdisjoint

python plot_cross_dataset.py \
    --eval-dirs outputs_1000g/eval_multilen_seqdisjoint_dnabert2 outputs_1000g/eval_multilen_seqdisjoint_evo2 outputs_1000g/eval_multilen_seqdisjoint_ntv2 \
    --inversion-model query_decoder \
    --output-dir outputs_1000g/cross_dataset_comparison_multilen_seqdisjoint

echo "Multi-length revision pipeline complete."
