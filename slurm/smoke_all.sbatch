#!/bin/bash
#SBATCH --job-name=smoke-all
#SBATCH --partition=L40Sday
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=92G
#SBATCH --array=0-5
#SBATCH --output=logs/smoke-%A_%a.out
set -uo pipefail
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate dna-inversion
export HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1

SLUGS=(dnabert1 grover genalm mistraldna seqlens ntv1)
FM=${SLUGS[$SLURM_ARRAY_TASK_ID]}
D=data/smoke
DC=conf/data/hg38_multilen_seqdisjoint/${FM}_multi_hg38_multilen_seqdisjoint_mean.yaml
EXPECT=$(grep -m1 '^embedding_dim:' "$DC" | awk '{print $2}')

echo "=== $FM (expect embedding_dim=$EXPECT) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for BS in 1024 4096; do
  GC=$(( BS > 2000 ? BS : 2000 ))
  rm -f $D/{train,val,test}_${FM}_25_hg38_multilen_seqdisjoint.h5
  START=$SECONDS
  python generate/generate_${FM}_embeddings.py \
      input_path=$D/hg38_smoke.h5 seq_length=25 num_sequences=4000 \
      mean=true deduplicate_sequences=false val_size=200 test_size=200 \
      batch_size=$BS gen_chunk_size=$GC \
      train_output_path=$D/train_${FM}_25_hg38_multilen_seqdisjoint.h5 \
      val_output_path=$D/val_${FM}_25_hg38_multilen_seqdisjoint.h5 \
      test_output_path=$D/test_${FM}_25_hg38_multilen_seqdisjoint.h5 \
      update_config=false || { echo "RESULT $FM FAIL (generator exited $?, bs=$BS)"; exit 1; }
  echo "TIMING $FM bs=$BS ${GC} $((SECONDS-START))s"
done

python - "$FM" "$EXPECT" <<'PY'
import h5py, sys
fm, expect = sys.argv[1], int(sys.argv[2])
p = f"data/smoke/train_{fm}_25_hg38_multilen_seqdisjoint.h5"
with h5py.File(p) as f:
    a = dict(f.attrs); shape = f["embeddings"].shape
ok = a.get("generation_complete") and a.get("embedding_dim") == expect and shape == (3600, expect)
print(f"  attrs: dim={a.get('embedding_dim')} complete={a.get('generation_complete')} "
      f"layer={a.get('resolved_layer_index')}/{a.get('num_hidden_states')} shape={shape}")
print(f"RESULT {fm} {'PASS' if ok else 'FAIL'}")
PY
