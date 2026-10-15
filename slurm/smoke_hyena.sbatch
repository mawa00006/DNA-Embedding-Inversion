#!/bin/bash
#SBATCH --job-name=smoke-hyena
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/smoke-%j.out
set -euo pipefail
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate dna-inversion
export HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch;print('torch',torch.__version__,torch.cuda.get_device_name(0),torch.cuda.get_device_capability())"

D=data/smoke
for BS in 256 1024 4096; do
  GC=$(( BS > 2000 ? BS : 2000 ))
  rm -f $D/{train,val,test}_hyenadna_large_25_hg38_multilen_seqdisjoint.h5
  echo "=== batch_size=$BS gen_chunk_size=$GC ==="
  START=$SECONDS
  python generate/generate_hyenadna_large_embeddings.py \
      input_path=$D/hg38_smoke.h5 seq_length=25 num_sequences=4000 \
      mean=true deduplicate_sequences=false val_size=200 test_size=200 \
      batch_size=$BS gen_chunk_size=$GC \
      train_output_path=$D/train_hyenadna_large_25_hg38_multilen_seqdisjoint.h5 \
      val_output_path=$D/val_hyenadna_large_25_hg38_multilen_seqdisjoint.h5 \
      test_output_path=$D/test_hyenadna_large_25_hg38_multilen_seqdisjoint.h5 \
      update_config=false
  echo "TIMING bs=$BS $((SECONDS-START))s"
done

python -c "
import h5py
f=h5py.File('$D/train_hyenadna_large_25_hg38_multilen_seqdisjoint.h5')
print(dict(f.attrs)); print('shape', f['embeddings'].shape)"
