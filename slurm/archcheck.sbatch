#!/bin/bash
#SBATCH --job-name=archcheck
#SBATCH --partition=test
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/archcheck-%j.out
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate dna-inversion
python - <<'PY'
import torch
print("torch      ", torch.__version__)
print("available  ", torch.cuda.is_available())
print("arch_list  ", torch.cuda.get_arch_list())
print("device     ", torch.cuda.get_device_name(0))
print("capability ", torch.cuda.get_device_capability(0))
x = torch.randn(2000, 2000, device="cuda")
print("matmul ok  ", float((x @ x).sum()))
PY
