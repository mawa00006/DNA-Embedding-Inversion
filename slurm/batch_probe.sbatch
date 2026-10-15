#!/bin/bash
#SBATCH --job-name=batch-probe
#SBATCH --partition=L40Sday
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=92G
#SBATCH --output=logs/batch-probe-%j.out
set -uo pipefail          # NOT -e: one bad model must not kill the run

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate dna-inversion
export HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch;print('torch',torch.__version__,torch.cuda.get_device_name(0),torch.cuda.get_device_capability())"

for f in conf/generate/*.yaml; do
    slug=$(basename "$f" .yaml)
    [ "$slug" = "evo2" ] && continue
    ck=$(grep -m1 '^checkpoint:' "$f" | awk '{print $2}')
    rev=$(grep -m1 '^revision:'  "$f" | awk '{print $2}')
    [ -n "$ck" ] || continue
    echo ""
    echo "################ $slug ################"
    if [ -n "$rev" ]; then
        python scripts/probe_checkpoint.py "$ck" --revision "$rev" 2>&1 | tail -45
    else
        python scripts/probe_checkpoint.py "$ck" 2>&1 | tail -45
    fi
    echo "################ end $slug ################"
done
