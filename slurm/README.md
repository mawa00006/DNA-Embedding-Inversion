# SLURM job scripts

The scripts that produced the reported results, run on the TCML cluster
(SLURM, mixed GTX 1080 Ti / RTX 2080 Ti / RTX A4000 / L40S nodes).

Most take the model slug and batch size from the environment, e.g.

    sbatch --export=ALL,FM=ntv2,BS=256 --gres=gpu:2080ti:1 --array=0-13%7 gen_l40s.sbatch

Array indices map to sequence lengths in the order
`10 15 20 25 30 35 40 45 50 60 70 80 90 100`.

## Pipeline order

| stage | script | notes |
|---|---|---|
| embedding extraction, hg38 | `gen_l40s.sbatch` | base conda env; one array task per sequence length |
| | `gen_caduceus.sbatch` | needs the `dna-caduceus` env |
| | `gen_aidodna.sbatch` | needs the `dna-aidodna` env |
| | `gen_evo2.sbatch` | needs the `dna-evo2` env and an L40S |
| attack training | `train_105k.sbatch` | 15 epochs, 7500 sequences per length |
| | `train30_all.sbatch` | 30 epochs, the budget used for the reported results |
| evaluation, hg38 | `eval_hg38.sbatch`, `eval_hg38_30ep.sbatch` | |
| embedding extraction, 1000G | `gen_1000g.sbatch` and the per-model variants | |
| evaluation, 1000G | `eval_ood.sbatch`, `eval_ood_30ep.sbatch` | writes to `outputs_1000g/` |
| identification (prepared, not run) | `id_smoke.sbatch`, `prepare_and_generate_identification.sh` | smoke stage only |

The `train_<model>_105k.sbatch`, `eval_*_<model>.sbatch` and
`gen_1000g_<model>.sbatch` files are single-model variants derived from the
array versions above, used when one model had to be rerun on its own.

## GPU requirements

Not every model runs on every card. An untyped `--gres=gpu:1` request is the
cause of most failures seen during this project.

| model | requires | reason |
|---|---|---|
| DNABERT-2 | A4000 or L40S | Triton flash-attention kernels need 82,944 B of shared memory; Turing provides 65,536 |
| Caduceus, AIDO.DNA | 2080 Ti or newer | prebuilt CUDA wheels ship no kernels for Pascal (sm_61) |
| Evo 2 | L40S | StripedHyena-2 kernels require compute capability >= 8.9 |
| NT-v1 | A4000 or larger | 10.15 GB of fp32 weights |
| all others | any | |
