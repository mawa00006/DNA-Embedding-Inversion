# Analysis scripts

Run against the evaluation output trees after the SLURM jobs finish.

| script | what it does |
|---|---|
| `check_digests.py` | verifies every embedding file is complete and carries the current corpus hash; run before training |
| `collect_results.py` | hg38 accuracy table, per model and per sequence length |
| `collect_all.py`, `collect_all_30ep.py` | hg38 and 1000G side by side; the `_30ep` variant reads the 30-epoch trees |
| `collect_variants.py`, `collect_variants_30ep.py` | splits the 1000G results by variant-bearing vs reference-matching and by seen vs unseen in training |
| `plot_val_curves_single.py` | validation loss curves for all models in one figure |

Usage, from the repository root:

    python analysis/check_digests.py ntv2 dnabert2
    python analysis/collect_all_30ep.py

Cross-model figures come from `plot_cross_dataset.py` at the repository root:

    python plot_cross_dataset.py --inversion-model query_decoder \
        --eval-dirs $(ls -d outputs/eval_105k_30ep_*) \
        --output-dir outputs/cross_dataset_30ep
