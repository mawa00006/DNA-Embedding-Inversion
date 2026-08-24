"""
DNA Embedding Inversion Attack - Evaluation Entry Point
"""

from __future__ import annotations

import os
import re

# Disable tokenizer parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import logging
from typing import Dict, List, Any, NamedTuple, Tuple

import time
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import (
    find_latest_run_dir,
    load_run_config,
    save_json,
    NUCLEOTIDES,
)
from src.plotting_utils import (
    configure_plot_style,
    get_fm_color,
    plot_nucleotide_frequencies,
    plot_accuracy_metrics,
    plot_per_position_accuracy,
    plot_nucleotide_confusion_matrix,
    plot_levenshtein_comparison,
)
from src.tokenizers import CharacterTokenizer, HuggingFaceTokenizer
from src.data import load_split_embeddings, create_dataset, compute_pooled_train_stats
import hydra.utils as hy_utils
import h5py
from src.evaluate import (
    reconstruct_sequences,
    compute_sequence_accuracy,
    compare_nucleotide_distributions,
    compute_all_levenshtein_similarities,
    load_model_from_run,
    save_sequences_to_csv,
    generate_random_baseline_sequences,
    compute_per_position_accuracy,
    compute_nucleotide_confusion_matrix,
    compute_nucleotide_frequencies,
    compute_shannon_entropy,
    compute_repetitiveness,
)
from src.evaluation.multirun import (
    is_multirun_directory,
    get_multirun_subdirs,
    aggregate_and_plot_multirun,
)
from src.evaluation.identification import evaluate_identification_block

# Use Agg backend for non-interactive plotting
matplotlib.use("Agg")

configure_plot_style()


def _pooled_train_stats_from_multi(
    multi_data_cfg: Dict[str, Any],
) -> Dict[str, float]:
    """Re-pool training stats from all per-length train_csvs of a multi run.

    Matches the count-weighted pooling done at training time in
    ``load_multi_split_embeddings`` so that evaluation normalizes inputs with
    the same statistics the model saw during training.
    """
    train_paths = [hy_utils.to_absolute_path(p) for p in multi_data_cfg["train_csvs"]]
    train_files = [h5py.File(p, "r", swmr=True) for p in train_paths]
    try:
        return compute_pooled_train_stats(train_files)
    finally:
        for h5 in train_files:
            h5.close()


def _train_stats_from_single(data_cfg: Any) -> Dict[str, float]:
    """Training stats for a single-length run, read from its train file's attrs.

    Only the ``emb_*`` attributes are touched, so this is cheap enough for paths
    that never open the training data itself -- notably identification, which
    brings its own cohort HDF5 but must still reproduce training-time
    normalization.
    """
    train_path = hy_utils.to_absolute_path(data_cfg["train_csv"])
    with h5py.File(train_path, "r", swmr=True) as h5:
        return compute_pooled_train_stats([h5])


def _decode_sequence(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _training_membership_mask(
    train_h5: h5py.File, sequences: List[str], chunk_size: int = 100_000
) -> np.ndarray:
    """Return whether each test sequence occurs exactly in the training shard."""
    train_sequences: set[str] = set()
    dataset = train_h5["sequences"]
    for start in range(0, len(dataset), chunk_size):
        train_sequences.update(
            _decode_sequence(value)
            for value in dataset[start : min(start + chunk_size, len(dataset))]
        )
    return np.asarray(
        [sequence in train_sequences for sequence in sequences], dtype=np.bool_
    )


def _load_1000g_variant_mask(
    data_cfg: DictConfig, true_sequences: List[str]
) -> np.ndarray | None:
    """Load the auditable reference-difference mask for a 1000G v2 test set."""
    source_path = data_cfg.get("ood_source_corpus")
    if source_path is None:
        return None
    source_path = hy_utils.to_absolute_path(source_path)
    with h5py.File(source_path, "r", swmr=True) as source:
        assert str(source.attrs["schema"]) == "dna_inversion_1000g_multilen_v2"
        group = source[f"lengths/{int(data_cfg.seq_length)}"]
        n = len(true_sequences)
        source_sequences = [
            _decode_sequence(value) for value in group["sequences"][:n]
        ]
        assert source_sequences == true_sequences, (
            "1000G source rows do not align with evaluated embedding rows"
        )
        return np.asarray(group["differs_from_reference"][:n], dtype=np.bool_)


def summarize_reconstruction_strata(
    true_sequences: List[str],
    predicted_sequences: List[str],
    levenshtein_similarities: List[float],
    masks: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, Any]]:
    """Summarize reconstruction metrics for named, auditable row subsets."""
    n = len(true_sequences)
    assert len(predicted_sequences) == n == len(levenshtein_similarities)
    output: Dict[str, Dict[str, Any]] = {}
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=np.bool_)
        assert mask.shape == (n,), (name, mask.shape, n)
        indices = np.flatnonzero(mask)
        item: Dict[str, Any] = {
            "num_sequences": int(len(indices)),
            "fraction_of_evaluation": float(len(indices) / n),
        }
        if len(indices):
            true_subset = [true_sequences[i] for i in indices]
            pred_subset = [predicted_sequences[i] for i in indices]
            lev_subset = np.asarray(
                [levenshtein_similarities[i] for i in indices], dtype=np.float64
            )
            nucleotide_accuracies = []
            for true, pred in zip(true_subset, pred_subset, strict=True):
                denominator = max(len(true), len(pred))
                matches = sum(
                    true[i] == pred[i] for i in range(min(len(true), len(pred)))
                )
                nucleotide_accuracies.append(
                    matches / denominator if denominator else 1.0
                )
            item.update(
                {
                    "accuracy_metrics": compute_sequence_accuracy(
                        true_subset, pred_subset
                    ),
                    "accuracy_mean": float(np.mean(nucleotide_accuracies)),
                    "accuracy_std": float(np.std(nucleotide_accuracies)),
                    "levenshtein_mean": float(np.mean(lev_subset)),
                    "levenshtein_std": float(np.std(lev_subset)),
                    "exact_match_rate": float(
                        np.mean(
                            [
                                true == pred
                                for true, pred in zip(
                                    true_subset, pred_subset, strict=True
                                )
                            ]
                        )
                    ),
                }
            )
        output[name] = item
    return output


def _evaluate_on_test_set(
    model: torch.nn.Module,
    tokenizer,
    data_cfg: DictConfig,
    mode: str,
    run_dir: str,
    output_dir: str,
    foundation_model: str,
    base_model_name: str,
    inversion_model: str,
    display_model_name: str,
    device: torch.device,
    logger: logging.Logger,
    identification_cfg: Dict[str, Any],
    max_samples: int | None,
    train_stats_override: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Run the full per-test-set evaluation pipeline with a pre-loaded model.

    Extracted so the multi-length flow can reuse the same logic across
    multiple per-length test files without reloading the model.
    """
    data_dict, counts_dict, train_stats = load_split_embeddings(data_cfg)
    logger.info(f"Loaded test data: {data_cfg['test_csv']} with {counts_dict['test']} samples")
    if train_stats_override is not None:
        assert train_stats_override, (
            "train_stats_override was supplied but is empty; multi runs must carry "
            "pooled training statistics to reproduce training-time normalization."
        )
        assert train_stats, (
            "Per-length train file is missing emb_* attrs; cannot verify that the "
            "pooled override is compatible with the per-length file's normalization."
        )
        logger.info(
            "Overriding per-length training stats with pooled multi-run stats "
            f"(was mean={train_stats['mean']:.4f}, std={train_stats['std']:.4f}) -> "
            f"(mean={train_stats_override['mean']:.4f}, "
            f"std={train_stats_override['std']:.4f})"
        )
        train_stats = train_stats_override
    if train_stats:
        logger.info(
            f"Training embedding stats (used for normalization) - min: {train_stats['min']:.4f}, "
            f"max: {train_stats['max']:.4f}, mean: {train_stats['mean']:.4f}, "
            f"std: {train_stats['std']:.4f}"
        )

    # Ensure output directory exists before any artifacts are written.
    os.makedirs(output_dir, exist_ok=True)

    test_dataset = create_dataset(
        data=data_dict["test"],
        mode=mode,
        tokenizer=tokenizer,
        embedding_dim=data_cfg.embedding_dim,
        seq_length=data_cfg.seq_length,
        normalization_stats=train_stats if train_stats else None,
        normalization_method=data_cfg.normalization_method,
        data_is_mean=data_cfg.mean,
        subset_fraction=data_cfg.subset_fraction,
        max_samples=max_samples,
    )

    base_dataset = test_dataset

    seq_length = data_cfg.seq_length
    true_sequences = []
    embeddings_list = []

    logger.info("Extracting sequences and embeddings...")
    start_time = time.time()

    for idx in range(len(test_dataset)):
        # Get sequence
        s = base_dataset.h5_file["sequences"][idx]
        if isinstance(s, bytes):
            true_sequences.append(s.decode("utf-8"))
        else:
            true_sequences.append(str(s))

        # Get embedding (already normalized by dataset)
        emb, _ = base_dataset[idx]
        embeddings_list.append(emb.numpy())

    # Prepare data for reconstruction
    # Pass as list directly to handle variable sequence lengths
    test_data_for_reconstruction = {"embeddings": embeddings_list}

    # Reconstruct sequences
    logger.info(f"Reconstructing {len(true_sequences)} sequences from embeddings...")

    # Standard reconstruction for per_token and mean modes
    predicted_sequences = reconstruct_sequences(
        model,
        test_data_for_reconstruction,
        device,
        tokenizer,
        mode,
        data_cfg.seq_length,
        data_cfg.embedding_dim,
        normalization_method=data_cfg.normalization_method,
        data_is_mean=data_cfg.mean,
    )

    assert len(predicted_sequences) == len(true_sequences)
    # Length is now decided by the model (EOS token); no external truncation.
    pred_lens = np.array([len(s) for s in predicted_sequences])
    true_lens = np.array([len(s) for s in true_sequences])
    logger.info(
        f"Predicted length stats: mean={pred_lens.mean():.2f}, "
        f"min={pred_lens.min()}, max={pred_lens.max()}, "
        f"true mean={true_lens.mean():.2f} (target seq_length={seq_length})"
    )
    end_time = time.time()
    total_time = end_time - start_time
    time_per_sequence = total_time / len(predicted_sequences)
    logger.info(
        f"Reconstructed {len(predicted_sequences)} sequences in {total_time:.2f}s ({time_per_sequence:.4f}s/seq)"
    )

    # Compute accuracy metrics
    logger.info("Computing accuracy metrics...")
    accuracy_metrics = compute_sequence_accuracy(true_sequences, predicted_sequences)
    logger.info("Accuracy metrics:")
    for metric, value in accuracy_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    # Compute per-sequence accuracy to get standard deviation
    nucleotide_accuracies = []
    for t, p in zip(true_sequences, predicted_sequences):
        # Matches / max(len_true, len_pred) to be consistent with overall accuracy logic
        max_len = max(len(t), len(p))
        matches = sum(1 for i in range(min(len(t), len(p))) if t[i] == p[i])
        if max_len > 0:
            nucleotide_accuracies.append(matches / max_len)
        else:
            nucleotide_accuracies.append(1.0)

    accuracy_mean = np.mean(nucleotide_accuracies)
    accuracy_std = np.std(nucleotide_accuracies)
    logger.info(f"Accuracy stats: mean={accuracy_mean:.4f}, std={accuracy_std:.4f}")

    # Compute all Levenshtein similarities for distribution analysis
    logger.info("Computing Levenshtein similarities...")
    levenshtein_similarities = compute_all_levenshtein_similarities(
        true_sequences, predicted_sequences
    )
    logger.info(
        f"Levenshtein similarity stats: min={min(levenshtein_similarities):.3f}, "
        f"max={max(levenshtein_similarities):.3f}, mean={np.mean(levenshtein_similarities):.3f}"
    )

    train_seen = _training_membership_mask(data_dict["train"], true_sequences)
    is_1000g = "1000g" in str(data_cfg.test_csv).lower()
    if bool(data_cfg.get("require_sequence_disjoint", False)) and not is_1000g:
        assert not np.any(train_seen), (
            f"Sequence-disjoint hg38 evaluation found {int(train_seen.sum())} test "
            "rows in training"
        )
    stratum_masks = {
        "all": np.ones(len(true_sequences), dtype=np.bool_),
        "train_seen": train_seen,
        "train_unseen": ~train_seen,
    }
    variant_bearing = _load_1000g_variant_mask(data_cfg, true_sequences) if is_1000g else None
    if variant_bearing is not None:
        stratum_masks.update(
            {
                "reference_identical": ~variant_bearing,
                "variant_bearing": variant_bearing,
                "variant_bearing_train_unseen": variant_bearing & ~train_seen,
            }
        )
    reconstruction_strata = summarize_reconstruction_strata(
        true_sequences,
        predicted_sequences,
        levenshtein_similarities,
        stratum_masks,
    )
    logger.info(
        "Evaluation strata: %s",
        ", ".join(
            f"{name}={values['num_sequences']}"
            for name, values in reconstruction_strata.items()
        ),
    )

    # Generate random baseline sequences for comparison
    logger.info("Generating random baseline sequences...")
    baseline_sequences = generate_random_baseline_sequences(true_sequences)
    logger.info(f"Generated {len(baseline_sequences)} random baseline sequences")

    # Compute baseline metrics
    logger.info("Computing baseline metrics...")
    baseline_accuracy = compute_sequence_accuracy(true_sequences, baseline_sequences)
    baseline_levenshtein = compute_all_levenshtein_similarities(true_sequences, baseline_sequences)
    logger.info("Baseline accuracy metrics:")
    for metric, value in baseline_accuracy.items():
        logger.info(f"  {metric}: {value:.4f}")
    logger.info(
        f"Baseline Levenshtein similarity stats: min={min(baseline_levenshtein):.3f}, "
        f"max={max(baseline_levenshtein):.3f}, mean={np.mean(baseline_levenshtein):.3f}"
    )

    # Compute improvement over baseline
    improvement = {
        key: (
            ((accuracy_metrics[key] - baseline_accuracy[key]) / baseline_accuracy[key] * 100)
            if baseline_accuracy[key] != 0
            else 0.0
        )
        for key in accuracy_metrics.keys()
    }
    logger.info("Model improvement over baseline (% increase):")
    for metric, value in improvement.items():
        logger.info(f"  {metric}: {value:.2f}%")

    # Compare nucleotide distributions
    logger.info("Comparing nucleotide distributions...")
    freq_comparison = compare_nucleotide_distributions(true_sequences, predicted_sequences)
    logger.info("True nucleotide frequencies:")
    for nuc, freq in freq_comparison["true"].items():
        logger.info(f"  {nuc}: {freq:.4f}")
    logger.info("Predicted nucleotide frequencies:")
    for nuc, freq in freq_comparison["pred"].items():
        logger.info(f"  {nuc}: {freq:.4f}")

    # Compute per-position accuracy
    logger.info("Computing per-position accuracy...")
    model_position_accuracy = compute_per_position_accuracy(true_sequences, predicted_sequences)
    baseline_position_accuracy = compute_per_position_accuracy(true_sequences, baseline_sequences)
    logger.info(
        f"Model mean position accuracy: {np.mean(model_position_accuracy):.4f}, "
        f"Baseline mean position accuracy: {np.mean(baseline_position_accuracy):.4f}"
    )

    # Compute nucleotide confusion matrix
    logger.info("Computing nucleotide confusion matrix...")
    confusion_matrix = compute_nucleotide_confusion_matrix(true_sequences, predicted_sequences)
    logger.info("Confusion matrix computed")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save numerical results
    # Compute difficulty metrics
    logger.info("Computing difficulty metrics (Entropy, Repetitiveness)...")
    shannon_entropies = compute_shannon_entropy(true_sequences)
    repetitiveness_scores = compute_repetitiveness(true_sequences)

    logger.info(
        f"Shannon Entropy stats (True): mean={np.mean(shannon_entropies):.3f}, std={np.std(shannon_entropies):.3f}"
    )

    # Compute metrics for predicted sequences
    logger.info("Computing metrics for predicted sequences...")
    predicted_shannon_entropies = compute_shannon_entropy(predicted_sequences)
    predicted_repetitiveness_scores = compute_repetitiveness(predicted_sequences)

    logger.info(
        f"Shannon Entropy stats (Pred): mean={np.mean(predicted_shannon_entropies):.3f}, std={np.std(predicted_shannon_entropies):.3f}"
    )

    # Save example reconstructions
    num_examples = len(true_sequences)
    examples = []
    for i in range(num_examples):
        examples.append(
            {
                "index": i,
                "true_sequence": true_sequences[i],
                "predicted_sequence": predicted_sequences[i],
                "match": true_sequences[i] == predicted_sequences[i],
                "seen_in_training": bool(train_seen[i]),
                "differs_from_reference": (
                    None if variant_bearing is None else bool(variant_bearing[i])
                ),
                "shannon_entropy": float(shannon_entropies[i]),
                "repetitiveness": float(repetitiveness_scores[i]),
                "predicted_shannon_entropy": float(predicted_shannon_entropies[i]),
                "predicted_repetitiveness": float(predicted_repetitiveness_scores[i]),
            }
        )
    save_json({"examples": examples}, os.path.join(output_dir, "example_reconstructions.json"))
    logger.info(f"Saved {num_examples} example reconstructions")

    # Save numerical results
    results = {
        "run_dir": run_dir,
        "output_dir": output_dir,
        "evaluation_dataset": (
            "1000g" if "1000g" in str(data_cfg.test_csv).lower() else "hg38"
        ),
        "test_data_path": str(data_cfg.test_csv),
        "seq_length": seq_length,
        "inversion_model": inversion_model,
        "foundation_model": foundation_model,
        "accuracy_metrics": accuracy_metrics,
        "baseline_accuracy_metrics": baseline_accuracy,
        "improvement_over_baseline": improvement,
        "nucleotide_frequencies": freq_comparison,
        "num_sequences": len(true_sequences),
        "levenshtein_mean": float(np.mean(levenshtein_similarities)),
        "levenshtein_std": float(np.std(levenshtein_similarities)),
        "baseline_levenshtein_mean": float(np.mean(baseline_levenshtein)),
        "baseline_levenshtein_std": float(np.std(baseline_levenshtein)),
        "accuracy_mean": float(accuracy_mean),
        "accuracy_std": float(accuracy_std),
        "reconstruction_strata": reconstruction_strata,
        "total_inference_time": total_time,
        "time_per_sequence": time_per_sequence,
        "shannon_entropy_mean": float(np.mean(shannon_entropies)),
        "repetitiveness_mean": float(np.mean(repetitiveness_scores)),
        "predicted_shannon_entropy_mean": float(np.mean(predicted_shannon_entropies)),
        "predicted_repetitiveness_mean": float(np.mean(predicted_repetitiveness_scores)),
    }

    # Save plot data for regeneration
    plot_data = {
        "freq_comparison": freq_comparison,
        "accuracy_metrics": accuracy_metrics,
        "levenshtein_similarities": levenshtein_similarities,
        "baseline_levenshtein": baseline_levenshtein,
        "model_position_accuracy": model_position_accuracy,
        "baseline_position_accuracy": baseline_position_accuracy,
        "confusion_matrix": confusion_matrix,
        "shannon_entropies": shannon_entropies,
        "repetitiveness_scores": repetitiveness_scores,
        "predicted_shannon_entropies": predicted_shannon_entropies,
        "predicted_repetitiveness_scores": predicted_repetitiveness_scores,
        "train_seen_mask": train_seen,
        "variant_bearing_mask": variant_bearing,
    }
    torch.save(plot_data, os.path.join(output_dir, "plot_data.pt"))

    # Create plots
    logger.info("Creating plots...")

    model_color = get_fm_color(display_model_name)

    plot_nucleotide_frequencies(
        freq_comparison,
        os.path.join(output_dir, "nucleotide_frequencies"),
        "Nucleotide Frequency Comparison (Test Set)",
        model_color=model_color,
    )
    logger.info("Saved nucleotide frequency plot")

    plot_accuracy_metrics(
        accuracy_metrics,
        os.path.join(output_dir, "accuracy_metrics"),
        model_color=model_color,
    )
    logger.info("Saved accuracy metrics plot")

    plot_per_position_accuracy(
        model_position_accuracy,
        baseline_position_accuracy,
        os.path.join(output_dir, "per_position_accuracy"),
        model_color=model_color,
    )
    logger.info("Saved per-position accuracy plot")

    plot_nucleotide_confusion_matrix(
        confusion_matrix,
        NUCLEOTIDES,
        os.path.join(output_dir, "confusion_matrix"),
        "Nucleotide Confusion Matrix (Model Predictions)",
        cmap=sns.light_palette(model_color, as_cmap=True),
    )
    logger.info("Saved nucleotide confusion matrix plot")

    plot_levenshtein_comparison(
        levenshtein_similarities,
        baseline_levenshtein,
        os.path.join(output_dir, "levenshtein_comparison"),
        model_color=model_color,
    )
    logger.info("Saved Levenshtein similarity comparison plot")

    # New metrics plots
    from src.plotting_utils import plot_metric_vs_similarity

    plot_metric_vs_similarity(
        shannon_entropies,
        levenshtein_similarities,
        "Shannon Entropy (Bits) [True]",
        os.path.join(output_dir, "levenshtein_vs_entropy_true"),
        model_color=model_color,
    )
    logger.info("Saved Levenshtein vs Entropy (True) plot")

    plot_metric_vs_similarity(
        predicted_shannon_entropies,
        levenshtein_similarities,
        "Shannon Entropy (Bits) [Predicted]",
        os.path.join(output_dir, "levenshtein_vs_entropy_predicted"),
        model_color=model_color,
    )
    logger.info("Saved Levenshtein vs Entropy (Predicted) plot")

    plot_metric_vs_similarity(
        repetitiveness_scores,
        levenshtein_similarities,
        "4-mer Redundancy [True]",
        os.path.join(output_dir, "levenshtein_vs_repetitiveness_true"),
        model_color=model_color,
    )
    logger.info("Saved Levenshtein vs Repetitiveness (True) plot")

    plot_metric_vs_similarity(
        predicted_repetitiveness_scores,
        levenshtein_similarities,
        "4-mer Redundancy [Predicted]",
        os.path.join(output_dir, "levenshtein_vs_repetitiveness_predicted"),
        model_color=model_color,
    )
    logger.info("Saved Levenshtein vs Repetitiveness (Predicted) plot")

    if identification_cfg["enabled"]:
        logger.info("=" * 80)
        logger.info("IDENTIFICATION EVALUATION")
        logger.info("=" * 80)
        identification_run_cfg = dict(identification_cfg)
        identification_run_cfg["training_sequence_h5"] = str(data_cfg.train_csv)
        identification_results = evaluate_identification_block(
            model=model,
            tokenizer=tokenizer,
            device=device,
            foundation_model=foundation_model,
            seq_length=data_cfg.seq_length,
            mode=mode,
            embedding_dim=data_cfg.embedding_dim,
            normalization_stats=train_stats,
            normalization_method=data_cfg.normalization_method,
            data_is_mean=data_cfg.mean,
            identification_cfg=identification_run_cfg,
            output_dir=output_dir,
            logger=logger,
            model_color=model_color,
        )
        results["identification"] = identification_results

    # Write the results JSON once, after the optional identification block, so its
    # presence marks a fully-evaluated (run, length): a run interrupted earlier is
    # recomputed rather than resumed with partial, identification-less results.
    save_json(results, os.path.join(output_dir, "evaluation_results.json"))

    return results


def _build_per_length_data_cfg(
    multi_cfg_dict: Dict[str, Any], idx: int
) -> DictConfig:
    """Construct a single-length data config from a multi-length config.

    Takes the i-th element of train_csvs/val_csvs/test_csvs/seq_lengths and emits a
    config with the regular train_csv/val_csv/test_csv/seq_length keys so the
    existing data loader can consume it unchanged.
    """
    sub = dict(multi_cfg_dict)
    sub["train_csv"] = multi_cfg_dict["train_csvs"][idx]
    sub["val_csv"] = multi_cfg_dict["val_csvs"][idx]
    sub["test_csv"] = multi_cfg_dict["test_csvs"][idx]
    sub["seq_length"] = multi_cfg_dict["seq_lengths"][idx]
    # Per-file SHAs are not tracked in the multi config; rely on the per-length
    # configs for integrity verification when used standalone.
    sub["skip_sha256_check"] = True
    sub["multi"] = False
    for k in ("seq_lengths", "train_seq_lengths", "train_csvs", "val_csvs", "test_csvs"):
        sub.pop(k, None)
    out = OmegaConf.create(sub)
    assert isinstance(out, DictConfig)
    return out


def _resolve_eval_lengths(
    run_dir: str, eval_seq_lengths: List[int] | None
) -> List[int | None]:
    """Pick which sequence lengths to evaluate a (sub-)run at.

    A multi-length training run (``data.multi=true``) is evaluated once per
    length: the explicit ``eval_seq_lengths`` if given, otherwise every length
    the model was trained on (``data.seq_lengths``). A regular single-length run
    is evaluated once at its baked-in length, signalled by a lone ``None`` so the
    caller keeps the un-suffixed output directory name.
    """
    run_config = load_run_config(run_dir)
    data_cfg = run_config["data"]
    is_multi = "multi" in data_cfg and data_cfg["multi"]
    if not is_multi:
        return [None]
    if eval_seq_lengths is not None:
        return list(eval_seq_lengths)
    return list(data_cfg["seq_lengths"])


class _PreparedRun(NamedTuple):
    """Loaded model + resolved data config for one (run, length) evaluation."""

    model: torch.nn.Module
    tokenizer: Any
    data_cfg: DictConfig
    mode: str
    foundation_model: str
    inversion_model: str
    display_model_name: str
    train_stats_override: Dict[str, float] | None


def _run_candidate_lengths(run_dir: str) -> Tuple[bool, List[int]]:
    """Return ``(is_multi, lengths)`` a run can be evaluated at.

    A multi-length run reports its full ``data.seq_lengths``; a regular run
    reports its single baked-in ``data.seq_length``.
    """
    data_cfg = load_run_config(run_dir)["data"]
    is_multi = "multi" in data_cfg and data_cfg["multi"]
    if is_multi:
        return True, list(data_cfg["seq_lengths"])
    return False, [data_cfg["seq_length"]]


def _prepare_run(
    run_dir: str,
    device: torch.device,
    eval_seq_length: int | None,
    logger: logging.Logger,
) -> _PreparedRun:
    """Load the model, tokenizer, and resolved data config for one run.

    Shared by the standard reconstruction eval and the standalone identification
    pass. For multi-length runs, ``eval_seq_length`` selects the per-length test
    config and pooled training stats are recomputed to match training-time
    normalization.
    """
    run_config = load_run_config(run_dir)
    mode = run_config["model"]["mode"]

    model_cfg = run_config["model"]
    base_model_name = model_cfg["model_name"]

    if model_cfg.get("model_type") == "encoder":
        d_model = model_cfg.get("d_model")
        dim_ff = model_cfg.get("dim_feedforward")
        n_layers = model_cfg.get("num_layers")
        params = []
        if d_model:
            params.append(f"d={d_model}")
        if dim_ff:
            params.append(f"df={dim_ff}")
        if n_layers:
            params.append(f"L={n_layers}")
        if params:
            base_model_name = f"Encoder ({', '.join(params)})"

    logger.info(f"Base model name from config: {base_model_name}")

    data_cfg_dict = run_config["data"]
    is_multi = "multi" in data_cfg_dict and data_cfg_dict["multi"]

    # Infer foundation model from data config filename (works for both
    # train_csv and the first entry of train_csvs).
    foundation_model = "unknown"
    if is_multi:
        first_train_csv = data_cfg_dict["train_csvs"][0]
    elif "train_csv" in data_cfg_dict:
        first_train_csv = data_cfg_dict["train_csv"]
    else:
        first_train_csv = None
    if first_train_csv is not None:
        basename = os.path.basename(first_train_csv)
        match = re.search(r"train_(.+?)_\d+_hg38", basename)
        if match:
            foundation_model = match.group(1)

    if foundation_model != "unknown" and foundation_model not in base_model_name:
        model_name = f"{base_model_name} ({foundation_model})"
    else:
        model_name = base_model_name

    logger.info(f"Final model name: {model_name}")
    logger.info(f"Using mode from config: {mode}")

    # Resolve which data_cfg to evaluate on. For multi-length runs, build a
    # per-length sub-config from the chosen `eval_seq_length`; otherwise use
    # the data config baked into the run as-is.
    train_stats_override: Dict[str, float] | None = None
    if is_multi:
        assert eval_seq_length is not None, (
            "Multi-length training run requires eval_seq_length to pick which "
            "per-length test set to evaluate against (e.g. eval_seq_length=50)."
        )
        seq_lengths = list(data_cfg_dict["seq_lengths"])
        assert eval_seq_length in seq_lengths, (
            f"eval_seq_length={eval_seq_length} not in available multi lengths {seq_lengths}"
        )
        idx = seq_lengths.index(eval_seq_length)
        logger.info(
            f"Multi-length run: evaluating at seq_length={eval_seq_length} "
            f"(index {idx} of {len(seq_lengths)})"
        )
        data_cfg = _build_per_length_data_cfg(data_cfg_dict, idx)
        # Reproduce training-time normalization: pool stats across all per-length
        # train files instead of using the single per-L file's stats.
        train_stats_override = _pooled_train_stats_from_multi(data_cfg_dict)
        assert train_stats_override, (
            "Multi training files are missing emb_* attrs; cannot reproduce "
            "training-time pooled normalization for eval."
        )
    else:
        if eval_seq_length is not None:
            logger.info(
                f"eval_seq_length={eval_seq_length} provided but run is not multi-length; "
                "ignoring (regular run keeps its baked-in data config)."
            )
        data_cfg = OmegaConf.create(data_cfg_dict)
        assert isinstance(data_cfg, DictConfig), "Failed to create DictConfig from data config"

    # Load model.
    model = load_model_from_run(run_dir, device)

    # Setup tokenizer.
    tokenizer_cfg = data_cfg_dict["tokenizer"]
    tokenizer_type = tokenizer_cfg["type"]
    if tokenizer_type == "char":
        tokenizer = CharacterTokenizer()
    elif tokenizer_type == "huggingface":
        tokenizer = HuggingFaceTokenizer(tokenizer_cfg["model_name"])
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    return _PreparedRun(
        model=model,
        tokenizer=tokenizer,
        data_cfg=data_cfg,
        mode=mode,
        foundation_model=foundation_model,
        inversion_model=base_model_name,
        display_model_name=model_name,
        train_stats_override=train_stats_override,
    )


def _redirect_test_csv_to_1000g(test_csv: str) -> str:
    """Map an hg38 per-length test path to its 1000G out-of-distribution file.

    The 1000G files live under data/1000g/ with "1000g" in place of "hg38"
    (data/hg38/test_<fm>_<L>_hg38.h5 -> data/1000g/test_<fm>_<L>_1000g.h5). Falls
    back to an "_1000g.h5" suffix if the path has no "hg38" component.
    """
    if "hg38" in test_csv:
        return test_csv.replace("hg38", "1000g")
    return test_csv.replace(".h5", "_1000g.h5")


def evaluate_single_run(
    run_dir: str,
    device: torch.device,
    output_dir: str,
    logger: logging.Logger,
    identification_cfg: Dict[str, Any],
    max_samples: int | None = None,
    eval_seq_length: int | None = None,
    ood_1000g: bool = False,
) -> Dict[str, Any] | None:
    """Evaluate a single training run on one test set.

    Output layout matches the regular (single-length) flow: the metrics, plots,
    and JSON for this run are written directly into ``output_dir``.

    For multi-length training runs (``data.multi=true``), ``eval_seq_length`` must
    be specified — the matching per-length test file is selected from the run's
    multi config and the same trained model is evaluated against it. Each
    invocation evaluates exactly one length; the caller is expected to loop
    externally if multiple lengths are needed (e.g. in run_experiments.sh).
    """
    logger.info(f"Evaluating run: {run_dir}")

    prep = _prepare_run(run_dir, device, eval_seq_length, logger)

    # Out-of-distribution mode: swap the (hg38) test file for its 1000G
    # counterpart, keeping the hg38 training-time normalization. Lengths without
    # a 1000G file are skipped so the caller can drop them from the aggregate.
    if ood_1000g:
        orig_test = str(prep.data_cfg.test_csv)
        patched = _redirect_test_csv_to_1000g(orig_test)
        if not os.path.exists(hy_utils.to_absolute_path(patched)):
            logger.warning(
                f"[1000g] OOD test set not found: {patched}; skipping "
                f"(seq_length={eval_seq_length})."
            )
            return None
        prep.data_cfg.test_csv = patched
        prep.data_cfg.skip_sha256_check = True
        logger.info(f"[1000g] Redirected test_csv {orig_test} -> {patched}")

    return _evaluate_on_test_set(
        model=prep.model,
        tokenizer=prep.tokenizer,
        data_cfg=prep.data_cfg,
        mode=prep.mode,
        run_dir=run_dir,
        output_dir=output_dir,
        foundation_model=prep.foundation_model,
        base_model_name=prep.inversion_model,
        inversion_model=prep.inversion_model,
        display_model_name=prep.display_model_name,
        device=device,
        logger=logger,
        identification_cfg=identification_cfg,
        max_samples=max_samples,
        train_stats_override=prep.train_stats_override,
    )


def run_identification_for_run(
    run_dir: str,
    device: torch.device,
    output_dir: str,
    logger: logging.Logger,
    identification_cfg: Dict[str, Any],
    eval_seq_length: int | None,
) -> Dict[str, Any]:
    """Run ONLY the identification block for one trained run.

    Reloads the model and reuses the reconstruction utilities, but skips the
    standard test-set reconstruction eval. Writes ``identification_results.json``
    and plots into ``output_dir``.
    """
    logger.info(f"Identification for run: {run_dir}")
    prep = _prepare_run(run_dir, device, eval_seq_length, logger)
    os.makedirs(output_dir, exist_ok=True)

    # The cohort HDF5 stores RAW embeddings; the decoder was trained on z-scored
    # ones. Without these stats the decoder sees inputs scaled by 1/std and, for
    # small-std foundation models (Evo 2: std 0.05), collapses to a near-constant
    # output that carries no genotype signal at all.
    train_stats = prep.train_stats_override
    if train_stats is None:
        train_stats = _train_stats_from_single(prep.data_cfg)
    assert train_stats, (
        "Training embedding stats are unavailable, so identification cannot "
        "reproduce training-time normalization. Regenerate the training HDF5 "
        "with emb_* attrs (generate/compute_embedding_statistics.py) first."
    )
    logger.info(
        "[identification] Normalizing cohort embeddings with training stats - "
        f"min: {train_stats['min']:.4f}, max: {train_stats['max']:.4f}, "
        f"mean: {train_stats['mean']:.4f}, std: {train_stats['std']:.4f}"
    )

    identification_run_cfg = dict(identification_cfg)
    identification_run_cfg["training_sequence_h5"] = str(prep.data_cfg.train_csv)
    return evaluate_identification_block(
        model=prep.model,
        tokenizer=prep.tokenizer,
        device=device,
        foundation_model=prep.foundation_model,
        seq_length=prep.data_cfg.seq_length,
        mode=prep.mode,
        embedding_dim=prep.data_cfg.embedding_dim,
        normalization_stats=train_stats,
        normalization_method=prep.data_cfg.normalization_method,
        data_is_mean=prep.data_cfg.mean,
        identification_cfg=identification_run_cfg,
        output_dir=output_dir,
        logger=logger,
        model_color=get_fm_color(prep.display_model_name),
    )


def regenerate_plots_single_run(
    run_dir: str,
    output_dir: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Regenerate plots for a single run using saved data.

    Parameters
    ----------
    run_dir : str
        Path to the training run directory.
    output_dir : str
        Directory to save evaluation results.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing evaluation results (loaded from json).
    """
    logger.info(f"Regenerating plots for run: {run_dir}")

    plot_data_path = os.path.join(run_dir, "plot_data.pt")
    results_path = os.path.join(run_dir, "evaluation_results.json")

    assert os.path.exists(plot_data_path), f"Plot data not found at {plot_data_path}"
    assert os.path.exists(results_path), f"Results file not found at {results_path}"

    # Load data
    plot_data = torch.load(plot_data_path, weights_only=False)

    freq_comparison = plot_data["freq_comparison"]
    accuracy_metrics = plot_data["accuracy_metrics"]
    levenshtein_similarities = plot_data["levenshtein_similarities"]
    baseline_levenshtein = plot_data.get("baseline_levenshtein")
    model_position_accuracy = plot_data["model_position_accuracy"]
    baseline_position_accuracy = plot_data["baseline_position_accuracy"]
    confusion_matrix = plot_data["confusion_matrix"]

    with open(results_path, "r") as f:
        res = json.load(f)
        inversion_model = res["inversion_model"]
        foundation_model = res["foundation_model"]

    # Construct display model name
    if foundation_model != "unknown" and foundation_model not in inversion_model:
        model_name = f"{inversion_model} ({foundation_model})"
    else:
        model_name = inversion_model

    # Create plots
    logger.info("Creating plots...")

    model_color = get_fm_color(model_name)

    plot_nucleotide_frequencies(
        freq_comparison,
        os.path.join(output_dir, "nucleotide_frequencies"),
        "Nucleotide Frequency Comparison (Test Set)",
        model_color=model_color,
    )
    logger.info("Saved nucleotide frequency plot")

    plot_accuracy_metrics(
        accuracy_metrics,
        os.path.join(output_dir, "accuracy_metrics"),
        model_color=model_color,
    )
    logger.info("Saved accuracy metrics plot")

    plot_per_position_accuracy(
        model_position_accuracy,
        baseline_position_accuracy,
        os.path.join(output_dir, "per_position_accuracy"),
        model_color=model_color,
    )
    logger.info("Saved per-position accuracy plot")

    plot_nucleotide_confusion_matrix(
        confusion_matrix,
        NUCLEOTIDES,
        os.path.join(output_dir, "confusion_matrix"),
        "Nucleotide Confusion Matrix (Model Predictions)",
        cmap=sns.light_palette(model_color, as_cmap=True),
    )
    logger.info("Saved nucleotide confusion matrix plot")

    if baseline_levenshtein is not None:
        plot_levenshtein_comparison(
            levenshtein_similarities,
            baseline_levenshtein,
            os.path.join(output_dir, "levenshtein_comparison"),
            model_color=model_color,
        )
        logger.info("Saved Levenshtein similarity comparison plot")

    # Regenerate new metric plots if data exists
    if "shannon_entropies" in plot_data and "repetitiveness_scores" in plot_data:
        from src.plotting_utils import plot_metric_vs_similarity

        shannon_entropies = plot_data["shannon_entropies"]
        repetitiveness_scores = plot_data["repetitiveness_scores"]

        plot_metric_vs_similarity(
            shannon_entropies,
            levenshtein_similarities,
            "Shannon Entropy (Bits) [True]",
            os.path.join(output_dir, "levenshtein_vs_entropy_true"),
            model_color=model_color,
        )
        logger.info("Saved Levenshtein vs Entropy (True) plot")

        plot_metric_vs_similarity(
            repetitiveness_scores,
            levenshtein_similarities,
            "4-mer Redundancy [True]",
            os.path.join(output_dir, "levenshtein_vs_repetitiveness_true"),
            model_color=model_color,
        )
        logger.info("Saved Levenshtein vs Repetitiveness (True) plot")

    if (
        "predicted_shannon_entropies" in plot_data
        and "predicted_repetitiveness_scores" in plot_data
    ):
        # Load if not already loaded (though it should be fine to re-import or use existing)
        from src.plotting_utils import plot_metric_vs_similarity

        predicted_shannon_entropies = plot_data["predicted_shannon_entropies"]
        predicted_repetitiveness_scores = plot_data["predicted_repetitiveness_scores"]

        plot_metric_vs_similarity(
            predicted_shannon_entropies,
            levenshtein_similarities,
            "Shannon Entropy (Bits) [Predicted]",
            os.path.join(output_dir, "levenshtein_vs_entropy_predicted"),
            model_color=model_color,
        )
        logger.info("Saved Levenshtein vs Entropy (Predicted) plot")

        plot_metric_vs_similarity(
            predicted_repetitiveness_scores,
            levenshtein_similarities,
            "4-mer Redundancy [Predicted]",
            os.path.join(output_dir, "levenshtein_vs_repetitiveness_predicted"),
            model_color=model_color,
        )
        logger.info("Saved Levenshtein vs Repetitiveness (Predicted) plot")

    # Add output_dir for aggregate function to find plot_data.pt
    res["output_dir"] = run_dir
    return res


@hydra.main(config_path="conf", config_name="evaluate", version_base=None)
def main(cfg: DictConfig) -> None:
    """Execute evaluation pipeline for trained DNA sequence reconstruction model.

    Supports both single-run and multi-run (Hydra multirun) evaluation modes.
    - Single run: evaluates one model and saves results
    - Multi-run: detects Hydra multirun directory, evaluates each sub-model,
      then creates aggregate comparison plots

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration for evaluation.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting evaluation pipeline")
    logger.info("Config:\n" + OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.device)

    # Find the run directory to evaluate
    if cfg.run_dir is None:
        # Resolve runs_base_dir relative to original CWD
        runs_base_dir = hydra.utils.to_absolute_path(cfg.runs_base_dir)
        run_dir = find_latest_run_dir(runs_base_dir)
        logger.info(f"Auto-detected latest run: {run_dir}")
    else:
        # Resolve specified run_dir relative to original CWD
        run_dir = hydra.utils.to_absolute_path(cfg.run_dir)
        logger.info(f"Using specified run: {run_dir}")

    assert os.path.exists(run_dir), f"Run directory not found: {run_dir}"

    # Get Hydra output directory
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir  # type: ignore[attr-defined]

    # Check if this is a multirun directory
    if cfg.identification_only:
        logger.info("=" * 80)
        logger.info("IDENTIFICATION-ONLY MODE - Running only the identification block")
        assert cfg.identification.enabled, (
            "identification_only=true requires identification.enabled=true"
        )
        id_lengths = set(cfg.identification.seq_lengths)
        assert id_lengths, "identification.seq_lengths is empty; nothing to run"

        if is_multirun_directory(run_dir):
            subdirs = get_multirun_subdirs(run_dir)
        else:
            subdirs = [run_dir]
        logger.info(
            f"Found {len(subdirs)} run(s); restricting to seq_lengths {sorted(id_lengths)}"
        )

        num_run = 0
        for subdir in subdirs:
            job_num = os.path.basename(subdir)
            is_multi, run_lengths = _run_candidate_lengths(subdir)
            for eval_length in run_lengths:
                if eval_length not in id_lengths:
                    continue
                length_suffix = f"_L{eval_length}" if is_multi else ""
                run_output_dir = os.path.join(hydra_output_dir, f"run_{job_num}{length_suffix}")
                results_path = os.path.join(run_output_dir, "identification_results.json")
                if os.path.exists(results_path):
                    logger.info(f"[identification] {results_path} exists, skipping")
                    num_run += 1
                    continue
                run_identification_for_run(
                    subdir,
                    device,
                    run_output_dir,
                    logger,
                    cfg.identification,
                    eval_seq_length=eval_length if is_multi else None,
                )
                num_run += 1

        assert num_run > 0, (
            f"No runs matched identification.seq_lengths={sorted(id_lengths)} under {run_dir}"
        )
        logger.info(f"IDENTIFICATION-ONLY COMPLETE - {num_run} run(s) at {hydra_output_dir}")

    elif cfg.only_plots:
        logger.info("=" * 80)
        logger.info("ONLY PLOTS MODE - Regenerating plots from existing evaluation results")

        # Check for multirun evaluation structure. Subdirs follow the pattern
        # ``run_<job_label>`` where job_label is either a numeric job index
        # (plain Hydra multirun) or an override_dirname like
        # ``dnabert2_multi_hg38_mean,model=encoder`` (Hydra sweep). Both must
        # be supported here; only_plots previously dropped the latter on the
        # floor because of an ``isdigit()`` filter.
        eval_subdirs = [
            d
            for d in os.listdir(run_dir)
            if os.path.isdir(os.path.join(run_dir, d))
            and d.startswith("run_")
            and os.path.exists(os.path.join(run_dir, d, "evaluation_results.json"))
        ]

        if len(eval_subdirs) > 0:
            logger.info(f"Detected {len(eval_subdirs)} multirun subdirectories")
            # Numeric job indices first (sorted by value), then non-numeric
            # names sorted lexicographically. Stable across re-runs.
            def _eval_subdir_sort_key(name: str) -> Tuple[int, int, str]:
                suffix = name[len("run_"):]
                if suffix.isdigit():
                    return (0, int(suffix), name)
                return (1, 0, name)

            eval_subdirs.sort(key=_eval_subdir_sort_key)

            results_list = []
            for subdir_name in eval_subdirs:
                subdir = os.path.join(run_dir, subdir_name)

                if cfg.aggregate_only:
                    # Load results directly from JSON without regenerating plots
                    results_path = os.path.join(subdir, "evaluation_results.json")
                    assert os.path.exists(results_path), f"Results file not found: {results_path}"
                    with open(results_path, "r") as f:
                        result = json.load(f)
                    result["output_dir"] = subdir
                    results_list.append(result)
                else:
                    logger.info(f"Re-plotting run {subdir_name}")

                    run_output_dir = os.path.join(hydra_output_dir, subdir_name)
                    os.makedirs(run_output_dir, exist_ok=True)

                    result = regenerate_plots_single_run(subdir, run_output_dir, logger)
                    if result:
                        results_list.append(result)

            # Create aggregate plots
            if results_list:
                logger.info("Creating aggregate comparison plots")
                aggregate_and_plot_multirun(results_list, hydra_output_dir, logger)
            else:
                logger.warning("No results found to aggregate")

        else:
            # Single run evaluation
            logger.info("Single run evaluation detected")
            regenerate_plots_single_run(run_dir, hydra_output_dir, logger)

        logger.info(f"Results saved to: {hydra_output_dir}")

    elif is_multirun_directory(run_dir):
        logger.info("=" * 80)
        logger.info("MULTIRUN DETECTED - Evaluating multiple models")

        # Get all subdirectories
        subdirs = get_multirun_subdirs(run_dir)
        logger.info(f"Found {len(subdirs)} runs to evaluate")

        # Evaluate each run. Multi-length runs are expanded over every requested
        # sequence length here, so a single eval directory ends up holding one
        # ``run_<job>_L<L>`` sub-run per (architecture, length) pair -- the same
        # flat layout the regular (single-length) sweep produces. This lets the
        # cross-length aggregator below run over one directory instead of the
        # caller fanning out into one eval directory per length.
        results_list = []
        for i, subdir in enumerate(subdirs):
            job_num = os.path.basename(subdir)
            eval_lengths = _resolve_eval_lengths(subdir, cfg.eval_seq_lengths)

            for eval_length in eval_lengths:
                length_suffix = "" if eval_length is None else f"_L{eval_length}"

                # Create output directory for this specific (run, length) pair.
                run_output_dir = os.path.join(
                    hydra_output_dir, f"run_{job_num}{length_suffix}"
                )

                # Resume: skip (run, length) pairs already evaluated by a prior
                # (possibly interrupted) invocation, loading their results so the
                # aggregate stays complete. Mirrors the identification-only path
                # so a restarted SLURM job fills only the missing lengths instead
                # of recomputing the whole eval dir or skipping it wholesale.
                results_path = os.path.join(run_output_dir, "evaluation_results.json")
                if os.path.exists(results_path):
                    logger.info(f"[eval] {results_path} exists, loading + skipping recompute")
                    with open(results_path) as f:
                        cached = json.load(f)
                    cached["output_dir"] = run_output_dir
                    results_list.append(cached)
                    continue

                logger.info(
                    f"Evaluating run {i+1}/{len(subdirs)} (job {job_num}) "
                    f"at seq_length={eval_length}"
                )

                result = evaluate_single_run(
                    subdir,
                    device,
                    run_output_dir,
                    logger,
                    cfg.identification,
                    cfg.max_samples,
                    eval_seq_length=eval_length,
                    ood_1000g=cfg.eval_ood_1000g,
                )
                # OOD runs return None for lengths whose 1000G file is missing.
                if result is not None:
                    results_list.append(result)

            logger.info(f"Completed evaluation for run {job_num}")

        # Create aggregate plots
        assert results_list, "No (run, length) pairs were evaluated; nothing to aggregate."
        logger.info("Creating aggregate comparison plots")
        aggregate_and_plot_multirun(results_list, hydra_output_dir, logger)

        logger.info("MULTIRUN EVALUATION COMPLETE")
        logger.info(f"Results saved to: {hydra_output_dir}")

    else:
        # Single run mode - original behavior
        logger.info("=" * 80)
        logger.info("SINGLE RUN MODE")
        logger.info("=" * 80)

        evaluate_single_run(
            run_dir,
            device,
            hydra_output_dir,
            logger,
            cfg.identification,
            cfg.max_samples,
            eval_seq_length=cfg.eval_seq_length,
            ood_1000g=cfg.eval_ood_1000g,
        )

        logger.info(f"\n{'='*80}")
        logger.info("EVALUATION COMPLETE")
        logger.info(f"Results saved to: {hydra_output_dir}")
        logger.info(f"{'='*80}")


if __name__ == "__main__":  # pragma: no cover
    main()  # type: ignore
