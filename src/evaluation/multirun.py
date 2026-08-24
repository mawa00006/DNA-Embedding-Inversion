"""
Multirun evaluation utilities.
"""

import os
import re
import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import save_json, NUCLEOTIDES
from src.plotting_utils import (
    plot_aggregate_metrics,
    get_series_colors,
    get_series_color,
    get_model_display_name,
    PLOT_STYLE,
    plot_similarity_ridge,
)


def _walk_run_subdirs(root: str, require_model: bool) -> List[str]:
    """Recursively find descendants of `root` containing .hydra/config.yaml.

    Stops descending once a run directory is found so we don't recurse into
    a run's own internal subdirectories. Supports sweep layouts where
    ``hydra.job.override_dirname`` contains a ``/`` (e.g. ``data=hg38/...``),
    which Hydra renders as an intermediate directory level.
    """
    if not os.path.isdir(root):
        return []

    found: List[str] = []
    for dirpath, dirnames, _ in os.walk(root):
        if dirpath == root:
            continue
        config_path = os.path.join(dirpath, ".hydra", "config.yaml")
        if not os.path.exists(config_path):
            continue
        if require_model and not os.path.exists(os.path.join(dirpath, "model.pt")):
            continue
        found.append(dirpath)
        dirnames.clear()  # don't descend into a run's own subdirs
    return found


def is_multirun_directory(run_dir: str) -> bool:
    """Check if a directory is a Hydra multirun directory.

    A multirun directory contains (possibly nested) subdirectories
    each with their own .hydra/config.yaml and model.pt.

    Parameters
    ----------
    run_dir : str
        Path to the directory to check.

    Returns
    -------
    bool
        True if multirun directory, False otherwise.
    """
    return len(_walk_run_subdirs(run_dir, require_model=True)) > 0


def get_multirun_subdirs(multirun_dir: str) -> List[str]:
    """Get all subdirectories that look like Hydra runs from a multirun directory.

    Parameters
    ----------
    multirun_dir : str
        Path to the multirun directory.

    Returns
    -------
    List[str]
        Sorted list of full paths to subdirectories.
    """
    subdirs = _walk_run_subdirs(multirun_dir, require_model=False)

    # Sort logic: try to extract the trailing job number (e.g. ..._0, ..._1)
    # If not found, fall back to alphabetical
    def sort_key(path: str) -> Any:
        base = os.path.basename(path)
        match = re.search(r"_(\d+)$", base)
        if match:
            # Sort by number, then by name (though number should be unique per prefix usually)
            return (0, int(match.group(1)), base)
        elif base.isdigit():
            # Old style just numbers
            return (0, int(base), base)
        else:
            # No number suffix found, put at end, alphabetical
            return (1, 0, base)

    subdirs.sort(key=sort_key)
    return subdirs


def aggregate_and_plot_multirun(
    results_list: List[Dict[str, Any]], output_dir: str, logger: logging.Logger
) -> None:
    """Create aggregate plots comparing multiple runs.

    Parameters
    ----------
    results_list : List[Dict[str, Any]]
        List of results from evaluate_single_run.
    output_dir : str
        Directory to save aggregate plots.
    logger : logging.Logger
        Logger instance.
    """
    logger.info("Creating aggregate comparison plots...")

    # Create aggregate output directory
    aggregate_dir = os.path.join(output_dir, "aggregate")
    os.makedirs(aggregate_dir, exist_ok=True)

    # Group results by model name
    grouped_by_model = defaultdict(list)

    for result in results_list:
        inv = result["inversion_model"]
        found = result["foundation_model"]
        if found != "unknown" and found not in inv:
            model_name = f"{inv} ({found})"
        else:
            model_name = inv

        grouped_by_model[model_name].append(result)

    logger.info(f"Found {len(grouped_by_model)} different models: {list(grouped_by_model.keys())}")

    # Prepare data structures for multi-model plotting
    model_data = {}
    baseline_data_by_len = defaultdict(list)

    for model_name, model_results in grouped_by_model.items():
        seq_lengths = []
        levenshtein_means = []
        levenshtein_stds = []
        accuracy_means = []
        accuracy_stds = []
        times_per_sequence = []
        nucleotide_freqs = {nuc: [] for nuc in NUCLEOTIDES}

        for result in model_results:
            seq_len = result["seq_length"]
            seq_lengths.append(seq_len)
            levenshtein_means.append(result["levenshtein_mean"])
            levenshtein_stds.append(result["levenshtein_std"])

            # Direct key access - crash fast if data is missing
            accuracy_means.append(result["accuracy_mean"])
            accuracy_stds.append(result["accuracy_std"])

            if "time_per_sequence" in result:
                times_per_sequence.append(result["time_per_sequence"])
            else:
                times_per_sequence.append(0.0)

            # Collect baseline data
            baseline_data_by_len[seq_len].append(
                {
                    "levenshtein_mean": result["baseline_levenshtein_mean"],
                    "levenshtein_std": result["baseline_levenshtein_std"],
                    "accuracy": result["baseline_accuracy_metrics"]["nucleotide_accuracy"],
                }
            )

            # Extract predicted nucleotide frequencies
            pred_freqs = result["nucleotide_frequencies"]["pred"]
            for nuc in NUCLEOTIDES:
                nucleotide_freqs[nuc].append(pred_freqs[nuc])

        # Sort by sequence length
        sorted_indices = np.argsort(seq_lengths)
        model_data[model_name] = {
            "seq_lengths": [seq_lengths[i] for i in sorted_indices],
            "levenshtein_means": [levenshtein_means[i] for i in sorted_indices],
            "levenshtein_stds": [levenshtein_stds[i] for i in sorted_indices],
            "accuracy_means": [accuracy_means[i] for i in sorted_indices],
            "accuracy_stds": [accuracy_stds[i] for i in sorted_indices],
            "times_per_sequence": [times_per_sequence[i] for i in sorted_indices],
            "nucleotide_freqs": {
                nuc: [nucleotide_freqs[nuc][i] for i in sorted_indices] for nuc in NUCLEOTIDES
            },
        }

    # Process baseline data
    baseline_seq_lengths = sorted(baseline_data_by_len.keys())
    baseline_lev_means = []
    baseline_lev_stds = []
    baseline_acc_means = []

    for seq_len in baseline_seq_lengths:
        items = baseline_data_by_len[seq_len]
        baseline_lev_means.append(np.mean([x["levenshtein_mean"] for x in items]))
        baseline_lev_stds.append(np.mean([x["levenshtein_std"] for x in items]))
        baseline_acc_means.append(np.mean([x["accuracy"] for x in items]))

    # Extract data for backwards compatibility (use first model)
    first_model = list(model_data.keys())[0] if model_data else "unknown"
    seq_lengths = model_data[first_model]["seq_lengths"] if model_data else []
    levenshtein_means = model_data[first_model]["levenshtein_means"] if model_data else []
    levenshtein_stds = model_data[first_model]["levenshtein_stds"] if model_data else []
    accuracy_means = model_data[first_model]["accuracy_means"] if model_data else []

    # Compute expected (true) nucleotide frequencies per sequence length
    # Assume dataset-constant frequencies: take from the first occurrence per sequence length
    true_freqs_by_len: Dict[int, Dict[str, float]] = {}
    for res in results_list:
        seq_len = res["seq_length"]
        if seq_len not in true_freqs_by_len:
            true_freqs_by_len[seq_len] = res["nucleotide_frequencies"]["true"]

    # Build expected nucleotide frequency lists aligned with `seq_lengths`
    expected_nucleotide_freqs = {nuc: [] for nuc in NUCLEOTIDES}
    for seq_len in seq_lengths:
        for nuc in NUCLEOTIDES:
            if seq_len in true_freqs_by_len:
                expected_nucleotide_freqs[nuc].append(true_freqs_by_len[seq_len][nuc])

    # Plot Levenshtein similarity vs sequence length (multi-model)
    plot_aggregate_metrics(
        model_data,
        os.path.join(aggregate_dir, "levenshtein_vs_seqlen"),
        y_key_mean="levenshtein_means",
        y_key_std="levenshtein_stds",
        ylabel="Levenshtein Similarity",
        title="",
        baseline_data={
            "seq_lengths": baseline_seq_lengths,
            "means": baseline_lev_means,
            "stds": baseline_lev_stds,
        },
    )
    logger.info("Saved Levenshtein vs sequence length plot")

    # Plot overall accuracy vs sequence length (multi-model)
    plot_aggregate_metrics(
        model_data,
        os.path.join(aggregate_dir, "accuracy_vs_seqlen"),
        y_key_mean="accuracy_means",
        y_key_std="accuracy_stds",
        ylabel="Nucleotide Accuracy",
        title="",
        baseline_data={
            "seq_lengths": baseline_seq_lengths,
            "means": baseline_acc_means,
            "stds": [0.0] * len(baseline_acc_means),
        },
    )
    logger.info("Saved accuracy vs sequence length plot")

    # Define markers and color map for manual plots
    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]
    model_color_map = get_series_colors(list(model_data.keys()))

    # Plot inference time vs sequence length (multi-model)
    plt.figure(figsize=(10, 6))

    for idx, (model_name, data) in enumerate(model_data.items()):
        clean_name = get_model_display_name(model_name)
        plt.plot(
            data["seq_lengths"],
            data["times_per_sequence"],
            marker=markers[idx % len(markers)],
            linestyle="-",
            linewidth=PLOT_STYLE["line_width"],
            markersize=PLOT_STYLE["marker_size"],
            label=clean_name,
            color=model_color_map[model_name],
        )

    plt.xlabel("Sequence Length", fontsize=PLOT_STYLE["label_fontsize"])
    plt.ylabel("Time per Sequence (s)", fontsize=PLOT_STYLE["label_fontsize"])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=PLOT_STYLE["legend_fontsize"], loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(aggregate_dir, "inference_time_vs_seqlen.png"), dpi=300)
    plt.savefig(os.path.join(aggregate_dir, "inference_time_vs_seqlen.pdf"), dpi=300)
    plt.close()
    logger.info("Saved inference time vs sequence length plot")

    # Plot nucleotide frequency vs sequence length (5 subplots, one per nucleotide)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for nuc_idx, nuc in enumerate(NUCLEOTIDES):
        ax = axes[nuc_idx]

        for idx, (model_name, data) in enumerate(model_data.items()):
            clean_name = get_model_display_name(model_name)
            ax.plot(
                data["seq_lengths"],
                data["nucleotide_freqs"][nuc],
                marker=markers[idx % len(markers)],
                linestyle="-",
                linewidth=PLOT_STYLE["line_width"],
                markersize=PLOT_STYLE["marker_size"],
                label=clean_name,
                color=model_color_map[model_name],
            )

        # Plot expected (true) nucleotide content aggregated across runs
        if expected_nucleotide_freqs and nuc in expected_nucleotide_freqs:
            ax.plot(
                seq_lengths,
                expected_nucleotide_freqs[nuc],
                marker=None,
                linestyle="--",
                linewidth=PLOT_STYLE["line_width"],
                color=get_series_color("expected"),
                label="Expected (True)",
            )

        ax.set_xlabel("Sequence Length", fontsize=PLOT_STYLE["label_fontsize"])
        ax.set_ylabel(f"Frequency of {nuc}", fontsize=PLOT_STYLE["label_fontsize"])
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=PLOT_STYLE["legend_fontsize"], loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(aggregate_dir, "nucleotide_frequency_vs_seqlen.png"), dpi=300)
    plt.savefig(os.path.join(aggregate_dir, "nucleotide_frequency_vs_seqlen.pdf"), dpi=300)
    plt.close()
    logger.info("Saved nucleotide frequency vs sequence length plot")

    # Generate ridge plot showing similarity progression across sequence lengths
    # Need to load raw similarity data from plot_data.pt files
    model_similarities_by_length: Dict[int, List[float]] = {}
    baseline_similarities_by_length: Dict[int, List[float]] = {}

    for result in results_list:
        seq_len = result["seq_length"]
        # Use output_dir if available (new format), fallback to run_dir (for regeneration)
        eval_output_dir = result.get("output_dir", result["run_dir"])

        # Try to find plot_data.pt in the evaluation output directory
        plot_data_path = os.path.join(eval_output_dir, "plot_data.pt")
        if os.path.exists(plot_data_path):
            plot_data = torch.load(plot_data_path, weights_only=False)
            if seq_len not in model_similarities_by_length:
                model_similarities_by_length[seq_len] = []
                baseline_similarities_by_length[seq_len] = []

            model_similarities_by_length[seq_len].extend(plot_data["levenshtein_similarities"])
            if plot_data.get("baseline_levenshtein") is not None:
                baseline_similarities_by_length[seq_len].extend(plot_data["baseline_levenshtein"])

    # Generate ridge plot if we have data for multiple sequence lengths
    if len(model_similarities_by_length) > 1:
        first_model = list(model_data.keys())[0] if model_data else "unknown"
        model_color = get_series_color(first_model)
        plot_similarity_ridge(
            model_similarities_by_length,
            baseline_similarities_by_length,
            os.path.join(aggregate_dir, "similarity_ridge"),
            model_color=model_color,
        )
        logger.info("Saved similarity ridge plot")
    else:
        logger.info("Skipping ridge plot (need data from multiple sequence lengths)")

    # Aggregate data for difficulty metrics plots
    # We need: {model_name: {'metric': [...], 'similarity': [...]}}
    entropy_data = defaultdict(lambda: {"metric": [], "similarity": []})
    repetitiveness_data = defaultdict(lambda: {"metric": [], "similarity": []})

    has_new_metrics = False

    for result in results_list:
        inv = result["inversion_model"]
        found = result["foundation_model"]
        if found != "unknown" and found not in inv:
            model_name = f"{inv} ({found})"
        else:
            model_name = inv

        eval_output_dir = result.get("output_dir", result["run_dir"])
        plot_data_path = os.path.join(eval_output_dir, "plot_data.pt")

        if os.path.exists(plot_data_path):
            try:
                plot_data = torch.load(plot_data_path, weights_only=False)
                if "shannon_entropies" in plot_data and "repetitiveness_scores" in plot_data:
                    has_new_metrics = True
                    levs = plot_data["levenshtein_similarities"]
                    ents = plot_data["shannon_entropies"]
                    reps = plot_data["repetitiveness_scores"]

                    if len(levs) == len(ents) == len(reps):
                        entropy_data[model_name]["metric"].extend(ents)
                        entropy_data[model_name]["similarity"].extend(levs)

                        repetitiveness_data[model_name]["metric"].extend(reps)
                        repetitiveness_data[model_name]["similarity"].extend(levs)
            except Exception as e:
                logger.warning(f"Could not load detailed plot data for {model_name}: {e}")

    if has_new_metrics:
        from src.plotting_utils import plot_aggregate_metric_vs_similarity

        plot_aggregate_metric_vs_similarity(
            entropy_data,
            "Shannon Entropy (Bits)",
            os.path.join(aggregate_dir, "aggregate_entropy_regression"),
        )
        logger.info("Saved aggregate entropy regression plot")

        plot_aggregate_metric_vs_similarity(
            repetitiveness_data,
            "4-mer Redundancy",
            os.path.join(aggregate_dir, "aggregate_repetitiveness_regression"),
        )
        logger.info("Saved aggregate repetitiveness regression plot")

    # Save aggregate comparison data
    comparison_data = {
        "sequence_lengths": seq_lengths,
        "levenshtein_mean": levenshtein_means,
        "levenshtein_std": levenshtein_stds,
        "overall_accuracy": accuracy_means,
        "num_runs": len(results_list),
        "models": list(model_data.keys()),
        "model_data": {
            model_name: {
                "seq_lengths": data["seq_lengths"],
                "levenshtein_means": data["levenshtein_means"],
                "accuracy_means": data["accuracy_means"],
                "times_per_sequence": data["times_per_sequence"],
                "nucleotide_freqs": data["nucleotide_freqs"],
            }
            for model_name, data in model_data.items()
        },
    }
    save_json(comparison_data, os.path.join(aggregate_dir, "comparison_metrics.json"))
    logger.info(f"Saved aggregate comparison data to {aggregate_dir}")
