"""Cross-dataset comparison: one inversion model across Foundation Models.

Reads evaluation_results.json files from each FM's eval directory, filters to a
single inversion architecture (``--inversion-model``, e.g. ``encoder`` or
``query_decoder``), and produces publication-ready Levenshtein Similarity and
Accuracy vs Sequence Length plots comparing DNABERT-2, Evo 2, NTv2 + Random
Baseline.
"""

import argparse
import json
import os
import logging

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from src.plotting_utils import (
    configure_plot_style,
    get_fm_color,
    get_series_color,
    PLOT_STYLE,
    _get_model_properties_from_name,
    _save_fig,
)

matplotlib.use("Agg")
configure_plot_style()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# Foundation model display names and internal keys
FM_CONFIG = {
    "hyenadna_large": "HyenaDNA-1M",
    "caduceus": "Caduceus-PH",
    "dnabert1": "DNABERT-1",
    "grover": "GROVER",
    "genalm": "GENA-LM",
    "mistraldna": "Mistral-DNA",
    "seqlens": "seqLens",
    "aidodna": "AIDO.DNA",
    "ntv1": "NT-v1",
    "dnabert2": "DNABERT-2",
    "evo2": "Evo 2",
    "ntv2": "NTv2",
}


def _load_architecture_results(eval_dir: str, inversion_model: str):
    """Load all evaluation_results.json files from eval_dir/run_*/,
    filter to runs of ``inversion_model`` (e.g. "encoder", "query_decoder"),
    and return sorted (seq_length, mean, std, baseline_mean) tuples."""
    seq_lengths = []
    lev_means = []
    lev_stds = []
    acc_means = []
    acc_stds = []
    baseline_lev_means = []
    baseline_lev_stds = []
    baseline_acc_means = []

    # Scan run subdirectories
    run_dirs = [
        d
        for d in os.listdir(eval_dir)
        if os.path.isdir(os.path.join(eval_dir, d)) and d.startswith("run_")
    ]
    assert len(run_dirs) > 0, f"No run_* subdirectories found in {eval_dir}"

    for run_dir_name in run_dirs:
        results_path = os.path.join(eval_dir, run_dir_name, "evaluation_results.json")
        if not os.path.exists(results_path):
            continue

        with open(results_path, "r") as f:
            res = json.load(f)

        # Keep only runs of the requested inversion model.
        # Use the same parsing as plotting_utils.
        model_name = res.get("inversion_model", "")
        props = _get_model_properties_from_name(model_name)
        if props["im"] != inversion_model:
            continue

        seq_lengths.append(res["seq_length"])
        lev_means.append(res["levenshtein_mean"])
        lev_stds.append(res["levenshtein_std"])
        acc_means.append(res["accuracy_mean"])
        acc_stds.append(res["accuracy_std"])
        baseline_lev_means.append(res["baseline_levenshtein_mean"])
        baseline_lev_stds.append(res["baseline_levenshtein_std"])
        baseline_acc_means.append(res["baseline_accuracy_metrics"]["nucleotide_accuracy"])

    assert len(seq_lengths) > 0, f"No {inversion_model} results found in {eval_dir}"

    # Sort by sequence length
    sort_idx = np.argsort(seq_lengths)
    return {
        "seq_lengths": np.array(seq_lengths)[sort_idx],
        "lev_means": np.array(lev_means)[sort_idx],
        "lev_stds": np.array(lev_stds)[sort_idx],
        "acc_means": np.array(acc_means)[sort_idx],
        "acc_stds": np.array(acc_stds)[sort_idx],
        "baseline_lev_means": np.array(baseline_lev_means)[sort_idx],
        "baseline_lev_stds": np.array(baseline_lev_stds)[sort_idx],
        "baseline_acc_means": np.array(baseline_acc_means)[sort_idx],
    }


def _plot_metric(
    fm_data: dict,
    metric_key: str,
    std_key: str,
    ylabel: str,
    output_path: str,
    baseline_key: str = None,
    baseline_std_key: str = None,
):
    """Create a single metric vs sequence length plot."""
    plt.figure(figsize=PLOT_STYLE["figsize"])
    markers = ["o", "s", "^"]

    for idx, (fm_key, display_name) in enumerate(FM_CONFIG.items()):
        if fm_key not in fm_data:
            continue

        data = fm_data[fm_key]
        color = get_fm_color(fm_key)

        plt.plot(
            data["seq_lengths"],
            data[metric_key],
            marker=markers[idx % len(markers)],
            linestyle="-",
            linewidth=PLOT_STYLE["line_width"],
            markersize=PLOT_STYLE["marker_size"],
            label=display_name,
            color=color,
        )
        plt.fill_between(
            data["seq_lengths"],
            data[metric_key] - data[std_key],
            data[metric_key] + data[std_key],
            color=color,
            alpha=0.2,
        )

    # Random baseline — average across all FM datasets at common seq lengths
    if baseline_key:
        # Build per-FM lookup: seq_length -> baseline value
        fm_baseline_lookup = {}
        fm_baseline_std_lookup = {}
        all_seq_lengths_set = set()
        for fm_key in FM_CONFIG:
            if fm_key not in fm_data:
                continue
            data = fm_data[fm_key]
            lookup = dict(zip(data["seq_lengths"].tolist(), data[baseline_key].tolist()))
            fm_baseline_lookup[fm_key] = lookup
            all_seq_lengths_set.update(lookup.keys())
            if baseline_std_key:
                std_lookup = dict(zip(data["seq_lengths"].tolist(), data[baseline_std_key].tolist()))
                fm_baseline_std_lookup[fm_key] = std_lookup

        # For each seq length present in ANY FM, average the available baselines
        common_seq_lengths = sorted(all_seq_lengths_set)
        avg_baseline = []
        avg_baseline_std = []
        for sl in common_seq_lengths:
            vals = [fm_baseline_lookup[k][sl] for k in fm_baseline_lookup if sl in fm_baseline_lookup[k]]
            avg_baseline.append(np.mean(vals))
            if baseline_std_key and fm_baseline_std_lookup:
                std_vals = [fm_baseline_std_lookup[k][sl] for k in fm_baseline_std_lookup if sl in fm_baseline_std_lookup[k]]
                avg_baseline_std.append(np.mean(std_vals) if std_vals else 0.0)

        avg_baseline = np.array(avg_baseline)
        baseline_seq_lengths = np.array(common_seq_lengths)

        baseline_color = get_series_color("baseline")
        plt.plot(
            baseline_seq_lengths,
            avg_baseline,
            marker="x",
            linestyle="--",
            linewidth=PLOT_STYLE["line_width"],
            markersize=PLOT_STYLE["marker_size"],
            label="Random Baseline",
            color=baseline_color,
        )
        if baseline_std_key and len(avg_baseline_std) > 0:
            avg_baseline_std = np.array(avg_baseline_std)
            plt.fill_between(
                baseline_seq_lengths,
                avg_baseline - avg_baseline_std,
                avg_baseline + avg_baseline_std,
                color=baseline_color,
                alpha=0.15,
            )

    plt.xlabel("Sequence Length", fontsize=PLOT_STYLE["label_fontsize"])
    plt.ylabel(ylabel, fontsize=PLOT_STYLE["label_fontsize"])
    plt.ylim(0, 1.05)
    plt.legend(fontsize=PLOT_STYLE["legend_fontsize"], loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# Per-position accuracy helpers (per-token runs)
# ---------------------------------------------------------------------------


def _load_per_position_accuracy(eval_dir: str):
    """Load per-position accuracy arrays from a per-token eval directory's plot_data.pt."""
    pt_path = os.path.join(eval_dir, "plot_data.pt")
    assert os.path.isfile(pt_path), f"plot_data.pt not found in {eval_dir}"
    data = torch.load(pt_path, weights_only=False)
    return {
        "model": np.asarray(data["model_position_accuracy"]),
        "baseline": np.asarray(data["baseline_position_accuracy"]),
    }


def _plot_cross_dataset_per_position_accuracy(
    fm_pos_acc: dict,
    output_path: str,
):
    """Overlay per-position accuracy for all FMs + averaged random baseline."""
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize_wide"])
    markers = ["o", "s", "^"]

    for idx, (fm_key, display_name) in enumerate(FM_CONFIG.items()):
        if fm_key not in fm_pos_acc:
            continue
        acc = fm_pos_acc[fm_key]["model"]
        color = get_fm_color(fm_key)
        ax.plot(
            np.arange(len(acc)),
            acc,
            label=display_name,
            color=color,
            linewidth=PLOT_STYLE["line_width"],
        )

    # Random baseline — average across all FM datasets
    baselines = [fm_pos_acc[k]["baseline"] for k in FM_CONFIG if k in fm_pos_acc]
    assert len(baselines) > 0
    avg_baseline = np.mean(baselines, axis=0)
    ax.plot(
        np.arange(len(avg_baseline)),
        avg_baseline,
        label="Random Baseline",
        color=get_series_color("baseline"),
        linestyle="--",
        linewidth=PLOT_STYLE["line_width"],
    )

    ax.set_xlabel("Position in Sequence", fontsize=PLOT_STYLE["label_fontsize"])
    ax.set_ylabel("Accuracy", fontsize=PLOT_STYLE["label_fontsize"])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=PLOT_STYLE["legend_fontsize"], loc="center right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(output_path)
    logger.info(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# Normalised Euclidean Distance helpers
# ---------------------------------------------------------------------------


def _load_ned_data(analysis_dir: str):
    """Load NED distributions from an analysis sweep directory.

    Walks the sweep recursively (Hydra sweep subdirs may be nested one level
    deeper when ``hydra.job.override_dirname`` contains a ``/``, e.g.
    ``data=hg38/...``), reading ``stats.json`` for the sequence length and
    ``norm_euclidean_dists.npy`` for the raw distance vector from every run
    directory that has both.

    Returns a dict mapping ``seq_length -> np.ndarray``.
    """
    ned_by_seqlen = {}
    for dirpath, _dirnames, filenames in os.walk(analysis_dir):
        if "stats.json" not in filenames or "norm_euclidean_dists.npy" not in filenames:
            continue
        with open(os.path.join(dirpath, "stats.json"), "r") as f:
            st = json.load(f)
        seq_length = int(st["avg_seq_len"])
        ned_by_seqlen[seq_length] = np.load(os.path.join(dirpath, "norm_euclidean_dists.npy"))
    assert len(ned_by_seqlen) > 0, f"No NED .npy files found in {analysis_dir}"
    return ned_by_seqlen


def _plot_merged_ned(
    fm_ned: dict,
    seq_length: int,
    output_path: str,
):
    """Overlay NED histograms for all FMs at a given sequence length."""
    x_max = 0.4
    shared_bins = np.linspace(0, x_max, 201)

    plt.figure(figsize=PLOT_STYLE["figsize"])

    for fm_key, display_name in FM_CONFIG.items():
        if fm_key not in fm_ned:
            continue
        ned_by_sl = fm_ned[fm_key]
        if seq_length not in ned_by_sl:
            continue
        color = get_fm_color(fm_key)
        sns.histplot(
            ned_by_sl[seq_length],
            bins=shared_bins,
            kde=True,
            color=color,
            label=display_name,
            alpha=0.5,
            element="step",
        )

    plt.xlabel("Normalized Euclidean Distance (d / √dim)", fontsize=PLOT_STYLE["label_fontsize"])
    plt.ylabel("Count", fontsize=PLOT_STYLE["label_fontsize"])
    plt.xlim(0, x_max)
    plt.legend(fontsize=PLOT_STYLE["legend_fontsize"], loc="center right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(output_path, bbox=False)
    logger.info(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset inversion-model comparison plots.")
    parser.add_argument(
        "--eval-dirs",
        nargs="+",
        required=True,
        help="Eval output directories, one per FM (e.g. outputs/eval_mean_dnabert2 ...)",
    )
    parser.add_argument(
        "--inversion-model",
        default="encoder",
        help="Inversion architecture to compare across FMs (e.g. encoder, "
        "query_decoder, decoder). Used to filter run_*/ subdirs and to prefix "
        "the output plot filenames.",
    )
    parser.add_argument(
        "--analysis-dirs",
        nargs="+",
        default=None,
        help="Analysis output directories, one per FM (e.g. outputs/analysis_mean_dnabert2 ...)",
    )
    parser.add_argument(
        "--per-token-eval-dirs",
        nargs="+",
        default=None,
        help="Per-token eval directories, one per FM "
        "(e.g. outputs/eval_per_token_dnabert2_100_hg38_per_token ...)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/cross_dataset_comparison",
        help="Directory to save comparison plots",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Per-position accuracy (per-token runs)
    # ------------------------------------------------------------------
    if args.per_token_eval_dirs:
        fm_pos_acc = {}
        for pt_dir in args.per_token_eval_dirs:
            assert os.path.isdir(pt_dir), f"Directory not found: {pt_dir}"
            dirname = os.path.basename(pt_dir.rstrip("/\\"))
            fm_key = None
            for key in FM_CONFIG:
                if key in dirname:
                    fm_key = key
                    break
            assert fm_key is not None, (
                f"Could not detect foundation model from directory name: {dirname}. "
                f"Expected one of: {list(FM_CONFIG.keys())}"
            )
            logger.info(f"Loading per-position accuracy for {FM_CONFIG[fm_key]} from {pt_dir}")
            fm_pos_acc[fm_key] = _load_per_position_accuracy(pt_dir)

        _plot_cross_dataset_per_position_accuracy(
            fm_pos_acc,
            output_path=os.path.join(args.output_dir, "per_position_accuracy"),
        )

    # Load encoder results from each eval directory
    fm_data = {}
    for eval_dir in args.eval_dirs:
        assert os.path.isdir(eval_dir), f"Directory not found: {eval_dir}"

        # Detect FM from directory name (outputs/eval_mean_dnabert2 -> dnabert2)
        dirname = os.path.basename(eval_dir.rstrip("/\\"))
        fm_key = None
        for key in FM_CONFIG:
            if key in dirname:
                fm_key = key
                break
        assert fm_key is not None, (
            f"Could not detect foundation model from directory name: {dirname}. "
            f"Expected one of: {list(FM_CONFIG.keys())}"
        )

        logger.info(
            f"Loading {args.inversion_model} results for {FM_CONFIG[fm_key]} from {eval_dir}"
        )
        fm_data[fm_key] = _load_architecture_results(eval_dir, args.inversion_model)
        logger.info(
            f"  Found {len(fm_data[fm_key]['seq_lengths'])} sequence lengths: "
            f"{fm_data[fm_key]['seq_lengths'].tolist()}"
        )

    assert len(fm_data) > 0, "No foundation model results loaded"

    # Plot 1: Levenshtein Similarity vs Sequence Length
    _plot_metric(
        fm_data,
        metric_key="lev_means",
        std_key="lev_stds",
        ylabel="Levenshtein Similarity",
        output_path=os.path.join(
            args.output_dir, f"{args.inversion_model}_levenshtein_vs_seqlen"
        ),
        baseline_key="baseline_lev_means",
        baseline_std_key="baseline_lev_stds",
    )

    # Plot 2: Accuracy vs Sequence Length
    _plot_metric(
        fm_data,
        metric_key="acc_means",
        std_key="acc_stds",
        ylabel="Nucleotide Accuracy",
        output_path=os.path.join(
            args.output_dir, f"{args.inversion_model}_accuracy_vs_seqlen"
        ),
        baseline_key="baseline_acc_means",
    )

    # ------------------------------------------------------------------
    # Merged NED plots (one per sequence length, all FMs overlaid)
    # ------------------------------------------------------------------
    if args.analysis_dirs:
        fm_ned = {}
        for analysis_dir in args.analysis_dirs:
            assert os.path.isdir(analysis_dir), f"Directory not found: {analysis_dir}"
            dirname = os.path.basename(analysis_dir.rstrip("/\\"))
            fm_key = None
            for key in FM_CONFIG:
                if key in dirname:
                    fm_key = key
                    break
            assert fm_key is not None, (
                f"Could not detect foundation model from directory name: {dirname}. "
                f"Expected one of: {list(FM_CONFIG.keys())}"
            )
            logger.info(f"Loading NED data for {FM_CONFIG[fm_key]} from {analysis_dir}")
            fm_ned[fm_key] = _load_ned_data(analysis_dir)

        # Determine common sequence lengths
        all_seq_lengths = set()
        for ned_by_sl in fm_ned.values():
            all_seq_lengths.update(ned_by_sl.keys())
        common_seq_lengths = sorted(all_seq_lengths)

        ned_dir = os.path.join(args.output_dir, "ned_merged")
        os.makedirs(ned_dir, exist_ok=True)

        for sl in common_seq_lengths:
            _plot_merged_ned(
                fm_ned,
                seq_length=sl,
                output_path=os.path.join(ned_dir, f"seqlen_{sl}"),
            )

        logger.info(f"NED merged plots saved to {ned_dir}")

    logger.info(f"All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
