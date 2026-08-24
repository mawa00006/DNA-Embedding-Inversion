"""Plotting utilities: styles, colors, and reusable plot functions."""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import seaborn as sns
import numpy as np
import scipy.stats as stats

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
PLOT_STYLE = {
    "label_fontsize": 15,
    "title_fontsize": 17,
    "tick_fontsize": 12,
    "legend_fontsize": 12,
    "annotation_fontsize": 12,
    "line_width": 2.5,
    "marker_size": 8,
    "figsize": (7, 5),
    "figsize_wide": (9, 5),
    "figsize_square": (6, 6),
}

NUCLEOTIDES = ["A", "C", "G", "T"]

_MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p"]

# ---------------------------------------------------------------------------
# Color palettes — FM determines hue, inversion model determines shade
# ---------------------------------------------------------------------------
_COLORBLIND_PALETTE = [mcolors.to_hex(c) for c in sns.color_palette("colorblind", 10)]

_FM_PALETTES = {
    "dnabert": sns.color_palette("Greens_r", 10).as_hex(),
    "ntv2": sns.color_palette("Oranges_r", 10).as_hex(),
    "evo2": sns.color_palette("Blues_r", 10).as_hex(),
    "other": sns.color_palette("Purples_r", 10).as_hex(),
}

_INVERSION_MODEL_INDEX = {
    "encoder": 0,
    "resnet": 1,
    "mlp": 2,
    "decoder": 3,
    "query_decoder": 4,
    "knn": 5,
    "linear": 6,
    "mask_predict": 7,
}

INVERSION_MODEL_DISPLAY_NAMES = {
    "encoder": "Encoder",
    "resnet": "ResNet",
    "mlp": "MLP",
    "decoder": "Decoder",
    "query_decoder": "Query Decoder",
    "knn": "Nearest Neighbor",
    "linear": "Linear",
    "mask_predict": "Mask-Predict",
}

# Keyword --> special color (checked before FM/IM parsing)
_SPECIAL_COLORS = {
    "random": "#555555",
    "baseline": "#555555",
    "true": "#2ca02c",
    "predicted": "#ff7f0e",
    "expected": "#000000",
    "median": "#9467bd",
}

# Vivid primary colors for single-FM plots (one line per FM)
_FM_PRIMARY_COLORS = {
    "dnabert": "#2ecc71",  # vivid green
    "evo2": "#3498db",  # vivid blue
    "ntv2": "#e67e22",  # vivid orange
    "other": "#9b59b6",  # vivid purple
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _save_fig(path: str, bbox: bool = True) -> None:
    """Save current figure as .png + .pdf and close."""
    kw = {"dpi": 300}
    if bbox:
        kw["bbox_inches"] = "tight"
    plt.savefig(f"{path}.png", **kw)
    plt.savefig(f"{path}.pdf", **kw)
    plt.close()


def _get_model_properties_from_name(name: str) -> Dict[str, str]:
    """Parse ``"InversionModel (FoundationModel)"`` into ``{"fm": ..., "im": ...}``."""
    normalized = name.lower()

    match = re.search(r"\((.+?)\)", normalized)
    if match:
        input_model = match.group(1).strip()
        inversion_part = normalized.replace(f"({input_model})", "").strip()
    else:
        input_model = ""
        inversion_part = normalized

    combined = input_model + " " + inversion_part

    # Foundation model
    if "dnabert" in combined:
        fm = "dnabert"
    elif "ntv2" in combined or "nucleotide_transformer" in combined:
        fm = "ntv2"
    elif "evo" in combined:
        fm = "evo2"
    else:
        fm = "other"

    # Inversion model. ``query_decoder`` must be checked before ``decoder``
    # because "decoder" is a substring of "query_decoder" -- otherwise both
    # collapse to the same label/colour in multi-model plots.
    im = "other"
    for key in ("encoder", "resnet", "mlp", "knn", "mask_predict", "query_decoder", "decoder", "linear"):
        if key in inversion_part:
            im = key
            break

    return {"fm": fm, "im": im}


# ---------------------------------------------------------------------------
# Public colour / label API
# ---------------------------------------------------------------------------
def get_model_display_name(model_name: str) -> str:
    """Return the display label for *model_name* (used in plot legends)."""
    im = _get_model_properties_from_name(model_name)["im"]
    return INVERSION_MODEL_DISPLAY_NAMES.get(im, im)


def configure_plot_style() -> None:
    """Configure Matplotlib/Seaborn for publication-ready plots."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.titlesize": PLOT_STYLE["title_fontsize"],
            "axes.labelsize": PLOT_STYLE["label_fontsize"],
            "xtick.labelsize": PLOT_STYLE["tick_fontsize"],
            "ytick.labelsize": PLOT_STYLE["tick_fontsize"],
            "legend.fontsize": PLOT_STYLE["legend_fontsize"],
            "axes.linewidth": 1.1,
            "grid.alpha": 0.3,
            "lines.linewidth": PLOT_STYLE["line_width"],
            # Font fallback because HPC environment has strange fonts
            "font.family": "sans-serif",
            "font.sans-serif": [
                "DejaVu Sans",
                "Liberation Sans",
                "FreeSans",
                "Bitstream Vera Sans",
                "sans-serif",
            ],
        }
    )


def get_series_color(name: str) -> str:
    """Return a colour for *name* (FM --> hue, inversion model --> shade)."""
    name_lower = name.lower()

    # Special-case keywords
    for keyword, color in _SPECIAL_COLORS.items():
        if keyword in name_lower:
            return color
    if name_lower in ("mean", "average"):
        return "#d62728"

    props = _get_model_properties_from_name(name)
    palette = _FM_PALETTES.get(props["fm"], _FM_PALETTES["other"])
    idx = _INVERSION_MODEL_INDEX.get(props["im"], 2)
    return palette[min(idx, len(palette) - 1)]


def get_fm_color(name: str) -> str:
    """Return a vivid primary colour for the foundation model in *name*.

    Use this for single-FM plots (one model line + baseline).
    Multi-model comparison plots should keep ``get_series_color`` for shade
    differentiation between inversion models.
    """
    props = _get_model_properties_from_name(name)
    return _FM_PRIMARY_COLORS[props["fm"]]


def get_series_colors(names: List[str]) -> Dict[str, str]:
    """Return a colour mapping, resolving collisions with ``husl``."""
    mapping = {n: get_series_color(n) for n in names}

    # Group by colour to detect collisions
    from collections import defaultdict

    by_color = defaultdict(list)
    for n, c in mapping.items():
        by_color[c].append(n)

    for color, collisions in by_color.items():
        if len(collisions) > 1:
            collisions.sort()
            for i, n in enumerate(collisions):
                mapping[n] = sns.color_palette("husl", len(collisions)).as_hex()[i]

    return mapping


# ---------------------------------------------------------------------------
# Generic distribution plot
# ---------------------------------------------------------------------------
def plot_distribution(
    data_or_dict,
    save_path,
    bins=50,
    kde=True,
    color=None,
    xlabel="Value",
    ylabel="Count",
    alpha=0.6,
    xlim=None,
    ylim=None,
):
    """Plot histogram(s). Pass a dict ``{label: values}`` for overlapping distributions.

    Parameters
    ----------
    xlim, ylim : tuple[float, float] | None
        If given, fix the x / y axis to ``(low, high)``.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    configure_plot_style()
    plt.figure(figsize=PLOT_STYLE["figsize"])

    if isinstance(data_or_dict, dict):
        for label, data in data_or_dict.items():
            sns.histplot(
                np.asarray(data),
                bins=bins,
                kde=kde,
                color=get_series_color(label),
                label=label,
                alpha=alpha,
                element="step",
            )
        plt.legend(fontsize=PLOT_STYLE["legend_fontsize"], loc="upper right")
    else:
        sns.histplot(
            np.asarray(data_or_dict),
            bins=bins,
            kde=kde,
            color=color or "#2b8cbe",
            alpha=alpha,
        )

    plt.xlabel(xlabel, fontsize=PLOT_STYLE["label_fontsize"])
    plt.ylabel(ylabel, fontsize=PLOT_STYLE["label_fontsize"])
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()
    _save_fig(save_path, bbox=False)


# ---------------------------------------------------------------------------
# Aggregate metric vs sequence length
# ---------------------------------------------------------------------------
def plot_aggregate_metrics(
    model_data: Dict[str, Dict[str, List[float]]],
    output_path: str,
    y_key_mean: str,
    y_key_std: str,
    ylabel: str,
    title: str,
    baseline_data: Optional[Dict[str, List[float]]] = None,
):
    """Line plot of metric ± std vs sequence length for multiple models."""
    plt.figure(figsize=PLOT_STYLE["figsize"])

    colors = get_series_colors(list(model_data.keys()))

    for idx, (model_name, data) in enumerate(model_data.items()):
        seq_lengths = np.array(data["seq_lengths"])
        means = np.array(data[y_key_mean])
        stds = np.array(data[y_key_std])

        order = np.argsort(seq_lengths)
        seq_lengths, means, stds = seq_lengths[order], means[order], stds[order]

        color = colors[model_name]
        plt.plot(
            seq_lengths,
            means,
            marker=_MARKERS[idx % len(_MARKERS)],
            linestyle="-",
            linewidth=PLOT_STYLE["line_width"],
            markersize=PLOT_STYLE["marker_size"],
            label=get_model_display_name(model_name),
            color=color,
        )
        plt.plot(seq_lengths, means - stds, linestyle="--", linewidth=1.0, color=color, alpha=0.5)
        plt.plot(seq_lengths, means + stds, linestyle="--", linewidth=1.0, color=color, alpha=0.5)

    if baseline_data:
        b_seqs = np.array(baseline_data["seq_lengths"])
        b_means = np.array(baseline_data["means"])
        b_stds = np.array(baseline_data["stds"])
        order = np.argsort(b_seqs)
        b_seqs, b_means, b_stds = b_seqs[order], b_means[order], b_stds[order]

        color = get_series_color("random baseline")
        plt.plot(
            b_seqs,
            b_means,
            marker="x",
            linestyle="--",
            linewidth=PLOT_STYLE["line_width"],
            markersize=PLOT_STYLE["marker_size"],
            label="Random Baseline",
            color=color,
        )
        plt.plot(b_seqs, b_means - b_stds, linestyle="--", linewidth=1.0, color=color, alpha=0.5)
        plt.plot(b_seqs, b_means + b_stds, linestyle="--", linewidth=1.0, color=color, alpha=0.5)

    plt.xlabel("Sequence Length", fontsize=PLOT_STYLE["label_fontsize"])
    plt.ylabel(ylabel, fontsize=PLOT_STYLE["label_fontsize"])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=PLOT_STYLE["legend_fontsize"], loc="upper right")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    _save_fig(output_path, bbox=False)


# ---------------------------------------------------------------------------
# Single-model bar / line plots
# ---------------------------------------------------------------------------
def plot_nucleotide_frequencies(
    freq_dict: dict,
    output_path: str,
    title: str = "",
    model_color: str = "#1f77b4",
):
    """Grouped bar chart of true vs predicted nucleotide frequencies."""
    true_freqs = [freq_dict["true"][n] for n in NUCLEOTIDES]
    pred_freqs = [freq_dict["pred"][n] for n in NUCLEOTIDES]

    x = np.arange(len(NUCLEOTIDES))
    w = 0.35

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize_square"])
    ax.bar(
        x - w / 2,
        true_freqs,
        w,
        label="True (Reference)",
        alpha=0.85,
        color="gray",
        edgecolor="black",
        linewidth=1,
    )
    ax.bar(
        x + w / 2,
        pred_freqs,
        w,
        label="Predicted (Model)",
        alpha=0.85,
        color=model_color,
        edgecolor="black",
        linewidth=1,
    )

    ax.set_xlabel("Nucleotide", fontsize=PLOT_STYLE["label_fontsize"])
    ax.set_ylabel("Frequency", fontsize=PLOT_STYLE["label_fontsize"])
    ax.set_xticks(x)
    ax.set_xticklabels(NUCLEOTIDES, fontsize=PLOT_STYLE["tick_fontsize"])
    ax.legend(fontsize=PLOT_STYLE["legend_fontsize"], loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    _save_fig(output_path)


def plot_accuracy_metrics(metrics: dict, output_path: str, model_color: str = "#9467bd"):
    """Bar chart of accuracy metrics with value labels."""
    names = list(metrics.keys())
    values = list(metrics.values())

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    bars = ax.bar(names, values, alpha=0.85, color=model_color)

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h,
            f"{h:.4f}",
            ha="center",
            va="bottom",
            fontsize=PLOT_STYLE["annotation_fontsize"],
        )

    ax.set_ylabel("Value", fontsize=PLOT_STYLE["label_fontsize"])
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=15, ha="right", fontsize=PLOT_STYLE["tick_fontsize"])
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_fig(output_path)


def plot_per_position_accuracy(
    model_accuracy: np.ndarray,
    baseline_accuracy: np.ndarray,
    output_path: str,
    model_color: str = "#1f77b4",
):
    """Line plot of per-position accuracy for model vs baseline."""
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize_wide"])
    ax.plot(
        np.arange(len(model_accuracy)),
        model_accuracy,
        label="Model",
        color=model_color,
        linewidth=PLOT_STYLE["line_width"],
    )
    ax.plot(
        np.arange(len(baseline_accuracy)),
        baseline_accuracy,
        label="Random Baseline",
        color=get_series_color("random baseline"),
        linestyle="--",
        linewidth=PLOT_STYLE["line_width"],
    )

    ax.set_xlabel("Position in Sequence", fontsize=PLOT_STYLE["label_fontsize"])
    ax.set_ylabel("Accuracy", fontsize=PLOT_STYLE["label_fontsize"])
    ax.legend(fontsize=PLOT_STYLE["legend_fontsize"], loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    _save_fig(output_path)


def plot_nucleotide_confusion_matrix(
    confusion_matrix: np.ndarray,
    nucleotides: List[str],
    output_path: str,
    title: str = "",
    cmap=None,
):
    """Heatmap of nucleotide confusion matrix."""
    plt.figure(figsize=PLOT_STYLE["figsize_square"])
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap=cmap or "Blues",
        xticklabels=nucleotides,
        yticklabels=nucleotides,
        annot_kws={"size": PLOT_STYLE["annotation_fontsize"]},
    )
    plt.xlabel("Predicted", fontsize=PLOT_STYLE["label_fontsize"])
    plt.ylabel("True", fontsize=PLOT_STYLE["label_fontsize"])
    plt.tight_layout()
    _save_fig(output_path)


# ---------------------------------------------------------------------------
# Levenshtein similarity plots
# ---------------------------------------------------------------------------
def plot_levenshtein_comparison(
    model_similarities: List[float],
    baseline_similarities: Optional[List[float]],
    output_path: str,
    model_color: str = "#1f77b4",
):
    """Overlapping histograms of model vs baseline Levenshtein similarity."""
    plt.figure(figsize=PLOT_STYLE["figsize"])
    sns.histplot(
        model_similarities, label="Model", color=model_color, kde=True, alpha=0.6, element="step"
    )

    if baseline_similarities:
        sns.histplot(
            baseline_similarities,
            label="Random Baseline",
            color=get_series_color("random baseline"),
            kde=True,
            alpha=0.4,
            element="step",
        )

    plt.xlabel("Levenshtein Similarity", fontsize=PLOT_STYLE["label_fontsize"])
    plt.ylabel("Count", fontsize=PLOT_STYLE["label_fontsize"])
    plt.legend(fontsize=PLOT_STYLE["legend_fontsize"], loc="upper right")
    plt.xlim(0, 1)
    plt.tight_layout()
    _save_fig(output_path)


def plot_similarity_ridge(
    model_similarities_by_len: Dict[int, List[float]],
    baseline_similarities_by_len: Dict[int, List[float]],
    output_path: str,
    model_color: str = "#1f77b4",
):
    """Ridge plot of similarity distributions across sequence lengths."""
    import pandas as pd

    sorted_lengths = sorted(model_similarities_by_len.keys())
    records = []
    for seq_len in sorted_lengths:
        for val in model_similarities_by_len[seq_len]:
            records.append({"Length": seq_len, "Similarity": val, "Type": "Model"})
        for val in baseline_similarities_by_len.get(seq_len, []):
            records.append({"Length": seq_len, "Similarity": val, "Type": "Baseline"})

    if not records:
        return

    df = pd.DataFrame(records)
    df["Length"] = pd.Categorical(df["Length"], categories=sorted_lengths, ordered=True)

    pal = {"Model": model_color, "Baseline": get_series_color("random baseline")}
    g = sns.FacetGrid(df, row="Length", hue="Type", aspect=5, height=1.5, palette=pal)

    g.map(sns.kdeplot, "Similarity", clip_on=False, fill=True, alpha=0.6, lw=1.5, bw_adjust=0.5)
    g.map(sns.kdeplot, "Similarity", clip_on=False, fill=False, lw=1.5, bw_adjust=0.5)
    g.refline(y=0, linewidth=1, linestyle="-", color=None, clip_on=False)

    n_rows = len(sorted_lengths)
    for idx, (ax, length) in enumerate(zip(g.axes.flat, sorted_lengths)):
        ax.set_zorder(n_rows - idx)
        ax.patch.set_alpha(0)
        ax.text(
            0,
            0.2,
            f"L={length}",
            fontweight="bold",
            color="black",
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontsize=PLOT_STYLE["tick_fontsize"],
        )

    g.figure.subplots_adjust(hspace=-0.15)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    g.add_legend(fontsize=PLOT_STYLE["legend_fontsize"], title="")

    _save_fig(output_path)


# ---------------------------------------------------------------------------
# Metric vs similarity (binned aggregate + single-model scatter)
# ---------------------------------------------------------------------------
def plot_aggregate_metric_vs_similarity(
    data: Dict[str, Dict[str, List[float]]],
    metric_label: str,
    output_path: str,
    n_bins: int = 20,
):
    """Binned line plot of metric vs Levenshtein similarity with ±1σ bands."""
    plt.figure(figsize=PLOT_STYLE["figsize"])

    colors = get_series_colors(list(data.keys()))
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for idx, (model_name, series) in enumerate(data.items()):
        metric_values = np.asarray(series["metric"])
        similarities = np.asarray(series["similarity"])
        color = colors[model_name]

        bin_idx = np.clip(np.digitize(similarities, bin_edges) - 1, 0, n_bins - 1)

        bin_means = np.full(n_bins, np.nan)
        bin_stds = np.full(n_bins, np.nan)
        for b in range(n_bins):
            mask = bin_idx == b
            if np.sum(mask) >= 2:
                bin_means[b] = np.mean(metric_values[mask])
                bin_stds[b] = np.std(metric_values[mask])

        valid = ~np.isnan(bin_means)
        plt.plot(
            bin_centers[valid],
            bin_means[valid],
            marker=_MARKERS[idx % len(_MARKERS)],
            linestyle="-",
            linewidth=PLOT_STYLE["line_width"],
            markersize=PLOT_STYLE["marker_size"],
            label=get_model_display_name(model_name),
            color=color,
        )
        plt.plot(
            bin_centers[valid],
            bin_means[valid] - bin_stds[valid],
            linestyle="--",
            linewidth=1.0,
            color=color,
            alpha=0.5,
        )
        plt.plot(
            bin_centers[valid],
            bin_means[valid] + bin_stds[valid],
            linestyle="--",
            linewidth=1.0,
            color=color,
            alpha=0.5,
        )

    plt.xlabel("Levenshtein Similarity", fontsize=PLOT_STYLE["label_fontsize"])
    plt.ylabel(metric_label, fontsize=PLOT_STYLE["label_fontsize"])
    plt.legend(fontsize=PLOT_STYLE["legend_fontsize"], loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.tight_layout()
    _save_fig(output_path)


def plot_metric_vs_similarity(
    metric_values: List[float],
    similarities: List[float],
    metric_name: str,
    output_path: str,
    model_color: str = "#1f77b4",
):
    """Scatter plot of metric vs similarity with regression line (single model)."""
    plt.figure(figsize=PLOT_STYLE["figsize"])

    plt.scatter(
        similarities,
        metric_values,
        alpha=0.5,
        color=model_color,
        edgecolors="w",
        s=PLOT_STYLE["marker_size"] * 5,
        label="Data Points",
    )

    if len(similarities) > 1:
        slope, intercept, r_value, p_value, _ = stats.linregress(similarities, metric_values)
        line = slope * np.array(similarities) + intercept
        plt.plot(
            similarities,
            line,
            color="red",
            linestyle="--",
            linewidth=PLOT_STYLE["line_width"],
            label=f"Fit: y={slope:.2f}x+{intercept:.2f}\nR={r_value:.2f}, p={p_value:.2e}",
        )

    plt.xlabel("Levenshtein Similarity", fontsize=PLOT_STYLE["label_fontsize"])
    plt.ylabel(metric_name, fontsize=PLOT_STYLE["label_fontsize"])
    plt.legend(fontsize=PLOT_STYLE["legend_fontsize"], loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.tight_layout()
    _save_fig(output_path)
