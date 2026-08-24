"""Evaluation utilities for DNA sequence reconstruction.

This module provides analysis functions for evaluating reconstructed sequences:
- Levenshtein distance (sequence similarity)
- Nucleotide frequency analysis
- Visualization and plotting functions
- Model loading and plotting utilities
"""

from __future__ import annotations

import os
import importlib.util
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch import nn
import Levenshtein
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import entropy

from src.utils import NUCLEOTIDES
from src.plotting_utils import (
    configure_plot_style,
    get_series_color,
    PLOT_STYLE,
    plot_aggregate_metrics,
)
from src.tokenizers import BaseTokenizer


configure_plot_style()


def levenshtein_similarity(seq1: str, seq2: str) -> float:
    """Compute normalized Levenshtein similarity between two sequences.

    Parameters
    ----------
    seq1 : str
        First DNA sequence.
    seq2 : str
        Second DNA sequence.

    Returns
    -------
    float
        The normalized Levenshtein similarity (0-1), where 1 means identical
        and 0 means completely different. Computed as 1 - (distance / max_length).
    """
    assert isinstance(seq1, str) and isinstance(seq2, str)

    raw_distance = Levenshtein.distance(seq1, seq2)
    max_len = max(len(seq1), len(seq2))

    # Avoid division by zero for empty sequences
    if max_len == 0:
        return 1.0

    normalized_distance = raw_distance / max_len
    return 1.0 - normalized_distance


def compute_nucleotide_frequencies(sequences: List[str]) -> Dict[str, float]:
    """Compute nucleotide frequency distribution across sequences.

    Parameters
    ----------
    sequences : List[str]
        List of DNA sequences.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping nucleotide to its frequency (0-1).
    """
    assert len(sequences) > 0, "Need at least one sequence"

    concatenated = "".join(seq.upper() for seq in sequences)
    assert len(concatenated) > 0

    invalid = set(concatenated) - set(NUCLEOTIDES)
    if len(invalid) > 0:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Stripping invalid nucleotides from sequences; allowed: A,C,G,T; found: {sorted(invalid)}"
        )
        concatenated = "".join(c for c in concatenated if c in NUCLEOTIDES)

    counts_all = Counter(concatenated)
    total_count = sum(counts_all[n] for n in NUCLEOTIDES)
    assert total_count > 0

    frequencies = {n: counts_all[n] / total_count for n in NUCLEOTIDES}
    return frequencies


def compute_shannon_entropy(sequences: List[str]) -> List[float]:
    """Compute Shannon entropy of nucleotide frequencies for each sequence.

    Parameters
    ----------
    sequences : List[str]
        List of DNA sequences.

    Returns
    -------
    List[float]
        List of Shannon entropy values (in bits).
    """
    entropies = []
    for seq in sequences:
        if not seq:
            entropies.append(0.0)
            continue

        counts = Counter(seq)
        total = len(seq)
        probs = [counts[n] / total for n in counts]
        # Base 2 for bits
        entropies.append(entropy(probs, base=2))

    return entropies


def compute_repetitiveness(sequences: List[str], k: int = 4) -> List[float]:
    """Compute repetitiveness metric based on unique k-mer ratio.

    Higher value means more repetitive (fewer unique k-mers).

    Parameters
    ----------
    sequences : List[str]
        List of DNA sequences.
    k : int
        K-mer size.

    Returns
    -------
    List[float]
        List of repetitiveness scores (unique_kmers / total_kmers).
    """
    scores = []
    for seq in sequences:
        if len(seq) < k:
            scores.append(1.0)  # Too short to be repetitive in k-mer sense
            continue

        num_kmers = len(seq) - k + 1
        kmers = set()
        for i in range(num_kmers):
            kmers.add(seq[i : i + k])

        # Ratio of unique k-mers to total possible k-mers in this sequence
        scores.append(1 - (len(kmers) / num_kmers))

    return scores


@torch.no_grad()
def reconstruct_sequences(
    model: nn.Module,
    data,
    device: torch.device,
    tokenizer: BaseTokenizer,
    mode: str = "per_token",
    seq_length: int | None = None,
    embedding_dim: int | None = None,
    normalization_stats: Dict[str, float] | None = None,
    normalization_method: str = "standard",
    data_is_mean: bool = False,
    inference_batch_size: int = 256,
) -> List[str]:
    """Reconstruct DNA sequences from embeddings using trained model.

    Parameters
    ----------
    model : nn.Module
        Trained sequence reconstruction model.
    data : h5py.File or dict-like
        Data object containing 'embeddings' key with embeddings to reconstruct from.
        Embeddings are stored as 1D arrays (mean-pooled or flattened per-nucleotide).
    device : torch.device
        Device to run inference on.
    tokenizer : BaseTokenizer
        Tokenizer for decoding sequences.
    mode : str
        Either "per_token" or "mean".
    seq_length : int | None
        Optional number of input embedding positions retained for legacy
        unpooled inputs. This never truncates the decoded output sequence.
    embedding_dim : int | None
        Embedding dimension needed for reshaping flattened arrays. Required when embeddings are 1D.
    normalization_stats : Dict[str, float] | None
        Training set normalization statistics to apply. Should contain keys 'min', 'max', 'mean', 'std'.
    normalization_method : str
        Normalization method: 'standard' (z-score) or 'minmax' (0-1 range).
    data_is_mean : bool
        If True, embeddings are pre-computed mean embeddings. If False, they are per-nucleotide.
    inference_batch_size : int
        Batch size for pre-computed mean embeddings. The legacy implementation
        issued one GPU forward pass per sequence; batching is equivalent because
        the reconstruction models are in evaluation mode and reduces evaluation
        time by orders of magnitude for the large identification cohorts.

    Returns
    -------
    List[str]
        List of reconstructed DNA sequences.
    """
    assert mode in [
        "per_token",
        "mean",
    ], f"Invalid mode: {mode}. Must be 'per_token' or 'mean'"
    assert normalization_method in [
        "standard",
        "minmax",
    ], f"Invalid normalization method: {normalization_method}"
    model.eval()

    reconstructed = []

    embeddings = data["embeddings"] if hasattr(data, "__getitem__") else data

    # Fast path used by the publication's mean-pooled models. All vectors have
    # the same width, so they can be normalized and decoded in batches without
    # changing EOS handling or any reconstruction metric.
    if mode == "mean" and data_is_mean:
        assert embedding_dim is not None
        assert inference_batch_size > 0
        for start in range(0, len(embeddings), inference_batch_size):
            end = min(start + inference_batch_size, len(embeddings))
            batch = np.asarray(embeddings[start:end], dtype=np.float32)
            if batch.ndim == 1:
                batch = batch.reshape(1, -1)
            assert batch.ndim == 2 and batch.shape[1] == embedding_dim, (
                f"Expected [batch,{embedding_dim}] mean embeddings, got {batch.shape}"
            )
            if normalization_stats:
                if normalization_method == "standard":
                    batch = (batch - normalization_stats["mean"]) / normalization_stats["std"]
                else:
                    batch = (batch - normalization_stats["min"]) / (
                        normalization_stats["max"] - normalization_stats["min"]
                    )
            batch_tensor = torch.from_numpy(batch).float().to(device)
            if hasattr(model, "iterative_decode"):
                indices_batch = model.iterative_decode(
                    batch_tensor, eos_id=tokenizer.eos_id
                ).cpu().numpy()
            else:
                pred = model(batch_tensor)
                if isinstance(pred, tuple):
                    pred = pred[0]
                indices_batch = torch.argmax(pred, dim=-1).cpu().numpy()
            reconstructed.extend(tokenizer.decode(indices) for indices in indices_batch)
        return reconstructed

    for i in range(len(embeddings)):
        # Load embedding on-the-fly (memory efficient for memory-mapped arrays)
        emb_flat = np.asarray(embeddings[i], dtype=np.float32)

        if mode == "mean":
            if data_is_mean:
                # Data is already mean-pooled, use directly
                assert emb_flat.ndim == 1, f"Expected 1D mean embedding, got {emb_flat.ndim}D"
                assert (
                    len(emb_flat) == embedding_dim
                ), f"Expected mean embedding of length {embedding_dim}, got {len(emb_flat)}"
                emb = emb_flat
            else:
                # Data is per-nucleotide, compute mean on-the-fly
                assert embedding_dim is not None, "embedding_dim required for per-nucleotide data"
                seq_length_actual = len(emb_flat) // embedding_dim
                emb = emb_flat.reshape(seq_length_actual, embedding_dim)

                # Truncate before computing mean if seq_length is specified
                if seq_length is not None:
                    emb = emb[:seq_length, :]

                emb = np.mean(emb, axis=0)

            assert emb.ndim == 1, f"Expected 1D mean embedding, got {emb.ndim}D"

            # Apply normalization using training set statistics (if provided)
            if normalization_stats:
                if normalization_method == "standard":
                    emb = (emb - normalization_stats["mean"]) / normalization_stats["std"]
                else:  # minmax
                    emb = (emb - normalization_stats["min"]) / (
                        normalization_stats["max"] - normalization_stats["min"]
                    )

            emb_tensor = torch.from_numpy(emb).float().unsqueeze(0).to(device)
        else:
            # Per-nucleotide mode
            assert not data_is_mean, "Cannot use per_token mode with mean data"
            assert embedding_dim is not None, "embedding_dim required for per_token mode"

            # Embeddings are stored flattened, need to reshape
            if emb_flat.ndim == 2:
                # Already shaped [seq_len, dim]
                assert (
                    emb_flat.shape[1] == embedding_dim
                ), f"Expected dim {embedding_dim}, got {emb_flat.shape[1]}"
                emb = emb_flat
            else:
                # Flattened 1D array
                seq_length_actual = len(emb_flat) // embedding_dim
                emb = emb_flat.reshape(seq_length_actual, embedding_dim)

            assert emb.ndim == 2, f"Expected 2D per-nucleotide embedding, got {emb.ndim}D"

            # Truncate if seq_length is specified
            if seq_length is not None:
                emb = emb[:seq_length, :]

            # Apply normalization using training set statistics (if provided)
            if normalization_stats:
                if normalization_method == "standard":
                    emb = (emb - normalization_stats["mean"]) / normalization_stats["std"]
                else:  # minmax
                    emb = (emb - normalization_stats["min"]) / (
                        normalization_stats["max"] - normalization_stats["min"]
                    )

            emb_tensor = torch.from_numpy(emb).float().unsqueeze(0).to(device)

        # Mask-Predict models refine their draft over several passes; everything
        # else predicts in a single argmax pass. Both return token ids that the
        # tokenizer decodes (stopping at the first EOS).
        if hasattr(model, "iterative_decode"):
            indices = model.iterative_decode(emb_tensor, eos_id=tokenizer.eos_id)
            indices = indices.squeeze(0).cpu().numpy()
        else:
            # Forward pass. Some models (e.g. query_decoder) return a tuple of
            # (seq_logits, length_logits); we only consume the sequence logits here
            # since length is implicit in the EOS-terminated decoded string.
            pred = model(emb_tensor)
            if isinstance(pred, tuple):
                pred = pred[0]

            # Remove batch dimension and convert to numpy
            pred_np = pred.squeeze(0).cpu().numpy()

            # Get indices
            indices = np.argmax(pred_np, axis=-1)

        # Decode through the first predicted EOS. Do not crop to seq_length or
        # to the (unavailable) true length: over-long predictions must remain
        # over-long so Levenshtein similarity penalises the insertion error.
        seq = tokenizer.decode(indices)
        reconstructed.append(seq)

    return reconstructed


def compute_sequence_accuracy(true_seqs: List[str], pred_seqs: List[str]) -> Dict[str, float]:
    """Compute various sequence accuracy metrics.

    Parameters
    ----------
    true_seqs : List[str]
        Ground truth sequences.
    pred_seqs : List[str]
        Predicted sequences.

    Returns
    -------
    Dict[str, float]
        Dictionary with accuracy metrics:
        - exact_match: fraction of perfectly reconstructed sequences
        - avg_levenshtein_similarity: average normalized Levenshtein similarity (0-1, higher is better)
        - nucleotide_accuracy: fraction of correctly predicted nucleotides
    """
    assert len(true_seqs) == len(pred_seqs), "Must have same number of sequences"
    assert len(true_seqs) > 0, "Need at least one sequence"

    n = len(true_seqs)
    exact_matches = 0
    total_lev_similarity = 0.0
    total_nucleotides = 0
    correct_nucleotides = 0

    # Compute per-sequence accuracy to get standard deviation
    nucleotide_accuracies_per_seq = []
    for t, p in zip(true_seqs, pred_seqs):
        # Matches / max(len_true, len_pred) to be consistent with overall accuracy logic
        max_len = max(len(t), len(p))
        matches = sum(1 for i in range(min(len(t), len(p))) if t[i] == p[i])
        if max_len > 0:
            nucleotide_accuracies_per_seq.append(matches / max_len)
        else:
            nucleotide_accuracies_per_seq.append(1.0)  # Both empty, consider 100% match

    # These are not returned, but could be used for logging/analysis
    # accuracy_mean = np.mean(nucleotide_accuracies_per_seq)
    # accuracy_std = np.std(nucleotide_accuracies_per_seq)
    # logger.info(f"Per-sequence nucleotide accuracy stats: mean={accuracy_mean:.4f}, std={accuracy_std:.4f}")

    for true_seq, pred_seq in zip(true_seqs, pred_seqs):
        # Exact match
        if true_seq == pred_seq:
            exact_matches += 1

        # Normalized Levenshtein similarity (higher is better)
        lev_similarity = levenshtein_similarity(true_seq, pred_seq)
        total_lev_similarity += lev_similarity

        # Nucleotide-level accuracy (overall)
        min_len = min(len(true_seq), len(pred_seq))
        for i in range(min_len):
            if true_seq[i] == pred_seq[i]:
                correct_nucleotides += 1
            total_nucleotides += 1

        # Account for length differences
        total_nucleotides += abs(len(true_seq) - len(pred_seq))

    assert total_nucleotides > 0

    return {
        "exact_match": exact_matches / n,
        "avg_levenshtein_similarity": total_lev_similarity / n,
        "nucleotide_accuracy": correct_nucleotides / total_nucleotides,
    }


def compare_nucleotide_distributions(
    true_seqs: List[str], pred_seqs: List[str]
) -> Dict[str, Dict[str, float]]:
    """Compare nucleotide frequency distributions between true and predicted sequences.

    Parameters
    ----------
    true_seqs : List[str]
        Ground truth sequences.
    pred_seqs : List[str]
        Predicted sequences.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dictionary with 'true' and 'pred' keys, each containing nucleotide frequencies.
    """
    true_freqs = compute_nucleotide_frequencies(true_seqs)
    pred_freqs = compute_nucleotide_frequencies(pred_seqs)

    return {"true": true_freqs, "pred": pred_freqs}


def compute_all_levenshtein_similarities(true_seqs: List[str], pred_seqs: List[str]) -> List[float]:
    """Compute normalized Levenshtein similarity for each sequence pair.

    Parameters
    ----------
    true_seqs : List[str]
        Ground truth sequences.
    pred_seqs : List[str]
        Predicted sequences.

    Returns
    -------
    List[float]
        List of normalized Levenshtein similarities (0-1, higher is better), one per sequence pair.
    """
    assert len(true_seqs) == len(pred_seqs), "Must have same number of sequences"

    similarities = []
    for true_seq, pred_seq in zip(true_seqs, pred_seqs):
        similarity = levenshtein_similarity(true_seq, pred_seq)
        similarities.append(similarity)

    return similarities


def load_model_from_run(run_dir: str, device: torch.device):
    """Load model architecture and weights from a training run directory.

    Parameters
    ----------
    run_dir : str
        Path to the training run directory.
    device : torch.device
        Device to load the model on.

    Returns
    -------
    nn.Module
        The loaded model with trained weights.
    """
    import logging

    logger = logging.getLogger(__name__)

    # Load the checkpoint
    checkpoint_path = os.path.join(run_dir, "model.pt")
    assert os.path.exists(checkpoint_path), f"model.pt not found in {run_dir}"

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model configuration
    config = checkpoint["config"]
    mode = checkpoint["mode"]

    # Load the model.py file from the run directory
    model_py_path = os.path.join(run_dir, "model.py")
    assert os.path.exists(model_py_path), f"model.py not found in {run_dir}"

    # Dynamically import the model module
    spec = importlib.util.spec_from_file_location("run_model", model_py_path)
    assert spec is not None and spec.loader is not None
    run_model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_model_module)

    model_type = config["model"]["model_type"]
    if model_type == "encoder":
        model_class_name = "EncoderReconstructor"
    elif model_type == "decoder":
        model_class_name = "DecoderReconstructor"
    elif model_type == "knn":
        model_class_name = "KNNReconstructor"
    elif model_type == "resnet":
        model_class_name = "ResNetReconstructor"
    elif model_type == "query_decoder":
        model_class_name = "QueryDecoderReconstructor"
    elif model_type == "mask_predict":
        model_class_name = "MaskPredictReconstructor"
    else:
        model_class_name = (
            "SequenceMeanReconstructor" if mode == "mean" else "SequenceReconstructor"
        )
    ModelClass = getattr(run_model_module, model_class_name)

    output_dim = checkpoint["output_dim"]
    effective_seq_length = checkpoint.get("effective_seq_length", config["data"]["seq_length"])

    logger.info(
        f"Loaded checkpoint with input_dim={checkpoint['input_dim']}, output_dim={output_dim}, "
        f"mode={mode}, effective_seq_length={effective_seq_length}"
    )

    # Build model kwargs based on model type and mode

    # Build model kwargs based on model type and mode (matching train.py logic)

    if model_type == "encoder" or model_type == "decoder":
        # Transformer/Decoder calculates output_dim internally
        model_kwargs = {
            "input_dim": checkpoint["input_dim"],
            "hidden_dims": config["model"]["hidden_dims"],
            "mode": mode,
            "seq_length": effective_seq_length,
            "output_dim": output_dim,
            "d_model": config["model"]["d_model"],
            "nhead": config["model"]["nhead"],
            "num_layers": config["model"]["num_layers"],
            "dim_feedforward": config["model"]["dim_feedforward"],
            "dropout": config["model"]["dropout"],
        }
    elif model_type == "knn":
        model_kwargs = {
            "input_dim": checkpoint["input_dim"],
            "output_dim": output_dim,
            "k": config["model"]["k"],
        }
    elif model_type == "resnet":
        model_kwargs = {
            "input_dim": checkpoint["input_dim"],
            "mode": mode,
            "seq_length": effective_seq_length,
            "output_dim": output_dim,
            "d_model": config["model"]["d_model"],
            "n_blocks": config["model"]["n_blocks"],
            "kernel_size": config["model"]["kernel_size"],
            "dropout": config["model"]["dropout"],
        }
    elif model_type == "query_decoder":
        model_kwargs = {
            "input_dim": checkpoint["input_dim"],
            "mode": mode,
            "seq_length": effective_seq_length,
            "output_dim": output_dim,
            "d_model": config["model"]["d_model"],
            "nhead": config["model"]["nhead"],
            "num_layers": config["model"]["num_layers"],
            "dim_feedforward": config["model"]["dim_feedforward"],
            "dropout": config["model"]["dropout"],
            "n_context_tokens": config["model"]["n_context_tokens"],
            "aux_length_loss_weight": config["model"]["aux_length_loss_weight"],
        }
    elif model_type == "mask_predict":
        model_kwargs = {
            "input_dim": checkpoint["input_dim"],
            "mode": mode,
            "seq_length": effective_seq_length,
            "output_dim": output_dim,
            "d_model": config["model"]["d_model"],
            "nhead": config["model"]["nhead"],
            "num_layers": config["model"]["num_layers"],
            "dim_feedforward": config["model"]["dim_feedforward"],
            "dropout": config["model"]["dropout"],
            "n_context_tokens": config["model"]["n_context_tokens"],
            "aux_length_loss_weight": config["model"]["aux_length_loss_weight"],
            "num_iterations": config["model"]["num_iterations"],
        }
    elif mode == "mean":
        # Mean mode MLP calculates output_dim from seq_length * output_dim
        model_kwargs = {
            "input_dim": checkpoint["input_dim"],
            "hidden_dims": config["model"]["hidden_dims"],
            "seq_length": effective_seq_length,
            "output_dim": output_dim,
            "dropout": config["model"]["dropout"],
        }
    else:
        # Per-nucleotide mode uses output_dim
        model_kwargs = {
            "input_dim": checkpoint["input_dim"],
            "hidden_dims": config["model"]["hidden_dims"],
            "output_dim": output_dim,
            "dropout": config["model"]["dropout"],
        }

    # Instantiate the model using parameters from checkpoint
    model = ModelClass(**model_kwargs)

    # For KNN, we need to resize buffers to match checkpoint before loading
    if model_type == "knn":
        state_dict = checkpoint["state_dict"]
        if "train_embeddings" in state_dict:
            # We can't use resize_ because the tensor might be on a different device or have different properties
            # Instead, we just replace the buffer with a new tensor of the correct size
            # This ensures load_state_dict finds a tensor of the correct shape
            model.train_embeddings = torch.empty(state_dict["train_embeddings"].shape)
        if "train_sequences" in state_dict:
            model.train_sequences = torch.empty(state_dict["train_sequences"].shape)

    # Load weights
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Loaded model from {run_dir}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    return model


def plot_nucleotide_frequencies(
    freq_dict: dict,
    output_path: str,
    title: str = "Nucleotide Frequencies",
    model_color: str = "#1f77b4",
):
    """Plot nucleotide frequency comparison.

    Parameters
    ----------
    freq_dict : dict
        Dictionary with 'true' and 'pred' keys containing nucleotide frequencies.
    output_path : str
        Path to save the plot.
    title : str
        Plot title.
    model_color : str
        Color for the model predictions.
    """
    true_freqs = [freq_dict["true"][nuc] for nuc in NUCLEOTIDES]
    pred_freqs = [freq_dict["pred"][nuc] for nuc in NUCLEOTIDES]

    x = np.arange(len(NUCLEOTIDES))
    width = 0.35

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize_square"])

    # Plot True frequencies (Reference - Gray)
    ax.bar(
        x - width / 2,
        true_freqs,
        width,
        label="True (Reference)",
        alpha=0.85,
        color="gray",
        edgecolor="black",
        linewidth=1,
    )

    # Plot Predicted frequencies (Model - model_color)
    ax.bar(
        x + width / 2,
        pred_freqs,
        width,
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
    # Frequencies are proportions in [0,1]
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_accuracy_metrics(metrics: dict, output_path: str, model_color: str = "#9467bd"):
    """Plot accuracy metrics as a bar chart.

    Parameters
    ----------
    metrics : dict
        Dictionary of accuracy metrics.
    output_path : str
        Path to save the plot.
    model_color : str
        Color for bars.
    """
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    bars = ax.bar(metric_names, metric_values, alpha=0.85, color=model_color)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=PLOT_STYLE["annotation_fontsize"],
        )

    ax.set_ylabel("Value", fontsize=PLOT_STYLE["label_fontsize"])
    # Metrics are proportions in [0,1]
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=15, ha="right", fontsize=PLOT_STYLE["tick_fontsize"])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def save_sequences_to_csv(true_seqs: list, pred_seqs: list, similarities: list, output_path: str):
    """Save true and predicted sequences to CSV file.

    Parameters
    ----------
    true_seqs : list
        Ground truth sequences.
    pred_seqs : list
        Predicted sequences.
    similarities : list
        Normalized Levenshtein similarities (0-1, higher is better) for each pair.
    output_path : str
        Path to save the CSV file.
    """
    assert len(true_seqs) == len(pred_seqs) == len(similarities)

    df = pd.DataFrame(
        {
            "index": range(len(true_seqs)),
            "true_sequence": true_seqs,
            "predicted_sequence": pred_seqs,
            "levenshtein_similarity": similarities,
            "exact_match": [t == p for t, p in zip(true_seqs, pred_seqs)],
        }
    )

    df.to_csv(output_path, index=False)


def generate_random_baseline_sequences(
    true_seqs: List[str], nucleotides: List[str] = ["A", "C", "G", "T"]
) -> List[str]:
    """Generate random baseline sequences matching true sequence characteristics.

    Generates random sequences with the same lengths and nucleotide distribution
    as the true sequences, representing a random guessing baseline.

    Parameters
    ----------
    true_seqs : List[str]
        Ground truth sequences to match lengths from.
    nucleotides : List[str]
        List of nucleotides to sample from.

    Returns
    -------
    List[str]
        List of randomly generated sequences with same lengths as true_seqs.
    """
    assert len(true_seqs) > 0

    # Compute nucleotide frequencies from true sequences
    freqs = compute_nucleotide_frequencies(true_seqs)

    # Create probability distribution (excluding 'N' if present)
    probs = [freqs.get(nuc, 0.0) for nuc in nucleotides]
    total = sum(probs)
    assert total > 0
    probs = [p / total for p in probs]

    # Generate random sequences with same lengths
    random_seqs = []
    for true_seq in true_seqs:
        seq_length = len(true_seq)
        random_seq = "".join(np.random.choice(nucleotides, size=seq_length, p=probs))
        random_seqs.append(random_seq)

    return random_seqs


def compute_per_position_accuracy(true_seqs: List[str], pred_seqs: List[str]) -> np.ndarray:
    """Compute accuracy at each sequence position.

    Parameters
    ----------
    true_seqs : List[str]
        Ground truth sequences.
    pred_seqs : List[str]
        Predicted sequences.

    Returns
    -------
    np.ndarray
        Array of shape (seq_length,) with accuracy at each position.
    """
    assert len(true_seqs) == len(pred_seqs)
    assert len(true_seqs) > 0

    # Find maximum sequence length
    max_len = max(max(len(s) for s in true_seqs), max(len(s) for s in pred_seqs))

    # Count correct predictions at each position
    correct_counts = np.zeros(max_len)
    total_counts = np.zeros(max_len)

    for true_seq, pred_seq in zip(true_seqs, pred_seqs):
        min_len = min(len(true_seq), len(pred_seq))
        for i in range(min_len):
            if true_seq[i] == pred_seq[i]:
                correct_counts[i] += 1
            total_counts[i] += 1

        # Account for length mismatches (counted as errors)
        for i in range(min_len, max(len(true_seq), len(pred_seq))):
            total_counts[i] += 1

    # Compute accuracy (avoid division by zero)
    accuracy = np.where(total_counts > 0, correct_counts / total_counts, 0)

    return accuracy


def compute_nucleotide_confusion_matrix(true_seqs: List[str], pred_seqs: List[str]) -> np.ndarray:
    """Compute confusion matrix for nucleotide predictions.

    Parameters
    ----------
    true_seqs : List[str]
        Ground truth sequences.
    pred_seqs : List[str]
        Predicted sequences.
    Returns
    -------
    np.ndarray
        Confusion matrix of shape (len(nucleotides), len(nucleotides)).
        Rows represent true nucleotides, columns represent predicted nucleotides.
    """
    assert len(true_seqs) == len(pred_seqs)

    n_classes = len(NUCLEOTIDES)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    nuc_to_idx = {nuc: i for i, nuc in enumerate(NUCLEOTIDES)}

    for true_seq, pred_seq in zip(true_seqs, pred_seqs):
        min_len = min(len(true_seq), len(pred_seq))
        for i in range(min_len):
            true_nuc = true_seq[i].upper()
            pred_nuc = pred_seq[i].upper()

            # Only count nucleotides in our list
            if true_nuc in nuc_to_idx and pred_nuc in nuc_to_idx:
                true_idx = nuc_to_idx[true_nuc]
                pred_idx = nuc_to_idx[pred_nuc]
                confusion[true_idx, pred_idx] += 1

    return confusion


def plot_per_position_accuracy(
    model_accuracy: np.ndarray,
    baseline_accuracy: np.ndarray,
    output_path: str,
    model_color: str = "#1f77b4",
):
    """Plot per-position accuracy for model and baseline.

    Parameters
    ----------
    model_accuracy : np.ndarray
        Model accuracy at each position.
    baseline_accuracy : np.ndarray
        Baseline accuracy at each position.
    output_path : str
        Path to save the plot.
    model_color : str
        Color for the model line.
    """
    positions_model = np.arange(len(model_accuracy))
    positions_baseline = np.arange(len(baseline_accuracy))

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize_wide"])
    baseline_color = get_series_color("random baseline")

    ax.plot(
        positions_model,
        model_accuracy,
        label="Model",
        linewidth=PLOT_STYLE["line_width"],
        alpha=0.85,
        color=model_color,
    )
    ax.plot(
        positions_baseline,
        baseline_accuracy,
        label="Random Baseline",
        linewidth=PLOT_STYLE["line_width"],
        alpha=0.85,
        linestyle="--",
        color=baseline_color,
    )

    # Add mean lines
    ax.axhline(
        np.mean(model_accuracy),
        color=model_color,
        linestyle=":",
        alpha=0.5,
        label=f"Model Mean: {np.mean(model_accuracy):.3f}",
    )
    ax.axhline(
        np.mean(baseline_accuracy),
        color=baseline_color,
        linestyle=":",
        alpha=0.5,
        label=f"Baseline Mean: {np.mean(baseline_accuracy):.3f}",
    )

    ax.set_xlabel("Position in Sequence", fontsize=PLOT_STYLE["label_fontsize"])
    ax.set_ylabel("Accuracy", fontsize=PLOT_STYLE["label_fontsize"])
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)  # Accuracy is a proportion in [0,1]

    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_nucleotide_confusion_matrix(
    confusion: np.ndarray,
    nucleotides: List[str],
    output_path: str,
    title: str = "Nucleotide Confusion Matrix",
    cmap: Any = "Blues",
):
    """Plot nucleotide confusion matrix as heatmap.

    Parameters
    ----------
    confusion : np.ndarray
        Confusion matrix.
    nucleotides : List[str]
        List of nucleotide labels.
    output_path : str
        Path to save the plot.
    title : str
        Plot title.
    cmap : Any
        Colormap to use.
    """
    # Normalize by row (true labels) to show percentages.
    row_sums = confusion.sum(axis=1, keepdims=True)
    confusion_normalized = np.zeros_like(confusion, dtype=float)
    np.divide(confusion, row_sums, out=confusion_normalized, where=(row_sums != 0))

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize_square"])

    # Create heatmap
    im = ax.imshow(confusion_normalized, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Proportion", fontsize=PLOT_STYLE["label_fontsize"])

    # Set ticks and labels
    ax.set_xticks(np.arange(len(nucleotides)))
    ax.set_yticks(np.arange(len(nucleotides)))
    ax.set_xticklabels(nucleotides, fontsize=PLOT_STYLE["tick_fontsize"])
    ax.set_yticklabels(nucleotides, fontsize=PLOT_STYLE["tick_fontsize"])

    # Add text annotations
    for i in range(len(nucleotides)):
        for j in range(len(nucleotides)):
            text_color = "white" if confusion_normalized[i, j] > 0.5 else "black"
            text = ax.text(
                j,
                i,
                f"{confusion_normalized[i, j]:.2f}\n({confusion[i, j]})",
                ha="center",
                va="center",
                color=text_color,
                fontsize=PLOT_STYLE["annotation_fontsize"],
            )

    ax.set_xlabel("Predicted Nucleotide", fontsize=PLOT_STYLE["label_fontsize"])
    ax.set_ylabel("True Nucleotide", fontsize=PLOT_STYLE["label_fontsize"])

    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_levenshtein_comparison(
    model_similarities: List[float],
    baseline_similarities: List[float],
    output_path: str,
    model_color: str = "#9467bd",
):
    """Plot KDE comparison of Levenshtein similarity distributions for model vs baseline.

    Parameters
    ----------
    model_similarities : List[float]
        Model's normalized Levenshtein similarities (higher is better).
    baseline_similarities : List[float]
        Baseline's normalized Levenshtein similarities (higher is better).
    output_path : str
        Path to save the plot.
    model_color : str
        Color for the model distribution.
    """
    plt.figure(figsize=PLOT_STYLE["figsize"])
    baseline_color = get_series_color("random baseline")

    # KDE comparison
    sns.kdeplot(
        model_similarities,
        fill=True,
        alpha=0.5,
        linewidth=PLOT_STYLE["line_width"],
        label="Model",
        color=model_color,
    )
    sns.kdeplot(
        baseline_similarities,
        fill=True,
        alpha=0.5,
        linewidth=PLOT_STYLE["line_width"],
        label="Random Baseline",
        color=baseline_color,
    )

    model_mean = np.mean(model_similarities)
    baseline_mean = np.mean(baseline_similarities)

    plt.axvline(
        model_mean,
        color=model_color,
        linestyle="--",
        linewidth=PLOT_STYLE["line_width"],
        label=f"Model Mean: {model_mean:.3f}",
    )
    plt.axvline(
        baseline_mean,
        color=baseline_color,
        linestyle="--",
        linewidth=PLOT_STYLE["line_width"],
        label=f"Baseline Mean: {baseline_mean:.3f}",
    )

    # Add improvement annotation
    improvement = ((model_mean - baseline_mean) / baseline_mean) * 100
    plt.text(
        0.98,
        0.98,
        f"Model Improvement: {improvement:.1f}%",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=PLOT_STYLE["annotation_fontsize"],
    )

    plt.xlabel("Levenshtein Similarity (higher is better)", fontsize=PLOT_STYLE["label_fontsize"])
    plt.ylabel("Density", fontsize=PLOT_STYLE["label_fontsize"])
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.xlim(0, 1)

    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_similarity_ridge(
    model_similarities_by_length: Dict[int, List[float]],
    baseline_similarities_by_length: Dict[int, List[float]],
    output_path: str,
    model_color: str = "#9467bd",
):
    """Plot ridge plot showing similarity distribution progression across sequence lengths.

    Creates a ridge plot (also known as joy plot) showing how Levenshtein similarity
    distributions evolve across different sequence lengths for both model and baseline.

    Parameters
    ----------
    model_similarities_by_length : Dict[int, List[float]]
        Dictionary mapping sequence length to list of model similarities.
    baseline_similarities_by_length : Dict[int, List[float]]
        Dictionary mapping sequence length to list of baseline similarities.
    output_path : str
        Path to save the plot.
    model_color : str
        Color for the model distribution.
    """
    baseline_color = get_series_color("random baseline")

    # Build DataFrame for seaborn FacetGrid
    rows = []
    seq_lengths = sorted(model_similarities_by_length.keys())

    for seq_len in seq_lengths:
        for sim in model_similarities_by_length[seq_len]:
            rows.append({"Sequence Length": seq_len, "Similarity": sim, "Type": "Model"})
        for sim in baseline_similarities_by_length[seq_len]:
            rows.append({"Sequence Length": seq_len, "Similarity": sim, "Type": "Random"})

    df = pd.DataFrame(rows)

    # Create ridge plot using FacetGrid with overlapping KDEs
    g = sns.FacetGrid(
        df,
        row="Sequence Length",
        hue="Type",
        aspect=5,
        height=1.2,
        palette={"Model": model_color, "Random": baseline_color},
    )

    # Draw KDE for each row
    g.map(
        sns.kdeplot,
        "Similarity",
        bw_adjust=0.5,
        clip=(0, 1),
        fill=True,
        alpha=0.5,
        linewidth=1.5,
    )
    g.map(sns.kdeplot, "Similarity", bw_adjust=0.5, clip=(0, 1), color="black", linewidth=1)

    # Adjust overlap
    g.figure.subplots_adjust(hspace=-0.1)

    # Remove axes details that overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    # Add sequence length labels on the left
    for ax, seq_len in zip(g.axes.flat, seq_lengths):
        ax.text(
            -0.05,
            0.5,
            f"L={seq_len}",
            transform=ax.transAxes,
            fontsize=PLOT_STYLE["tick_fontsize"],
            ha="right",
            va="center",
        )

    # Add legend
    g.add_legend(title="", fontsize=PLOT_STYLE["legend_fontsize"])

    # Set x-axis label on bottom plot
    g.axes[-1, 0].set_xlabel(
        "Levenshtein Similarity (higher is better)", fontsize=PLOT_STYLE["label_fontsize"]
    )

    # Set overall title
    g.figure.suptitle("")  # No title for publication-ready plots

    g.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
    g.savefig(f"{output_path}.pdf", dpi=300, bbox_inches="tight")
    plt.close()
