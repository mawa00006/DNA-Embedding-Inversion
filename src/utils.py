"""Utility helpers: logging, determinism, hashing, and lightweight wandb integration.

Design principles (scientific code):
----------------------------------
* Fail fast: use ``assert`` for invariants so that any divergence stops execution.
* Determinism: explicit seeding and disabling of non‑deterministic backends.
* Reproducibility: log all configuration and metrics both to stdout and a file.
* Simplicity: minimal hidden state; explicit returns instead of globals where possible.
"""

from __future__ import annotations

import logging
import os
import random
import hashlib
from typing import Any, Dict, Callable, List, Type
import json
import importlib.util

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import seaborn as sns
import torch
from omegaconf import OmegaConf, DictConfig
from transformers import set_seed

import wandb

NUCLEOTIDES = ["A", "C", "G", "T"]


from src.plotting_utils import (
    configure_plot_style,
    get_series_color,
    get_series_colors,
    PLOT_STYLE,
)


def dynamic_import_class(module_file_path: str, class_name: str) -> Type:
    """Dynamically import a class from a Python file.

    Parameters
    ----------
    module_file_path : str
        Path to the Python module file (e.g., "src/mlp.py").
    class_name : str
        Name of the class to import from the module.

    Returns
    -------
    Type
        The imported class.

    Raises
    ------
    AssertionError
        If the module file does not exist or the class is not found.
    """
    spec = importlib.util.spec_from_file_location("dynamic_module", module_file_path)
    assert spec is not None, f"Failed to load spec from {module_file_path}"
    assert spec.loader is not None, f"No loader found for {module_file_path}"

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def set_determinism(seed: int) -> None:
    """Set seeds for python, numpy, and torch, enforce deterministic behavior.

    Raises
    ------
    AssertionError
        If seed is negative.
    """
    assert seed >= 0, "Seed must be non-negative"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def file_sha256(path: str) -> str:
    """Return SHA256 for a file path."""
    with open(path, "rb") as f:
        h = hashlib.file_digest(f, "sha256")
    return h.hexdigest().upper()


def load_sequences_from_file(
    file_path: str, max_length: int | None = None, max_sequences: int | None = None
) -> List[str]:
    """Load DNA sequences from a text file or grouped multi-length HDF5.

    Text files retain the legacy one-sequence-per-line behaviour, including the
    optional truncation to ``max_length``.  A multi-length source HDF5 produced
    by ``scripts/prepare_hg38_multilen.py`` is selected by exact length instead:
    ``max_length=L`` reads ``/lengths/L/sequences`` without truncation.  This
    lets all three foundation-model embedding generators consume the same
    coordinate-aware source corpus while preserving their old text inputs.

    Parameters
    ----------
    file_path : str
        Path to the input file containing DNA sequences.
    max_length : int | None
        Maximum length for sequences. If provided, sequences longer than this will be truncated.
    max_sequences : int | None
        Maximum number of sequences to load. If provided, only the first max_sequences are loaded.

    Returns
    -------
    List[str]
        List of DNA sequences.
    """
    import h5py

    if h5py.is_hdf5(file_path):
        assert max_length is not None, (
            "Selecting a multi-length HDF5 source requires max_length/seq_length"
        )
        with h5py.File(file_path, "r", swmr=True) as handle:
            schema = str(handle.attrs.get("schema", ""))
            supported_schemas = {
                "dna_inversion_hg38_multilen_v1",
                "dna_inversion_hg38_multilen_v2",
                "dna_inversion_1000g_multilen_v1",
                "dna_inversion_1000g_multilen_v2",
            }
            assert schema in supported_schemas, (
                f"Unsupported sequence-source HDF5 schema {schema!r}: {file_path}"
            )
            group_path = f"lengths/{int(max_length)}"
            assert group_path in handle, (
                f"Sequence length {max_length} missing from {file_path}; "
                f"available={sorted(map(int, handle['lengths'].keys()))}"
            )
            dataset = handle[f"{group_path}/sequences"]
            n = len(dataset) if max_sequences is None else min(len(dataset), max_sequences)
            raw = dataset[:n]
            sequences = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in raw
            ]
            if max_sequences is None and "ordered_sequence_sha256" in handle[
                group_path
            ].attrs:
                digest = hashlib.sha256()
                for sequence in sequences:
                    digest.update(sequence.encode("ascii"))
                    digest.update(b"\n")
                expected = str(
                    handle[group_path].attrs["ordered_sequence_sha256"]
                )
                assert digest.hexdigest() == expected, (
                    f"Sequence-source content hash mismatch in {file_path}:{group_path}"
                )
        assert all(len(sequence) == max_length for sequence in sequences), (
            f"Multi-length source group {group_path} contains a malformed sequence"
        )
        return sequences

    with open(file_path, "r") as f:
        sequences = [line.strip() for line in f if line.strip()]

    if max_sequences is not None:
        sequences = sequences[:max_sequences]

    if max_length is not None:
        sequences = [seq[:max_length] for seq in sequences]

    return sequences


def load_identification_csv(
    file_path: str,
) -> Dict[str, List]:
    """Load a per-individual identification CSV produced by ``scripts/prepare_identification.py``.

    Expected header: ``individual_id,locus_id,haplotype,sequence``.

    Order is preserved (no dedup, no shuffle) so downstream code can keep
    parallel arrays of metadata aligned with embeddings.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    dict
        Keys ``individual_ids`` (list[str]), ``locus_ids`` (list[int]),
        ``haplotypes`` (list[int]), ``sequences`` (list[str]).
    """
    individual_ids: List[str] = []
    locus_ids: List[int] = []
    haplotypes: List[int] = []
    sequences: List[str] = []

    with open(file_path, "r") as f:
        header = f.readline().strip()
        assert header == "individual_id,locus_id,haplotype,sequence", (
            f"Unexpected identification CSV header: {header!r}"
        )
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            assert len(parts) == 4, f"Malformed identification CSV row: {line!r}"
            individual_ids.append(parts[0])
            locus_ids.append(int(parts[1]))
            haplotypes.append(int(parts[2]))
            sequences.append(parts[3])

    assert len(sequences) > 0, f"Identification CSV is empty: {file_path}"
    return {
        "individual_ids": individual_ids,
        "locus_ids": locus_ids,
        "haplotypes": haplotypes,
        "sequences": sequences,
    }


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""

    def default(self, o):  # noqa: D102
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save a dictionary to a JSON file with support for NumPy types.

    Parameters
    ----------
    data : Dict[str, Any]
        The dictionary to save.
    path : str
        The output file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, cls=NpEncoder)


def is_topk_hit(
    scores: np.ndarray,
    target_idx: int,
    k: int,
    rng: np.random.Generator,
    descending: bool,
) -> bool:
    """Whether ``target_idx`` ranks within the top ``k`` entries of ``scores``.

    Ties at the cut-off are broken uniformly at random via ``rng``, so that
    equal-scoring candidates - including a fully uninformative all-equal vector -
    do not systematically favour any index. ``descending=True`` ranks larger
    scores first (e.g. similarities); ``descending=False`` ranks smaller first
    (e.g. distances).
    """
    assert scores.ndim == 1, f"scores must be 1D, got shape {scores.shape}"
    assert k >= 1, f"k must be >= 1, got {k}"
    target = scores[target_idx]
    if descending:
        n_better = int((scores > target).sum())
    else:
        n_better = int((scores < target).sum())
    if n_better >= k:
        return False
    n_tied = int((scores == target).sum())  # includes target itself
    slots = k - n_better
    if slots >= n_tied:
        return True
    return bool(rng.random() < slots / n_tied)


def maybe_init_wandb(cfg: DictConfig):  # noqa: D401
    """Initialize a wandb run if enabled in config, else return None.

    Notes
    -----
    Keeps dependency optional: if wandb isn't installed and the config requests it,
    an ImportError will surface clearly.
    """
    if not cfg.train.use_wandb:
        return None
    if wandb is None:  # pragma: no cover
        raise ImportError("wandb requested in config but package not installed")
    plain_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    assert isinstance(plain_cfg, dict)
    run = wandb.init(  # type: ignore[assignment]
        project=cfg.train.project,
        name=cfg.train.run_name,
        config=plain_cfg,  # type: ignore[arg-type]
        mode="online",
        reinit=False,
    )
    return run


def log_factory(logger: logging.Logger, wandb_run) -> Callable[[Dict[str, Any]], None]:
    """Create a logging callable bridging std logging and optional wandb logging."""

    def log_fn(obj: Dict[str, Any]):
        logger.info(
            " | ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in obj.items()
            )
        )
        if wandb_run is not None:
            wandb_run.log(obj)

    return log_fn


def one_hot_encode_sequence(sequence: str, one_hot_dim: int = 4) -> np.ndarray:
    """Encode a DNA sequence to one-hot vectors per nucleotide.

    Parameters
    ----------
    sequence : str
        DNA sequence string (e.g., "ACGTACGT").
    one_hot_dim : int
        Dimension of one-hot encoding for each nucleotide. Must be >= 4 for A/C/G/T;
        extra dimensions (if any) remain zeros for compatibility/testing.

    Returns
    -------
    np.ndarray
        One-hot encoded array of shape (len(sequence), one_hot_dim).
        Mapping:
        - A: [1, 0, 0, 0]
        - C: [0, 1, 0, 0]
        - G: [0, 0, 1, 0]
        - T: [0, 0, 0, 1]
    """
    assert one_hot_dim >= 4, "one_hot_dim must be at least 4 for A, C, G, T"
    assert isinstance(sequence, str), "sequence must be a string"
    assert len(sequence) > 0, "sequence cannot be empty"

    nucleotide_map = {"A": 0, "C": 1, "G": 2, "T": 3}

    # Convert sequence to uppercase and then to numpy array of characters
    seq_upper = sequence.upper()
    seq_array = np.array(list(seq_upper), dtype="U1")

    # Validate characters: only A/C/G/T are allowed
    valid_mask = np.isin(seq_array, list(nucleotide_map.keys()))
    assert bool(np.all(valid_mask)), "Sequence contains invalid nucleotides; allowed: A,C,G,T"

    # Vectorized mapping
    indices = np.empty(len(seq_array), dtype=np.int32)
    for nuc, idx in nucleotide_map.items():
        mask = seq_array == nuc
        indices[mask] = idx

    # Create one-hot encoding using advanced indexing
    encoded = np.zeros((len(sequence), one_hot_dim), dtype=np.float32)
    encoded[np.arange(len(sequence)), indices] = 1.0

    return encoded


def find_latest_run_dir(base_dir: str = "outputs") -> str:
    """Find the most recent training run directory.

    Parameters
    ----------
    base_dir : str
        Base directory containing training runs (default: "outputs").

    Returns
    -------
    str
        Path to the most recent run directory.
    """
    assert os.path.exists(base_dir), f"Base directory {base_dir} does not exist"
    # List all directories that match the pattern train-*
    run_dirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("train-")
    ]

    assert len(run_dirs) > 0, f"No training run directories found in {base_dir}"

    # Sort by modification time and get the latest
    latest_dir = max(run_dirs, key=os.path.getmtime)
    return latest_dir


def load_run_config(run_dir: str) -> Dict[str, Any]:
    """Load configuration from a training run directory.

    Parameters
    ----------
    run_dir : str
        Path to training run directory.

    Returns
    -------
    Dict[str, Any]
        The loaded configuration dictionary.
    """
    config_path = os.path.join(run_dir, ".hydra", "config.yaml")
    assert os.path.exists(config_path), f"Config file not found at {config_path}"

    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


def normalize_embeddings(
    embeddings: np.ndarray, stats: Dict[str, float], method: str = "standard"
) -> np.ndarray:
    """Normalize embeddings using precomputed statistics.

    Parameters
    ----------
    embeddings : np.ndarray
        Embeddings to normalize, shape (..., embedding_dim).
    stats : Dict[str, float]
        Dictionary with keys 'min', 'max', 'mean', 'std' containing normalization parameters.
    method : str
        Normalization method: 'standard' (z-score) or 'minmax' (0-1 range).

    Returns
    -------
    np.ndarray
        Normalized embeddings with same shape as input.
    """
    assert method in ["standard", "minmax"], f"Invalid normalization method: {method}"
    assert stats, "Statistics dictionary cannot be empty"

    if method == "standard":
        assert "mean" in stats and "std" in stats
        return (embeddings - stats["mean"]) / stats["std"]

    assert method == "minmax"
    assert "min" in stats and "max" in stats
    return (embeddings - stats["min"]) / (stats["max"] - stats["min"])


def denormalize_embeddings(
    normalized_embeddings: np.ndarray, stats: Dict[str, float], method: str = "standard"
) -> np.ndarray:
    """Denormalize embeddings back to original scale.

    Parameters
    ----------
    normalized_embeddings : np.ndarray
        Normalized embeddings to denormalize, shape (..., embedding_dim).
    stats : Dict[str, float]
        Dictionary with keys 'min', 'max', 'mean', 'std' containing normalization parameters.
    method : str
        Normalization method used: 'standard' (z-score) or 'minmax' (0-1 range).

    Returns
    -------
    np.ndarray
        Denormalized embeddings with same shape as input.
    """
    assert method in ["standard", "minmax"], f"Invalid normalization method: {method}"
    assert stats, "Statistics dictionary cannot be empty"

    if method == "standard":
        assert "mean" in stats and "std" in stats
        return normalized_embeddings * stats["std"] + stats["mean"]

    assert method == "minmax"
    assert "min" in stats and "max" in stats
    return normalized_embeddings * (stats["max"] - stats["min"]) + stats["min"]


def update_yaml_keys(yaml_path: str, updates: Dict[str, Any]) -> None:
    """Update specific keys in a YAML file while preserving comments/structure.

    This uses text processing (regex) rather than a YAML parser to ensure
    no formatting or comments are lost. It assumes keys are top-level or
    identifiable by 'key: value' pattern.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML file.
    updates : Dict[str, Any]
        Dictionary of Key -> Value to update.
    """
    import re

    if not os.path.exists(yaml_path):
        # Create new if doesn't exist (basic dump)
        with open(yaml_path, "w") as f:
            for k, v in updates.items():
                f.write(f"{k}: {v}\n")
        return

    with open(yaml_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    keys_updated = set()

    for line in lines:
        updated_line = line
        # Check if this line defines one of our keys
        # We look for "key: ..." at start of line (ignoring whitespace)
        # We be careful not to match "key_suffix:" or "# key:"
        stripped = line.strip()
        if not stripped.startswith("#"):
            for key, value in updates.items():
                # Regex to match "key:" possibly followed by spaces/values
                # We want to match "key:" or "key :" but not "key_foo:"
                pattern = rf"^(\s*){re.escape(key)}\s*:(.*)"
                match = re.match(pattern, line)
                if match:
                    indent = match.group(1)
                    # comments = match.group(2).split("#", 1)
                    # existing_comment = (" #" + comments[1]) if len(comments) > 1 else ""

                    # We just replace the value part.
                    # If we write the value, we should handle types (str, bool, int)
                    if isinstance(value, bool):
                        val_str = str(value).lower()
                    elif value is None:
                        val_str = "null"
                    else:
                        val_str = str(value)

                    # Try to preserve existing comment
                    existing_comment = ""
                    if "#" in line:
                        existing_comment = " #" + line.split("#", 1)[1].strip()

                    updated_line = f"{indent}{key}: {val_str}{existing_comment}\n"
                    keys_updated.add(key)
                    break
        new_lines.append(updated_line)

    # Append missing keys
    if len(keys_updated) < len(updates):
        new_lines.append("\n# Auto-generated updates\n")
        for key, value in updates.items():
            if key not in keys_updated:
                if isinstance(value, bool):
                    val_str = str(value).lower()
                elif value is None:
                    val_str = "null"
                else:
                    val_str = str(value)
                new_lines.append(f"{key}: {val_str}\n")

    with open(yaml_path, "w") as f:
        f.writelines(new_lines)


def variable_length_collate(batch):
    """Robust collate function for variable length inputs.

    Handles both mean mode (stacking embeddings, padding sequences) and
    per-token mode (padding embeddings and sequences).
    """
    import torch

    elem = batch[0]
    # elem is (emb, seq)
    emb, seq = elem

    if emb.dim() == 1:
        # Mean mode (fixed size embedding, variable size sequence?)
        # Sequence might still be variable length.
        # So we pad sequence, stack embeddings.
        embs = torch.stack([item[0] for item in batch])

        seqs = [item[1] for item in batch]
        max_len = max(len(s) for s in seqs)
        padded_seqs = torch.full((len(seqs), max_len), -100, dtype=torch.long)
        for i, s in enumerate(seqs):
            padded_seqs[i, : len(s)] = s

        return embs, padded_seqs

    else:
        # Per-token mode (variable size embedding, variable size sequence)
        embs = [item[0] for item in batch]
        seqs = [item[1] for item in batch]

        max_len_emb = max(len(e) for e in embs)
        max_len_seq = max(len(s) for s in seqs)

        input_dim = embs[0].size(-1)
        padded_embs = torch.zeros(len(batch), max_len_emb, input_dim)
        padded_seqs = torch.full((len(batch), max_len_seq), -100, dtype=torch.long)

        for i, (e, s) in enumerate(zip(embs, seqs)):
            padded_embs[i, : len(e), :] = e
            padded_seqs[i, : len(s)] = s

        return padded_embs, padded_seqs
