"""Data utilities and dataset definitions for DNA sequence reconstruction."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import hydra.utils as hy_utils
import h5py
import logging

from omegaconf import DictConfig
from src.utils import file_sha256
from src.tokenizers import BaseTokenizer


def _to_scalar(value) -> float:
    """Coerce an HDF5 attribute value (possibly a 0-d / size-1 array) to a Python float."""
    if isinstance(value, np.ndarray):
        assert value.size == 1, (
            f"Expected scalar attribute, got array of size {value.size}. "
            "Statistics should be global (computed across all embedding values), not per-dimension."
        )
        return float(value.item())
    return float(value)


def compute_pooled_train_stats(train_files: List[h5py.File]) -> Dict[str, float]:
    """Pool min/max/mean/std across multiple per-length training HDF5 files.

    Uses count-weighted pooling so the resulting (mean, std) matches what a
    single pass over the concatenated training set would compute. The returned
    dict is empty when the files do not carry the ``emb_*`` attributes.
    """
    assert len(train_files) > 0, "Need at least one train file to pool stats"
    if "emb_min" not in train_files[0].attrs:
        return {}

    counts = [len(h5["embeddings"]) for h5 in train_files]
    mus = [_to_scalar(h5.attrs["emb_mean"]) for h5 in train_files]
    sigmas = [_to_scalar(h5.attrs["emb_std"]) for h5 in train_files]
    mins = [_to_scalar(h5.attrs["emb_min"]) for h5 in train_files]
    maxs = [_to_scalar(h5.attrs["emb_max"]) for h5 in train_files]

    total_n = sum(counts)
    assert total_n > 0, "No training samples found across multi files"
    pooled_mean = sum(c * m for c, m in zip(counts, mus)) / total_n
    pooled_var = (
        sum(c * (s * s + (m - pooled_mean) ** 2) for c, m, s in zip(counts, mus, sigmas))
        / total_n
    )

    return {
        "min": min(mins),
        "max": max(maxs),
        "mean": pooled_mean,
        "std": pooled_var**0.5,
    }


class _LazyH5Reader:
    """Owns one HDF5 handle per process for the sample-level reads in __getitem__.

    HDF5 handles are not fork-safe: a handle opened in the parent and inherited by
    a forked DataLoader worker returns corrupt data rather than failing, so the
    handle is keyed by PID and reopened the first time a given process reads.
    """

    def __init__(self, path: str):
        self.path = path
        self._pid: int | None = None
        self._embeddings = None
        self._sequences = None

    def datasets(self):
        if self._pid != os.getpid():
            handle = h5py.File(self.path, "r", swmr=True)
            self._embeddings = handle["embeddings"]
            self._sequences = handle["sequences"]
            self._pid = os.getpid()
        return self._embeddings, self._sequences


class DNAEmbeddingDataset(Dataset):
    """Dataset wrapper for per-nucleotide embeddings and DNA sequences.

    This dataset uses lazy loading via HDF5 - embeddings are loaded from disk on-the-fly
    during __getitem__ to minimize memory usage for large datasets.

    Each sample contains:
    - embeddings: shape (seq_length, embedding_dim) - loaded on-the-fly
    - sequences_indices: shape (seq_length,) - computed on-the-fly (LongTensor)
    """

    def __init__(
        self,
        h5_file: h5py.File,
        tokenizer: BaseTokenizer,
        embedding_dim: int,
        seq_length: int | None = None,
        normalization_stats: Dict[str, float] | None = None,
        normalization_method: str = "standard",
        max_samples: int | None = None,
    ):
        assert tokenizer is not None, "Tokenizer must be provided"
        assert seq_length is None or seq_length > 0, "seq_length must be positive or None"
        assert normalization_method in [
            "standard",
            "minmax",
        ], f"Invalid normalization method: {normalization_method}"
        assert max_samples is None or max_samples > 0, "max_samples must be positive or None"

        self.h5_file = h5_file
        self.num_rows = len(h5_file["embeddings"])
        self._reader = _LazyH5Reader(h5_file.filename)
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.normalization_stats = normalization_stats
        self.normalization_method = normalization_method
        self.max_samples = max_samples
        self.tokenizer = tokenizer

    def __len__(self) -> int:  # noqa: D401
        if self.max_samples is not None:
            return min(self.num_rows, self.max_samples)
        return self.num_rows

    def __getitem__(self, idx: int):  # noqa: D401
        # Load embedding from disk on-the-fly (HDF5 handles memory mapping)
        embeddings, sequences = self._reader.datasets()
        emb_flat = np.asarray(embeddings[idx], dtype=np.float32)

        # Reshape from flattened array back to 2D [seq_length, embedding_dim]
        seq_length = len(emb_flat) // self.embedding_dim
        emb = emb_flat.reshape(seq_length, self.embedding_dim)
        assert emb.ndim == 2 and emb.shape[1] == self.embedding_dim

        # Truncate embeddings if seq_length is specified
        if self.seq_length is not None:
            emb = emb[: self.seq_length, :]

        # Apply normalization using training set statistics (if provided)
        if self.normalization_stats:
            if self.normalization_method == "standard":
                emb = (emb - self.normalization_stats["mean"]) / self.normalization_stats["std"]
            else:  # minmax
                emb = (emb - self.normalization_stats["min"]) / (
                    self.normalization_stats["max"] - self.normalization_stats["min"]
                )

        # Decode sequence from bytes and tokenize on-the-fly
        seq_bytes = sequences[idx]
        seq_str = seq_bytes.decode("utf-8") if isinstance(seq_bytes, bytes) else str(seq_bytes)

        seq_indices = self.tokenizer.encode(seq_str)

        # Truncate sequence indices if seq_length is specified (to match embedding truncation)
        if self.seq_length is not None:
            seq_indices = seq_indices[: self.seq_length]

        # Convert to torch tensors
        emb_tensor = torch.from_numpy(emb).float()
        # seq_indices is already a LongTensor from tokenizer
        return emb_tensor, seq_indices


class DNAMeanEmbeddingDataset(Dataset):
    """Dataset wrapper for mean nucleotide embeddings and DNA sequences.

    This dataset uses lazy loading via HDF5. Supports both:
    - Pre-computed mean embeddings (data_is_precomputed=True)
    - On-the-fly mean computation from per-nucleotide embeddings (data_is_precomputed=False)

    Each sample contains:
    - embeddings: shape (embedding_dim) - mean-pooled embedding
    - sequences_indices: shape (seq_length,) - computed on-the-fly (LongTensor)
    """

    def __init__(
        self,
        h5_file: h5py.File,
        tokenizer: BaseTokenizer,
        embedding_dim: int,
        seq_length: int | None = None,
        normalization_stats: Dict[str, float] | None = None,
        normalization_method: str = "standard",
        data_is_precomputed: bool = False,
        max_samples: int | None = None,
    ):
        assert tokenizer is not None, "Tokenizer must be provided"
        assert seq_length is None or seq_length > 0, "seq_length must be positive or None"
        assert normalization_method in [
            "standard",
            "minmax",
        ], f"Invalid normalization method: {normalization_method}"
        assert max_samples is None or max_samples > 0, "max_samples must be positive or None"

        self.h5_file = h5_file
        self.num_rows = len(h5_file["embeddings"])
        self._reader = _LazyH5Reader(h5_file.filename)
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.normalization_stats = normalization_stats
        self.normalization_method = normalization_method
        self.data_is_precomputed = data_is_precomputed
        self.max_samples = max_samples
        self.tokenizer = tokenizer

    def __len__(self) -> int:  # noqa: D401
        if self.max_samples is not None:
            return min(self.num_rows, self.max_samples)
        return self.num_rows

    def __getitem__(self, idx: int):  # noqa: D401
        # Load embedding from disk on-the-fly (HDF5 handles memory mapping)
        embeddings, sequences = self._reader.datasets()
        emb_flat = np.asarray(embeddings[idx], dtype=np.float32)

        if self.data_is_precomputed:
            # Embeddings are already mean-pooled - just validate shape
            assert emb_flat.ndim == 1 and len(emb_flat) == self.embedding_dim
            emb_mean = emb_flat
        else:
            # Compute mean from per-nucleotide embeddings
            seq_length = len(emb_flat) // self.embedding_dim
            emb = emb_flat.reshape(seq_length, self.embedding_dim)
            assert emb.ndim == 2 and emb.shape[1] == self.embedding_dim

            # Truncate embeddings if seq_length is specified
            if self.seq_length is not None:
                emb = emb[: self.seq_length, :]

            # Compute mean embedding on-the-fly to save memory
            emb_mean = np.mean(emb, axis=0)

        # Apply normalization using training set statistics (if provided)
        if self.normalization_stats:
            if self.normalization_method == "standard":
                emb_mean = (emb_mean - self.normalization_stats["mean"]) / self.normalization_stats[
                    "std"
                ]
            else:  # minmax
                emb_mean = (emb_mean - self.normalization_stats["min"]) / (
                    self.normalization_stats["max"] - self.normalization_stats["min"]
                )

        # Decode sequence from bytes and tokenize on-the-fly
        seq_bytes = sequences[idx]
        seq_str = seq_bytes.decode("utf-8") if isinstance(seq_bytes, bytes) else str(seq_bytes)

        seq_indices = self.tokenizer.encode(seq_str)

        # Truncate sequence indices if seq_length is specified
        if self.seq_length is not None:
            seq_indices = seq_indices[: self.seq_length]

        # Append EOS so the model learns where the sequence ends. In multi-length
        # training the EOS position is the only signal that distinguishes a
        # length-15 target from a length-100 one (both share the same mean embedding shape).
        eos = torch.tensor([self.tokenizer.eos_id], dtype=torch.long)
        seq_indices = torch.cat([seq_indices, eos], dim=0)

        # Convert to torch tensors
        emb_tensor = torch.from_numpy(emb_mean).float()
        # seq_indices is already a LongTensor from tokenizer
        return emb_tensor, seq_indices


def max_target_length(
    files: List[h5py.File],
    seq_lengths: List[int],
    tokenizer: BaseTokenizer,
    chunk_size: int = 100_000,
) -> int:
    """Longest target the datasets built over ``files`` can produce, in tokens.

    Mirrors what ``__getitem__`` does to build a target -- tokenize, truncate to
    the file's ``seq_length``, append EOS -- and returns the maximum over every
    sequence, so a decoder of this width can represent all of them exactly.

    Every sequence is scanned rather than a sample. The bound has to hold for
    each individual target, and the token-length tail is far thinner than a
    sample reveals: over the 3.9M real 100-mers of the hg38 training set the
    DNABERT-2 tokenizer emits 29 tokens exactly once (28 twice, 27 for 24 of
    them), while the first 20k rows top out at 25. A sampled bound plus fixed
    headroom is therefore a coin flip on runs that last days; an exact pass over
    the 31.6M-sequence multi-length set costs ~10-15 min (~35-70k sequences/sec,
    slower when the per-FM runs scan concurrently) and cannot be wrong.

    Scanning is also what keeps the width stable: it ignores ``subset_fraction``
    and ``max_samples``, so a calibration run on 5% of the data builds the same
    architecture as the full run, and a requeued job rebuilds the width its
    checkpoint was saved with.
    """
    assert len(files) == len(seq_lengths), (
        f"max_target_length mismatch: {len(files)} files vs {len(seq_lengths)} seq_lengths"
    )
    logger = logging.getLogger(__name__)

    longest = 0
    for h5, seq_len in zip(files, seq_lengths):
        sequences = h5["sequences"]
        file_longest = 0
        for start in range(0, len(sequences), chunk_size):
            chunk = [
                s.decode("utf-8") if isinstance(s, bytes) else str(s)
                for s in sequences[start : start + chunk_size]
            ]
            for n_tokens in tokenizer.token_lengths(chunk):
                # +1 for the EOS the dataset appends after truncating to seq_len.
                target_len = min(n_tokens, seq_len) + 1
                file_longest = max(file_longest, target_len)
        logger.info(
            f"Scanned {len(sequences)} sequences of length {seq_len} in "
            f"{os.path.basename(h5.filename)}: longest target = {file_longest} tokens"
        )
        longest = max(longest, file_longest)

    assert longest > 0, "Scanned no sequences while sizing the decoder"
    return longest


def _validate_embedding_input_attrs(
    h5_file: h5py.File, *, expected_digest: str, expected_length: int
) -> None:
    """Verify that a release embedding shard is bound to its source group."""
    assert bool(h5_file.attrs.get("generation_complete", False)), (
        f"Embedding file is incomplete: {h5_file.filename}"
    )
    assert int(h5_file.attrs["seq_length"]) == expected_length
    assert str(h5_file.attrs["input_sequence_sha256"]) == expected_digest, (
        f"Embedding/source provenance mismatch: {h5_file.filename}"
    )


def _validate_multilen_release_source(
    cfg: DictConfig, data_dict: Dict[str, List[h5py.File]]
) -> None:
    """Validate v2 source invariants and every per-length embedding binding."""
    if not bool(cfg.get("require_sequence_disjoint", False)):
        return
    source_path = hy_utils.to_absolute_path(cfg.source_corpus)
    with h5py.File(source_path, "r", swmr=True) as source:
        assert str(source.attrs["schema"]) == str(cfg.source_schema)
        assert bool(source.attrs["generation_complete"])
        assert bool(source.attrs["sequence_content_unique_per_length"])
        assert bool(source.attrs["sequence_content_split_disjoint"])
        split_lengths = {
            "train": list(cfg.train_seq_lengths),
            "val": list(cfg.seq_lengths),
            "test": list(cfg.seq_lengths),
        }
        val_size = int(source.attrs["val_size_per_length"])
        test_size = int(source.attrs["test_size_per_length"])
        for split, files in data_dict.items():
            for length, h5_file in zip(split_lengths[split], files, strict=True):
                group = source[f"lengths/{int(length)}"]
                expected_digest = str(group.attrs["ordered_sequence_sha256"])
                _validate_embedding_input_attrs(
                    h5_file,
                    expected_digest=expected_digest,
                    expected_length=int(length),
                )
                target = int(group.attrs["target_count"])
                expected_rows = {
                    "train": target - val_size - test_size,
                    "val": val_size,
                    "test": test_size,
                }[split]
                assert len(h5_file["sequences"]) == expected_rows, (
                    f"Unexpected {split} size for length {length}: "
                    f"{len(h5_file['sequences'])} != {expected_rows}"
                )


def load_split_embeddings(
    cfg: DictConfig,
) -> Tuple[Dict[str, h5py.File], Dict[str, int], Dict[str, Dict[str, float]]]:
    """Load DNA sequences and embeddings from separate train/val/test HDF5 files.

    HDF5 format: Contains 'sequences' (variable-length string dataset) and
    'embeddings' (variable-length float dataset with per-nucleotide embeddings).

    This function uses HDF5 for true lazy loading with memory mapping.
    Data is loaded on-the-fly when accessed via dataset __getitem__.

    Parameters
    ----------
    cfg : DictConfig
        Data configuration containing paths for train/val/test HDF5 files and embedding dimension.

    Returns
    -------
    Tuple[Dict[str, h5py.File], Dict[str, int], Dict[str, float]]
        - data_dict: dictionary with keys 'train', 'val', 'test' containing HDF5 file handles
        - counts_dict: dictionary with keys 'train', 'val', 'test' containing sample counts
        - train_stats: dictionary containing training set normalization statistics
          (with keys 'min', 'max', 'mean', 'std') to be used for all splits
    """
    data_dict = {}
    counts_dict = {}
    train_stats = {}

    for split in ["train", "val", "test"]:
        csv_key = f"{split}_csv"
        sha_key = f"{split}_sha256"

        path = hy_utils.to_absolute_path(cfg[csv_key])
        assert path.endswith(".h5") or path.endswith(
            ".hdf5"
        ), f"Expected HDF5 file for {split}, got {path}"

        # Skip sha256 check if configured to do so (key is optional)
        if not cfg.skip_sha256_check:
            assert file_sha256(path) == cfg[sha_key], (
                f"SHA256 mismatch for {split} data file. "
                "Expected configured hash; data file may be corrupted or changed."
            )

        # Load HDF5 file in read-only mode with SWMR for concurrent access
        h5_file = h5py.File(path, "r", swmr=True)

        assert "sequences" in h5_file, f"Key 'sequences' missing from {split} HDF5 file"
        assert "embeddings" in h5_file, f"Key 'embeddings' missing from {split} HDF5 file"

        # Validate first sample to ensure correct format
        first_emb_flat = np.asarray(h5_file["embeddings"][0], dtype=np.float32)
        assert first_emb_flat.ndim == 1, f"Expected 1D embedding array in {split}"

        # Check if data is mean-pooled or per-nucleotide based on config
        if cfg.mean:
            # Mean embeddings: should be exactly embedding_dim length
            assert len(first_emb_flat) == cfg.embedding_dim, (
                f"Expected mean embedding of length {cfg.embedding_dim} in {split}, "
                f"got {len(first_emb_flat)}"
            )
        else:
            # Per-nucleotide embeddings: should be seq_length * embedding_dim
            seq_length = len(first_emb_flat) // cfg.embedding_dim
            first_emb = first_emb_flat.reshape(seq_length, cfg.embedding_dim)
            assert first_emb.shape[1] == cfg.embedding_dim, (
                f"Expected embedding_dim={cfg.embedding_dim} in {split}, "
                f"got {first_emb.shape[1]} for first sample"
            )

        data_dict[split] = h5_file
        counts_dict[split] = len(h5_file["embeddings"])

        # Load normalization statistics only from training set to avoid data leakage
        if split == "train" and "emb_min" in h5_file.attrs:
            # All statistics should be scalars representing global values across all embeddings
            def to_scalar(value) -> float:
                if isinstance(value, np.ndarray):
                    assert value.size == 1, (
                        f"Expected scalar attribute, got array of size {value.size}. "
                        "Statistics should be global (computed across all embedding values), not per-dimension."
                    )
                    return float(value.item())
                return float(value)

            train_stats["min"] = to_scalar(h5_file.attrs["emb_min"])
            train_stats["max"] = to_scalar(h5_file.attrs["emb_max"])
            train_stats["mean"] = to_scalar(h5_file.attrs["emb_mean"])
            train_stats["std"] = to_scalar(h5_file.attrs["emb_std"])

    if bool(cfg.get("require_sequence_disjoint", False)):
        source_path = hy_utils.to_absolute_path(cfg.source_corpus)
        length = int(cfg.seq_length)
        with h5py.File(source_path, "r", swmr=True) as source:
            assert str(source.attrs["schema"]) == str(cfg.source_schema)
            assert bool(source.attrs["sequence_content_split_disjoint"])
            expected_digest = str(
                source[f"lengths/{length}"].attrs["ordered_sequence_sha256"]
            )
        for split in ("train", "val"):
            _validate_embedding_input_attrs(
                data_dict[split],
                expected_digest=expected_digest,
                expected_length=length,
            )
        if "1000g" not in os.path.basename(data_dict["test"].filename):
            _validate_embedding_input_attrs(
                data_dict["test"],
                expected_digest=expected_digest,
                expected_length=length,
            )
        else:
            ood_source_path = hy_utils.to_absolute_path(cfg.ood_source_corpus)
            with h5py.File(ood_source_path, "r", swmr=True) as ood_source:
                assert str(ood_source.attrs["schema"]) == (
                    "dna_inversion_1000g_multilen_v2"
                )
                ood_digest = str(
                    ood_source[f"lengths/{length}"].attrs[
                        "ordered_sequence_sha256"
                    ]
                )
            _validate_embedding_input_attrs(
                data_dict["test"],
                expected_digest=ood_digest,
                expected_length=length,
            )

    return data_dict, counts_dict, train_stats


def create_dataset(
    data: h5py.File,
    mode: str,
    tokenizer: BaseTokenizer,
    embedding_dim: int,
    seq_length: int | None = None,
    normalization_stats: Dict[str, float] | None = None,
    normalization_method: str = "standard",
    data_is_mean: bool = False,
    subset_fraction: float | None = None,
    max_samples: int | None = None,
) -> Dataset:
    """Create dataset based on mode with lazy loading via HDF5.

    Parameters
    ----------
    data : h5py.File
        HDF5 file handle containing embeddings and sequences.
    mode : str
        Either "per_token" or "mean".
    tokenizer : BaseTokenizer
        Tokenizer instance to encode sequences.
    embedding_dim : int
        Expected embedding dimension for validation.
    seq_length : int | None
        Sequence length to use. If specified, sequences and embeddings will be truncated.
    normalization_stats : Dict[str, float] | None
        Training set normalization statistics to apply. Should contain keys 'min', 'max', 'mean', 'std'.
        All splits (train, val, test) should use the same training statistics to avoid data leakage.
    normalization_method : str
        Normalization method: 'standard' (z-score) or 'minmax' (0-1 range).
    data_is_mean : bool
        If True, embeddings are already mean-pooled. If False, they are per-nucleotide.
    subset_fraction : float | None
        Fraction of data to use (0.0 to 1.0). If None, use all data.
    max_samples : int | None
        Maximum number of samples to use. If provided, overrides subset_fraction.

    Returns
    -------
    Dataset
        Either DNAEmbeddingDataset or DNAMeanEmbeddingDataset, optionally wrapped in Subset.
        All use lazy loading - data is loaded from disk on-the-fly during __getitem__.
    """
    assert mode in ["per_token", "mean"], f"Invalid mode: {mode}"

    # Calculate max_samples if subset_fraction is provided
    if max_samples is None and subset_fraction is not None:
        logger = logging.getLogger(__name__)
        total_size = len(data["embeddings"])
        max_samples = int(total_size * subset_fraction)
        max_samples = max(1, max_samples)  # Ensure at least 1 sample

        logger.info(
            f"Subsetting data: {max_samples} samples "
            f"({subset_fraction * 100:.1f}% of {total_size})"
        )
    elif max_samples is not None:
        logger = logging.getLogger(__name__)
        total_size = len(data["embeddings"])
        logger.info(f"Using max_samples: {max_samples} (total available: {total_size})")

    if mode == "per_token":
        dataset = DNAEmbeddingDataset(
            data,
            tokenizer,
            embedding_dim,
            seq_length,
            normalization_stats,
            normalization_method,
            max_samples=max_samples,
        )
    else:
        assert mode == "mean"
        dataset = DNAMeanEmbeddingDataset(
            data,
            tokenizer,
            embedding_dim,
            seq_length,
            normalization_stats,
            normalization_method,
            data_is_precomputed=data_is_mean,
            max_samples=max_samples,
        )

    return dataset


def load_multi_split_embeddings(
    cfg: DictConfig,
) -> Tuple[Dict[str, List[h5py.File]], Dict[str, int], Dict[str, float]]:
    """Load multiple per-length HDF5 files for multi-length training/eval.

    Expects ``cfg`` to contain ``train_csvs``, ``val_csvs``, ``test_csvs`` (lists of paths)
    and ``seq_lengths`` (list of the corresponding sequence lengths, same order).

    ``val_csvs``/``test_csvs`` align 1:1 with ``seq_lengths`` (one file per length).
    ``train_csvs`` may instead carry EXTRA per-length shards appended after the
    base files (to add training data only at chosen lengths without re-embedding
    the existing data); its per-entry lengths are then given by
    ``train_seq_lengths``. The base train files must stay at indices
    0..len(seq_lengths)-1 so per-length eval (which indexes ``train_csvs`` by the
    position of a length in ``seq_lengths``) keeps pointing at the right file.

    Training normalization statistics are pooled across all training files using
    the sample-count-weighted mean and the pooled variance formula
    ``var = sum_i n_i (sigma_i^2 + (mu_i - mu)^2) / sum_i n_i``.

    Returns
    -------
    Tuple[Dict[str, List[h5py.File]], Dict[str, int], Dict[str, float]]
        data_dict: per-split lists of HDF5 file handles, aligned with the matching
        per-split lengths (train -> train_seq_lengths, val/test -> seq_lengths).
        counts_dict: total sample counts per split (summed across files).
        train_stats: pooled training statistics with keys 'min', 'max', 'mean', 'std'.
    """
    seq_lengths = list(cfg.seq_lengths)
    assert len(seq_lengths) > 0, "Multi mode requires non-empty seq_lengths"
    # Train shards may include extra per-length data beyond the base set; their
    # lengths are listed in train_seq_lengths (which equals seq_lengths when no
    # extra shards are configured).
    train_seq_lengths = list(cfg.train_seq_lengths)

    data_dict: Dict[str, List[h5py.File]] = {}
    counts_dict: Dict[str, int] = {}

    split_lengths = {
        "train": train_seq_lengths,
        "val": seq_lengths,
        "test": seq_lengths,
    }
    for split in ["train", "val", "test"]:
        csvs_key = f"{split}_csvs"
        paths = list(cfg[csvs_key])
        expected_lengths = split_lengths[split]
        assert len(paths) == len(expected_lengths), (
            f"Length mismatch in multi cfg: {csvs_key} has {len(paths)} entries "
            f"but the matching length list has {len(expected_lengths)}"
        )

        files: List[h5py.File] = []
        total = 0
        for csv_path in paths:
            path = hy_utils.to_absolute_path(csv_path)
            assert path.endswith(".h5") or path.endswith(
                ".hdf5"
            ), f"Expected HDF5 file for {split}, got {path}"

            h5_file = h5py.File(path, "r", swmr=True)
            assert "sequences" in h5_file, f"Key 'sequences' missing in {path}"
            assert "embeddings" in h5_file, f"Key 'embeddings' missing in {path}"

            first_emb_flat = np.asarray(h5_file["embeddings"][0], dtype=np.float32)
            assert first_emb_flat.ndim == 1, f"Expected 1D embedding array in {path}"
            if cfg.mean:
                assert len(first_emb_flat) == cfg.embedding_dim, (
                    f"Expected mean embedding of length {cfg.embedding_dim} in {path}, "
                    f"got {len(first_emb_flat)}"
                )

            files.append(h5_file)
            total += len(h5_file["embeddings"])

        data_dict[split] = files
        counts_dict[split] = total

    train_stats = compute_pooled_train_stats(data_dict["train"])
    _validate_multilen_release_source(cfg, data_dict)

    return data_dict, counts_dict, train_stats


def create_multi_dataset(
    data_files: List[h5py.File],
    seq_lengths: List[int],
    mode: str,
    tokenizer: BaseTokenizer,
    embedding_dim: int,
    normalization_stats: Dict[str, float] | None = None,
    normalization_method: str = "standard",
    data_is_mean: bool = False,
    subset_fraction: float | None = None,
    max_samples: int | None = None,
) -> Dataset:
    """Concatenate per-length datasets into a single multi-length dataset.

    All datasets share the same tokenizer, embedding_dim, and normalization
    statistics. Each sub-dataset is built with its own ``seq_length`` so
    sequences/embeddings are truncated/tokenized correctly per file.
    """
    assert len(data_files) == len(seq_lengths), (
        f"Multi dataset mismatch: {len(data_files)} files vs {len(seq_lengths)} seq_lengths"
    )
    sub_datasets = []
    for h5, seq_len in zip(data_files, seq_lengths):
        ds = create_dataset(
            data=h5,
            mode=mode,
            tokenizer=tokenizer,
            embedding_dim=embedding_dim,
            seq_length=seq_len,
            normalization_stats=normalization_stats,
            normalization_method=normalization_method,
            data_is_mean=data_is_mean,
            subset_fraction=subset_fraction,
            max_samples=max_samples,
        )
        sub_datasets.append(ds)
    return ConcatDataset(sub_datasets)
