"""Resumable, incremental writing of split (train/val/test) embedding HDF5 files.

The per-length mean-embedding generators (DNABERT-2 / Evo2 / NTv2) embed up to
~1M sequences per length per model -- long enough that a SLURM time limit may
interrupt them. This module persists embeddings incrementally so a requeued job
loads what already exists, counts how many samples are missing, and continues
from there instead of re-embedding from scratch.

Design
------
* The train/val/test assignment for every input sequence is precomputed
  deterministically from ``seed`` (a fixed permutation), so a resumed run
  reproduces the exact same split.
* Sequences are processed in input order. After a prefix of ``p`` sequences has
  been written, the three files together hold exactly ``p`` rows, so the resume
  point is simply the total row count across the splits.
* Datasets are created resizable (``maxshape=(None, ...)``) and flushed every
  chunk, so a crash loses at most one in-flight chunk.
* Global training statistics (over the train split) are accumulated as rows are
  written -- seeded from the existing train rows on resume -- and stamped onto
  all three files at completion together with a ``generation_complete`` marker.
  A file is "complete" iff that marker is set (or, for legacy files written
  before this module, iff ``emb_mean`` is present and no marker exists).
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Callable, Dict, List

import h5py
import hydra.utils as hy_utils
import numpy as np

from src.utils import file_sha256, update_yaml_keys

SPLITS = ("train", "val", "test")


def ordered_sequence_sha256(sequences: List[str]) -> str:
    """Hash an ordered DNA-sequence list using the source-corpus convention."""
    digest = hashlib.sha256()
    for sequence in sequences:
        digest.update(sequence.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def ordered_identification_sha256(
    sequences: List[str],
    individual_ids: List[str],
    locus_ids: List[int],
    haplotypes: List[int],
) -> str:
    """Hash all ordered identification inputs, including aligned row labels."""
    digest = hashlib.sha256()
    for sequence, individual_id, locus_id, haplotype in zip(
        sequences, individual_ids, locus_ids, haplotypes, strict=True
    ):
        digest.update(sequence.encode("ascii"))
        digest.update(b"\t")
        digest.update(individual_id.encode("utf-8"))
        digest.update(b"\t")
        digest.update(str(locus_id).encode("ascii"))
        digest.update(b"\t")
        digest.update(str(haplotype).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def identification_csv_input_sha256(path: str) -> str:
    """Stream the identification-input digest directly from its release CSV."""
    digest = hashlib.sha256()
    with open(path, "r", encoding="utf-8") as handle:
        assert handle.readline().strip() == (
            "individual_id,locus_id,haplotype,sequence"
        )
        for line in handle:
            line = line.strip()
            if not line:
                continue
            individual_id, locus_id, haplotype, sequence = line.split(",")
            digest.update(sequence.encode("ascii"))
            digest.update(b"\t")
            digest.update(individual_id.encode("utf-8"))
            digest.update(b"\t")
            digest.update(locus_id.encode("ascii"))
            digest.update(b"\t")
            digest.update(haplotype.encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def deterministic_split_labels(
    n: int,
    train_split: float,
    val_split: float,
    seed: int,
    *,
    val_size: int | None = None,
    test_size: int | None = None,
) -> np.ndarray:
    """Assign each input index to 'train'/'val'/'test' reproducibly from ``seed``.

    A fixed permutation routes the first ``train_n`` shuffled positions to train,
    the next ``val_n`` to val, and the rest to test -- mirroring the original
    generators' shuffle-then-slice split, but as a per-index label array so the
    assignment can be reproduced exactly on resume.

    Two sizing modes:
      * fractional (default): ``train_split`` / ``val_split`` of ``n`` go to train
        / val, the remainder to test.
      * fixed-count (``val_size``/``test_size`` set): exactly that many rows go to
        val / test and ALL the rest to train. This keeps val/test at a constant
        size across sequence lengths while only the train size grows with length,
        so a single generation pass per length replaces the old base+ext merge.
    """
    assert n >= 0, "n must be non-negative"
    if val_size is not None:
        assert test_size is not None, "val_size and test_size must be set together"
        val_n = min(int(val_size), n)
        test_n = min(int(test_size), n - val_n)
        train_n = n - val_n - test_n
    else:
        train_n = int(n * train_split)
        val_n = int(n * val_split)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    labels = np.empty(n, dtype=object)
    labels[perm[:train_n]] = "train"
    labels[perm[train_n : train_n + val_n]] = "val"
    labels[perm[train_n + val_n :]] = "test"
    return labels


def mean_pool_in_batches(
    seqs: List[str], batch_size: int, embed_batch: Callable[[List[str]], List[np.ndarray]]
) -> List[np.ndarray]:
    """Mean-pool each sequence's token embeddings, computing ``embed_batch`` in sub-batches.

    ``embed_batch`` maps a list of sequences to a list of ``[num_tokens, D]`` arrays;
    each is reduced to ``[D]``. Shared by the Evo2 / NTv2 mean generators, which
    differ only in the underlying batched-embedding call.
    """
    out: List[np.ndarray] = []
    for start in range(0, len(seqs), batch_size):
        out.extend(emb.mean(axis=0) for emb in embed_batch(seqs[start : start + batch_size]))
    return out


class _TrainStats:
    """Streaming global statistics over train-split embedding values."""

    def __init__(self) -> None:
        self.n = 0  # number of scalar values seen
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = np.inf
        self.max = -np.inf

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        v = values.astype(np.float64, copy=False).ravel()
        self.n += v.size
        self.sum += float(v.sum())
        self.sumsq += float(np.square(v).sum())
        self.min = min(self.min, float(v.min()))
        self.max = max(self.max, float(v.max()))

    def finalize(self) -> Dict[str, float]:
        assert self.n > 0, "No train-split values accumulated; cannot compute stats"
        mean = self.sum / self.n
        var = max(self.sumsq / self.n - mean * mean, 0.0)
        return {"min": self.min, "max": self.max, "mean": mean, "std": var ** 0.5}


def _file_is_complete(path: str, legacy_key: str) -> bool:
    """True if ``path`` holds a finished generation (new marker or legacy attr).

    ``legacy_key`` is the attribute that pre-dates this module's
    ``generation_complete`` marker and only ever appeared on fully-written files:
    ``emb_mean`` for mean-split files, ``identification_mode`` for id cohort files.
    """
    with h5py.File(path, "r") as f:
        if "generation_complete" in f.attrs:
            return bool(f.attrs["generation_complete"])
        # Legacy files were only ever flushed once, fully, so presence of the
        # legacy key implies completeness.
        return legacy_key in f.attrs


def generation_complete(
    train_path: str, expected_input_sha256: str | None = None
) -> bool:
    """Whether the anchor output is complete and belongs to the current input."""
    if not os.path.exists(train_path) or not _file_is_complete(train_path, "emb_mean"):
        return False
    if expected_input_sha256 is None:
        return True
    with h5py.File(train_path, "r") as f:
        assert "input_sequence_sha256" in f.attrs, (
            f"Complete embedding file has no input provenance and cannot be safely "
            f"reused: {train_path}. Use a new versioned output path."
        )
        observed = str(f.attrs["input_sequence_sha256"])
    assert observed == expected_input_sha256, (
        f"Embedding input changed for {train_path}: expected "
        f"{expected_input_sha256}, found {observed}. Use a new output path."
    )
    return True


def identification_complete(
    path: str, expected_input_sha256: str | None = None
) -> bool:
    """Whether an identification H5 is complete and belongs to current rows."""
    if not os.path.exists(path) or not _file_is_complete(path, "identification_mode"):
        return False
    if expected_input_sha256 is None:
        return True
    with h5py.File(path, "r") as f:
        assert "identification_input_sha256" in f.attrs, (
            f"Complete identification file has no input provenance and cannot be "
            f"safely reused: {path}. Use a new versioned output path."
        )
        observed = str(f.attrs["identification_input_sha256"])
    assert observed == expected_input_sha256, (
        f"Identification input changed for {path}: expected "
        f"{expected_input_sha256}, found {observed}. Use a new output path."
    )
    return True


def _stamp_or_validate_input_provenance(
    f: "h5py.File", *, digest: str, num_sequences: int, seq_length: int
) -> None:
    """Bind a new/partial embedding file to exactly one ordered input corpus."""
    if "input_sequence_sha256" in f.attrs:
        assert str(f.attrs["input_sequence_sha256"]) == digest, (
            f"Input provenance mismatch for {f.filename}; use a new output path"
        )
        assert int(f.attrs["input_num_sequences"]) == num_sequences
        assert int(f.attrs["seq_length"]) == seq_length
        return
    assert len(f["sequences"]) == 0, (
        f"Non-empty partial file lacks input provenance: {f.filename}. It cannot "
        "be resumed safely; use a new versioned output path."
    )
    f.attrs["input_sequence_sha256"] = digest
    f.attrs["input_num_sequences"] = num_sequences
    f.attrs["seq_length"] = seq_length


def _open_split(path: str, embedding_dim: int) -> "h5py.File":
    """Open an existing resumable split file for append, or create a fresh one.

    Datasets are resizable so rows can be appended chunk by chunk.
    """
    if os.path.exists(path):
        f = h5py.File(path, "a")
        assert "sequences" in f and "embeddings" in f, f"Corrupt partial split file: {path}"
        return f
    f = h5py.File(path, "w")
    dt_str = h5py.string_dtype(encoding="utf-8")
    f.create_dataset("sequences", shape=(0,), maxshape=(None,), dtype=dt_str)
    f.create_dataset(
        "embeddings",
        shape=(0, embedding_dim),
        maxshape=(None, embedding_dim),
        dtype=np.float32,
        chunks=(min(4096, 1 << 20), embedding_dim),
    )
    f.attrs["embedding_dim"] = embedding_dim
    f.attrs["generation_complete"] = False
    return f


def _resize_append(ds: "h5py.Dataset", values: Any) -> None:
    """Grow a resizable dataset by ``len(values)`` rows and write them at the end."""
    n0 = ds.shape[0]
    ds.resize((n0 + len(values),) + ds.shape[1:])
    ds[n0:] = values


def _append(f: "h5py.File", seqs: List[str], embs: List[np.ndarray]) -> None:
    """Append a chunk of sequences + mean embeddings to a resizable split file."""
    if not seqs:
        return
    _resize_append(f["sequences"], seqs)
    _resize_append(f["embeddings"], np.asarray(embs, dtype=np.float32))


def _seed_stats_from_existing_train(train_path: str, stats: _TrainStats, logger: logging.Logger) -> None:
    """Rebuild train statistics from rows already on disk (resume path)."""
    with h5py.File(train_path, "r") as f:
        n = len(f["embeddings"])
        if n == 0:
            return
        logger.info(f"[gen] Re-reading {n} existing train rows to seed statistics...")
        step = 50_000
        for i in range(0, n, step):
            stats.update(f["embeddings"][i : min(i + step, n)])


def generate_split_h5_resumable(
    *,
    sequences: List[str],
    embed_chunk: Callable[[List[str]], List[np.ndarray]],
    out_paths: Dict[str, str],
    embedding_dim: int,
    checkpoint: str,
    train_split: float,
    val_split: float,
    seed: int,
    chunk_size: int,
    logger: logging.Logger,
    val_size: int | None = None,
    test_size: int | None = None,
) -> Dict[str, str]:
    """Embed ``sequences`` into resumable train/val/test HDF5 files.

    Only the missing suffix is embedded: existing rows are counted up front and
    used as the resume point. ``embed_chunk`` maps a list of sequences to a list
    of mean-pooled ``[embedding_dim]`` arrays (the caller owns model loading and
    GPU batching). Returns the per-split SHA256 hashes computed at completion.

    When ``val_size``/``test_size`` are given the split is sized by fixed counts
    (constant val/test, variable train) instead of the ``train_split``/``val_split``
    fractions -- see ``deterministic_split_labels``.
    """
    for split in SPLITS:
        assert split in out_paths, f"out_paths missing '{split}'"
    n = len(sequences)
    assert n > 0, "Cannot generate embeddings for an empty sequence list"
    seq_length = len(sequences[0])
    assert all(len(sequence) == seq_length for sequence in sequences)
    input_digest = ordered_sequence_sha256(sequences)
    labels = deterministic_split_labels(
        n, train_split, val_split, seed, val_size=val_size, test_size=test_size
    )

    # Open one handle per UNIQUE path: a train-only run (train_split=1.0) may point
    # val and test at the same throwaway file, and two write-handles on one file
    # would corrupt it.
    handles_by_path: Dict[str, "h5py.File"] = {}
    for split in SPLITS:
        p = out_paths[split]
        if p not in handles_by_path:
            handles_by_path[p] = _open_split(p, embedding_dim)
            _stamp_or_validate_input_provenance(
                handles_by_path[p],
                digest=input_digest,
                num_sequences=n,
                seq_length=seq_length,
            )
    files = {split: handles_by_path[out_paths[split]] for split in SPLITS}
    try:
        n_done = sum(len(h["sequences"]) for h in handles_by_path.values())
        assert n_done <= n, (
            f"On-disk rows ({n_done}) exceed the {n} input sequences; the inputs "
            f"or seed changed since the partial files were written. Delete "
            f"{list(out_paths.values())} to regenerate."
        )
        if n_done == n:
            logger.info(f"[gen] All {n} samples already present; finalizing only.")
        else:
            logger.info(f"[gen] Resuming generation: {n_done}/{n} present, {n - n_done} to embed.")

        stats = _TrainStats()
        if n_done > 0:
            _seed_stats_from_existing_train(out_paths["train"], stats, logger)

        for start in range(n_done, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_seqs = sequences[start:end]
            chunk_embs = embed_chunk(chunk_seqs)
            assert len(chunk_embs) == len(chunk_seqs), (
                f"embed_chunk returned {len(chunk_embs)} for {len(chunk_seqs)} sequences"
            )
            chunk_labels = labels[start:end]
            for split in SPLITS:
                sel = [i for i in range(len(chunk_seqs)) if chunk_labels[i] == split]
                if not sel:
                    continue
                s_seqs = [chunk_seqs[i] for i in sel]
                s_embs = [chunk_embs[i] for i in sel]
                _append(files[split], s_seqs, s_embs)
                if split == "train":
                    stats.update(np.asarray(s_embs, dtype=np.float32))
            for f in files.values():
                f.flush()
            logger.info(f"[gen] Wrote {end}/{n} samples.")

        # Finalize: stamp global train stats + completion marker on every split.
        train_stats = stats.finalize()
        logger.info(
            f"[gen] Train stats: min={train_stats['min']:.4f} max={train_stats['max']:.4f} "
            f"mean={train_stats['mean']:.4f} std={train_stats['std']:.4f}"
        )
        for f in handles_by_path.values():
            f.attrs["embedding_dim"] = embedding_dim
            f.attrs["checkpoint"] = checkpoint
            f.attrs["emb_min"] = train_stats["min"]
            f.attrs["emb_max"] = train_stats["max"]
            f.attrs["emb_mean"] = train_stats["mean"]
            f.attrs["emb_std"] = train_stats["std"]
            f.attrs["generation_complete"] = True
    finally:
        for f in handles_by_path.values():
            f.close()

    hashes = {split: file_sha256(out_paths[split]) for split in SPLITS}
    for split in SPLITS:
        with h5py.File(out_paths[split], "r") as f:
            n_rows = len(f["sequences"])
        logger.info(f"[gen] {split}: {n_rows} rows, sha256={hashes[split]}")
    return hashes


def _open_identification(path: str, embedding_dim: int) -> "h5py.File":
    """Open an existing resumable identification file for append, or create one.

    Holds the same sequences/embeddings as a split file plus the three per-row
    label datasets the identification eval consumes, all resizable.
    """
    if os.path.exists(path):
        f = h5py.File(path, "a")
        for key in ("sequences", "embeddings", "individual_ids", "locus_ids", "haplotypes"):
            assert key in f, f"Corrupt partial identification file (missing '{key}'): {path}"
        return f
    f = h5py.File(path, "w")
    dt_str = h5py.string_dtype(encoding="utf-8")
    f.create_dataset("sequences", shape=(0,), maxshape=(None,), dtype=dt_str)
    f.create_dataset(
        "embeddings",
        shape=(0, embedding_dim),
        maxshape=(None, embedding_dim),
        dtype=np.float32,
        chunks=(min(4096, 1 << 20), embedding_dim),
    )
    f.create_dataset("individual_ids", shape=(0,), maxshape=(None,), dtype=dt_str)
    f.create_dataset("locus_ids", shape=(0,), maxshape=(None,), dtype=np.int32)
    f.create_dataset("haplotypes", shape=(0,), maxshape=(None,), dtype=np.int8)
    f.attrs["embedding_dim"] = embedding_dim
    f.attrs["generation_complete"] = False
    return f


def generate_identification_h5_resumable(
    *,
    sequences: List[str],
    individual_ids: List[str],
    locus_ids: List[int],
    haplotypes: List[int],
    embed_chunk: Callable[[List[str]], List[np.ndarray]],
    out_path: str,
    embedding_dim: int,
    checkpoint: str,
    chunk_size: int,
    logger: logging.Logger,
) -> str:
    """Embed a per-individual identification cohort into a resumable HDF5 file.

    Rows are written in input order (no shuffle, no split, no train stats) with
    their aligned ``(individual_id, locus_id, haplotype)`` labels. Only the
    missing suffix is embedded on resume. Returns the file's SHA256.
    """
    n = len(sequences)
    assert len(individual_ids) == n and len(locus_ids) == n and len(haplotypes) == n, (
        "Identification labels must align 1:1 with sequences"
    )
    assert n > 0, "Cannot generate identification embeddings for an empty cohort"
    seq_length = len(sequences[0])
    assert all(len(sequence) == seq_length for sequence in sequences)
    input_digest = ordered_identification_sha256(
        sequences, individual_ids, locus_ids, haplotypes
    )

    f = _open_identification(out_path, embedding_dim)
    try:
        if "identification_input_sha256" in f.attrs:
            assert str(f.attrs["identification_input_sha256"]) == input_digest, (
                f"Identification input provenance mismatch for {out_path}; use a "
                "new output path"
            )
            assert int(f.attrs["input_num_sequences"]) == n
            assert int(f.attrs["seq_length"]) == seq_length
        else:
            assert len(f["sequences"]) == 0, (
                f"Non-empty identification file lacks input provenance: {out_path}. "
                "Use a new versioned output path."
            )
            f.attrs["identification_input_sha256"] = input_digest
            f.attrs["input_num_sequences"] = n
            f.attrs["seq_length"] = seq_length
        n_done = len(f["sequences"])
        assert n_done <= n, (
            f"On-disk rows ({n_done}) exceed the {n} input sequences; inputs changed. "
            f"Delete {out_path} to regenerate."
        )
        if n_done == n:
            logger.info(f"[gen-id] All {n} rows already present; finalizing only.")
        else:
            logger.info(f"[gen-id] Resuming: {n_done}/{n} present, {n - n_done} to embed.")

        for start in range(n_done, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_seqs = sequences[start:end]
            chunk_embs = embed_chunk(chunk_seqs)
            assert len(chunk_embs) == len(chunk_seqs), (
                f"embed_chunk returned {len(chunk_embs)} for {len(chunk_seqs)} sequences"
            )
            _resize_append(f["sequences"], chunk_seqs)
            _resize_append(f["embeddings"], np.asarray(chunk_embs, dtype=np.float32))
            _resize_append(f["individual_ids"], individual_ids[start:end])
            _resize_append(f["locus_ids"], np.asarray(locus_ids[start:end], dtype=np.int32))
            _resize_append(f["haplotypes"], np.asarray(haplotypes[start:end], dtype=np.int8))
            f.flush()
            logger.info(f"[gen-id] Wrote {end}/{n} rows.")

        f.attrs["embedding_dim"] = embedding_dim
        f.attrs["checkpoint"] = checkpoint
        f.attrs["identification_mode"] = True
        f.attrs["generation_complete"] = True
    finally:
        f.close()

    sha = file_sha256(out_path)
    logger.info(f"[gen-id] {out_path}: {n} rows, sha256={sha}")
    return sha


def run_generation(
    cfg: Any,
    *,
    sequences: List[str],
    individual_ids,
    locus_ids,
    haplotypes,
    make_embed_chunk: Callable[[], Callable[[List[str]], List[np.ndarray]]],
    logger: logging.Logger,
) -> bool:
    """Run the resumable mean / identification generation when applicable.

    Shared across the three embedding generators. ``make_embed_chunk()`` loads the
    model (once) and returns ``embed_chunk``; it is only invoked when there is work
    to do, so an already-complete output is a cheap no-op (no model load). The
    embedding dimension is probed from the first sequence here.

    Returns True if generation was handled here (the caller should return), or
    False for the legacy all-at-once per-token (``mean=false``) case, which the
    caller must still handle itself.
    """
    identification_mode = bool(cfg.identification_mode)
    use_mean = bool(cfg.mean)
    checkpoint = cfg.checkpoint
    chunk_size = int(cfg.gen_chunk_size)

    if identification_mode:
        out_path = hy_utils.to_absolute_path(cfg.test_output_path)
        input_digest = ordered_identification_sha256(
            sequences, individual_ids, locus_ids, haplotypes
        )
        if identification_complete(out_path, input_digest):
            logger.info(f"[gen] Identification output already complete: {out_path}; skipping.")
            return True
        assert use_mean, "Identification generation requires mean=true"
        embed_chunk = make_embed_chunk()
        dim = int(embed_chunk([sequences[0]])[0].shape[-1])
        generate_identification_h5_resumable(
            sequences=sequences,
            individual_ids=individual_ids,
            locus_ids=locus_ids,
            haplotypes=haplotypes,
            embed_chunk=embed_chunk,
            out_path=out_path,
            embedding_dim=dim,
            checkpoint=checkpoint,
            chunk_size=chunk_size,
            logger=logger,
        )
        return True

    if use_mean:
        out_paths = {sp: hy_utils.to_absolute_path(cfg[f"{sp}_output_path"]) for sp in SPLITS}
        input_digest = ordered_sequence_sha256(sequences)
        if generation_complete(out_paths["train"], input_digest):
            logger.info(f"[gen] Mean output already complete: {out_paths['train']}; skipping.")
            return True
        embed_chunk = make_embed_chunk()
        dim = int(embed_chunk([sequences[0]])[0].shape[-1])
        # Fixed-count split (constant val/test, variable train) when val_size is
        # configured; otherwise the train_split/val_split fractions.
        val_size = None if cfg.val_size is None else int(cfg.val_size)
        test_size = None if cfg.test_size is None else int(cfg.test_size)
        hashes = generate_split_h5_resumable(
            sequences=sequences,
            embed_chunk=embed_chunk,
            out_paths=out_paths,
            embedding_dim=dim,
            checkpoint=checkpoint,
            train_split=float(cfg.train_split),
            val_split=float(cfg.val_split),
            seed=int(cfg.seed),
            chunk_size=chunk_size,
            logger=logger,
            val_size=val_size,
            test_size=test_size,
        )
        if cfg.update_config:
            config_path = hy_utils.to_absolute_path(cfg.update_config)
            logger.info(f"[gen] Updating config file {config_path}...")
            update_yaml_keys(
                str(config_path),
                {
                    "train_sha256": hashes["train"],
                    "val_sha256": hashes["val"],
                    "test_sha256": hashes["test"],
                    "skip_sha256_check": False,
                    "train_csv": cfg.train_output_path,
                    "val_csv": cfg.val_output_path,
                    "test_csv": cfg.test_output_path,
                    "embedding_dim": dim,
                    "seq_length": cfg.seq_length,
                },
            )
        return True

    return False
